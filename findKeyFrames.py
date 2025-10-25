#!/usr/bin/env python3
#python extract_keyframes_kmeans.py --video test_video.mp4 --outdir keyframes_out
import os
import json
import argparse
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# -------------------------
# Helper: 3D angle calculation
# -------------------------
def angle_3d(a, b, c):
    """Return angle at b formed by a-b-c using 3D coords that have .x,.y,.z"""
    a = np.array([a.x, a.y, a.z], dtype=float)
    b = np.array([b.x, b.y, b.z], dtype=float)
    c = np.array([c.x, c.y, c.z], dtype=float)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return 0.0
    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    # prefer interior angle
    if ang > 180:
        ang = 360 - ang
    return ang

# -------------------------
# Compute chosen joint angles from MediaPipe world landmarks
# Indices follow mp.solutions.pose.PoseLandmark
# -------------------------
def compute_angles_from_world_landmarks(lm):
    """
    lm: list-like of world landmarks (landmark objects with x,y,z)
    returns: dict of named angles (float)
    """
    # MediaPipe indices
    # left: shoulder=11, elbow=13, wrist=15, hip=23, knee=25, ankle=27
    # right: shoulder=12, elbow=14, wrist=16, hip=24, knee=26, ankle=28
    angles = {}
    try:
        angles["L_Elbow"] = angle_3d(lm[11], lm[13], lm[15])
        angles["R_Elbow"] = angle_3d(lm[12], lm[14], lm[16])
        # Shoulder angle: hip-shoulder-elbow measures torso->upper-arm
        angles["L_Shoulder"] = angle_3d(lm[23], lm[11], lm[13])
        angles["R_Shoulder"] = angle_3d(lm[24], lm[12], lm[14])
        # Hips/knees
        angles["L_Knee"] = angle_3d(lm[23], lm[25], lm[27])
        angles["R_Knee"] = angle_3d(lm[24], lm[26], lm[28])
        # Torso bend: shoulder-hip-knee
        angles["Torso"] = angle_3d(lm[11], lm[23], lm[25])
        # Optionally add more (wrists, head tilt) if needed
    except Exception:
        # In case any landmark index missing or degenerate
        # Return zeros to keep consistent vector size
        default = {k: 0.0 for k in ["L_Elbow","R_Elbow","L_Shoulder","R_Shoulder","L_Knee","R_Knee","Torso"]}
        return default
    return angles

# -------------------------
# Core pipeline
# -------------------------
def extract_keyframes_kmeans(video_path, out_dir,
                             min_visibility=0.45,
                             smoothing_window=5,
                             pca_components=6,
                             random_state=42):
    """
    Process video, compute 3D angles, cluster frames into 3 phases, return representative frames.

    Returns dict summary with:
      - 'key_frames': list of dicts {phase, frame_idx, saved_image, angles}
      - 'angles_npy': path to saved numpy of all angles (frames x features)
    """
    os.makedirs(out_dir, exist_ok=True)
    # MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    world_landmarks = []   # list of lists of landmarks
    frame_indices = []     # original frame numbers (useful for mapping)

    frame_count = 0
    print("Extracting poses from video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        # require world landmarks and check visibility of key landmarks
        if results.pose_world_landmarks:
            # Check visibility for a few key landmarks to avoid occlusions
            vis_ok = True
            sample_idxs = [11, 13, 23, 25]  # L_shoulder, L_elbow, L_hip, L_knee (representative)
            for si in sample_idxs:
                try:
                    v = results.pose_landmarks.landmark[si].visibility
                    if v < min_visibility:
                        vis_ok = False
                        break
                except Exception:
                    vis_ok = False
                    break
            if vis_ok:
                frames.append(frame.copy())
                world_landmarks.append(results.pose_world_landmarks.landmark)
                frame_indices.append(frame_count)
        frame_count += 1

    cap.release()
    pose.close()

    if len(world_landmarks) < 10:
        # Too few valid frames — fallback: use raw frames w/o filtering or raise
        raise RuntimeError("Too few valid frames with high-visibility landmarks. Try lowering --min_visibility or using better video.")

    # -------------------------
    # Compute angles per frame
    # -------------------------
    print(f"Computing angles for {len(world_landmarks)} frames...")
    angle_names = ["L_Elbow","R_Elbow","L_Shoulder","R_Shoulder","L_Knee","R_Knee","Torso"]
    angle_matrix = np.zeros((len(world_landmarks), len(angle_names)), dtype=float)

    for i, lm in enumerate(world_landmarks):
        angles = compute_angles_from_world_landmarks(lm)
        angle_matrix[i, :] = np.array([angles[name] for name in angle_names])

    # -------------------------
    # Smooth each angle time-series
    # -------------------------
    if smoothing_window > 1:
        smoothed = np.copy(angle_matrix)
        for col in range(smoothed.shape[1]):
            smoothed[:, col] = np.convolve(angle_matrix[:, col], np.ones(smoothing_window)/smoothing_window, mode='same')
    else:
        smoothed = angle_matrix

    # -------------------------
    # Compute temporal deltas (motion)
    # -------------------------
    # Use absolute frame-to-frame change of smoothed angles as additional features
    deltas = np.zeros_like(smoothed)
    deltas[1:, :] = np.abs(smoothed[1:, :] - smoothed[:-1, :])
    deltas[0, :] = deltas[1, :]  # copy first diff

    # Combine features: angles + deltas
    features = np.hstack([smoothed, deltas])  # shape (N, 14)

    # -------------------------
    # Standardize and optionally PCA
    # -------------------------
    scaler = StandardScaler()
    Xs = scaler.fit_transform(features)

    if pca_components and pca_components > 0 and pca_components < Xs.shape[1]:
        pca = PCA(n_components=pca_components, random_state=random_state)
        Xr = pca.fit_transform(Xs)
    else:
        Xr = Xs

    # -------------------------
    # KMeans clustering (k=3)
    # -------------------------
    k = 3
    print("Clustering frames into 3 swing phases (KMeans)...")
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(Xr)
    centroids = kmeans.cluster_centers_

    # -------------------------
    # Map clusters to temporal order -> phases
    # -------------------------
    # For each cluster compute mean frame index (temporal center)
    cluster_frame_means = []
    for cluster_id in range(k):
        idxs = np.where(labels == cluster_id)[0]
        mean_frame_pos = np.mean(idxs) if len(idxs) > 0 else np.inf
        cluster_frame_means.append((cluster_id, mean_frame_pos))
    # sort ascending by mean frame position (earliest -> latest)
    cluster_frame_means.sort(key=lambda x: x[1])
    phase_names = ["Backswing", "Impact", "FollowThrough"]
    cluster_to_phase = {}
    for order, (cluster_id, _) in enumerate(cluster_frame_means):
        cluster_to_phase[cluster_id] = phase_names[order]

    # -------------------------
    # Find centroid-closest frame for each cluster (representative)
    # -------------------------
    rep_frames = []
    for cluster_id in range(k):
        members = np.where(labels == cluster_id)[0]
        if len(members) == 0:
            continue
        # compute distance in reduced feature space to centroid
        centroid = centroids[cluster_id]
        dists = np.linalg.norm(Xr[members] - centroid, axis=1)
        closest_idx_within_members = members[int(np.argmin(dists))]
        orig_frame_idx = frame_indices[closest_idx_within_members]
        saved_image_name = f"{cluster_to_phase[cluster_id]}_frame_{orig_frame_idx}.png"
        saved_image_path = os.path.join(out_dir, saved_image_name)

        # Draw landmarks & save image with label
        frame_rgb = frames[closest_idx_within_members].copy()
        # draw 2D landmarks overlay for visualization using pose estimation again
        # We'll run a quick 2D pose for overlay clarity
        mp2 = mp.solutions.pose
        pose2 = mp2.Pose(static_image_mode=True, min_detection_confidence=0.5)
        res2 = pose2.process(cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))
        if res2.pose_landmarks:
            mp_drawing.draw_landmarks(frame_rgb, res2.pose_landmarks, mp2.POSE_CONNECTIONS)
        pose2.close()
        # Put label text
        cv2.putText(frame_rgb, f"{cluster_to_phase[cluster_id]}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 4)
        cv2.imwrite(saved_image_path, frame_rgb)

        # prepare angle info for the representative frame
        rep_angles = {angle_names[i]: float(smoothed[closest_idx_within_members, i]) for i in range(len(angle_names))}
        rep_frames.append({
            "cluster_id": int(cluster_id),
            "phase": cluster_to_phase[cluster_id],
            "member_index_in_valid_frames": int(closest_idx_within_members),
            "original_frame_index_in_video": int(orig_frame_idx),
            "saved_image": saved_image_path,
            "angles": rep_angles
        })

    # -------------------------
    # Save angle matrix and summary JSON
    # -------------------------
    angles_npy_path = os.path.join(out_dir, "all_smoothed_angles.npy")
    np.save(angles_npy_path, smoothed)

    summary = {
        "video_path": video_path,
        "num_valid_frames": int(len(frames)),
        "frame_indices": frame_indices,
        "angle_names": angle_names,
        "angles_npy": angles_npy_path,
        "key_frames": rep_frames
    }
    summary_json_path = os.path.join(out_dir, "keyframe_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {len(rep_frames)} representative key frames to {out_dir}")
    print("Summary JSON:", summary_json_path)
    return summary

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Extract robust golf-swing key frames using KMeans over pose features.")
    p.add_argument("--video", "--video_path", dest="video", required=True, help="Path to swing video (single swing preferred).")
    p.add_argument("--outdir", dest="outdir", default="keyframes_out", help="Output directory to save images & summary.")
    p.add_argument("--min_visibility", type=float, default=0.45, help="Min landmark visibility to accept a frame (0-1).")
    p.add_argument("--smoothing", type=int, default=5, help="Smoothing window for angle time-series (frames).")
    p.add_argument("--pca", type=int, default=6, help="PCA components (set 0 to disable).")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    summary = extract_keyframes_kmeans(
        video_path=args.video,
        out_dir=args.outdir,
        min_visibility=args.min_visibility,
        smoothing_window=args.smoothing,
        pca_components=args.pca
    )
