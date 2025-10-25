#!/usr/bin/env python3
"""
compare_user_to_ideal.py (updated for 7-angle format)

Usage:
  python compare_user_to_ideal.py --ideal ideal_angles.npy --webcam
  python compare_user_to_ideal.py --ideal ideal_angles.npy --video user.mp4

Expects ideal .npy to be either:
 - (3, 7) -> 3 key frames (Backswing, Impact, Mid) x 7 angles
 - (T, 7) -> per-frame angles; script will pick 3 key rows automatically
"""
import argparse
import json
import time
import os
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

# ---------- CONFIG ----------
ANGLE_NAMES = ["L_Elbow","R_Elbow","L_Shoulder","R_Shoulder","L_Knee","R_Knee","Torso"]
PHASE_NAMES = ["Backswing", "Impact", "MidSwing"]
SMOOTH_WINDOW = 5
TOLERANCE_DEG = 10
MAX_DIFF_FOR_SCORE = 30.0

# ---------- HELPERS ----------
def angle_3d(a, b, c):
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
    ang = float(np.degrees(np.arccos(cosang)))
    if ang > 180:
        ang = 360 - ang
    return ang

def compute_angles_from_world_landmarks(lm):
    """
    Return angles in the same order as ANGLE_NAMES:
    L_Elbow, R_Elbow, L_Shoulder, R_Shoulder, L_Knee, R_Knee, Torso
    """
    try:
        l_elbow = angle_3d(lm[11], lm[13], lm[15])        # shoulder-elbow-wrist
        r_elbow = angle_3d(lm[12], lm[14], lm[16])
        l_shoulder = angle_3d(lm[23], lm[11], lm[13])    # hip-shoulder-elbow
        r_shoulder = angle_3d(lm[24], lm[12], lm[14])
        l_knee = angle_3d(lm[23], lm[25], lm[27])        # hip-knee-ankle
        r_knee = angle_3d(lm[24], lm[26], lm[28])
        torso = angle_3d(lm[11], lm[23], lm[25])         # shoulder-hip-knee
        return [l_elbow, r_elbow, l_shoulder, r_shoulder, l_knee, r_knee, torso]
    except Exception:
        return [0.0] * len(ANGLE_NAMES)

def pick_key_indices_from_series(angle_matrix):
    """Pick 3 key indices using left elbow column (col 0): argmax, argmin, mid"""
    if angle_matrix.shape[0] < 3:
        raise ValueError("Not enough frames to choose key indices")
    left_elbow = angle_matrix[:, 0]
    idx_top = int(np.argmax(left_elbow))
    idx_impact = int(np.argmin(left_elbow))
    idx_mid = int(len(left_elbow) // 2)
    return [idx_top, idx_impact, idx_mid]

def smooth_matrix(mat, window=5):
    if window <= 1:
        return mat
    sm = np.copy(mat)
    for c in range(mat.shape[1]):
        sm[:, c] = np.convolve(mat[:, c], np.ones(window)/window, mode='same')
    return sm

def compute_similarity_score(mean_abs_diff, max_diff=MAX_DIFF_FOR_SCORE):
    score = max(0.0, 100.0 * (1.0 - (mean_abs_diff / max_diff)))
    return float(score)

# ---------- LOAD IDEAL ----------
def load_ideal(ideal_path):
    arr = np.load(ideal_path, allow_pickle=True)
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        raise ValueError("Ideal file must be a 2D numpy array (3x7 or Tx7).")
    if arr.shape[1] != len(ANGLE_NAMES):
        raise ValueError(f"Ideal file has {arr.shape[1]} columns but {len(ANGLE_NAMES)} angles expected.")
    if arr.shape[0] == 3:
        return arr.astype(float)
    # else pick keys from per-frame series
    idxs = pick_key_indices_from_series(arr)
    return arr[idxs, :].astype(float)

# ---------- EXTRACT USER ANGLES ----------
def extract_user_angle_sequence_from_video(video_path_or_index, use_webcam=False, max_frames=None):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0) if use_webcam else cv2.VideoCapture(video_path_or_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open capture: " + str(video_path_or_index))
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    angles_list = []
    frames = []
    frame_idx = 0
    print("Capturing frames and extracting angles (press 'q' to stop when using webcam)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if use_webcam:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_world_landmarks:
            lm = results.pose_world_landmarks.landmark
            angles = compute_angles_from_world_landmarks(lm)
            angles_list.append(angles)
            frames.append(frame.copy())
        if use_webcam:
            display = frame.copy()
            cv2.putText(display, f"Frames captured: {len(angles_list)}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.imshow("Position yourself (q to finish)", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_idx += 1
        if max_frames and frame_idx >= max_frames:
            break
        if use_webcam:
            display = frame.copy()
            if results.pose_landmarks:
                # Draw landmarks and connections
                mp.solutions.drawing_utils.draw_landmarks(
                    display,
                    results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                )
            # Show frame count
            cv2.putText(display, f"Frames captured: {len(angles_list)}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.imshow("Position yourself (q to finish)", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    pose.close()
    cap.release()
    if use_webcam:
        cv2.destroyAllWindows()
    if len(angles_list) == 0:
        raise RuntimeError("No valid pose frames found.")
    angle_matrix = np.array(angles_list, dtype=float)
    return angle_matrix, frames

# ---------- COMPARE ----------
def compare_keyframes(ideal_key_angles, user_key_angles):
    results = {"phases": []}
    per_phase_scores = []
    for i, phase in enumerate(PHASE_NAMES):
        ideal_vec = ideal_key_angles[i]
        user_vec = user_key_angles[i]
        diffs = np.abs(user_vec - ideal_vec)
        mean_diff = float(np.mean(diffs))
        score = compute_similarity_score(mean_diff)
        per_joint = {}
        for j, name in enumerate(ANGLE_NAMES):
            d = float(diffs[j])
            status = "Good" if d <= TOLERANCE_DEG else "Adjust"
            per_joint[name] = {"diff_deg": d, "status": status}
        results["phases"].append({
            "phase": phase,
            "mean_diff_deg": mean_diff,
            "score_percent": score,
            "per_joint": per_joint
        })
        per_phase_scores.append(score)
    results["overall_score_percent"] = float(np.mean(per_phase_scores))
    return results

def overlay_feedback_on_frames(frames, user_key_idxs, results, out_dir=None):
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    saved = []
    for i, idx in enumerate(user_key_idxs):
        if idx < 0 or idx >= len(frames):
            continue
        img = frames[idx].copy()
        phase = PHASE_NAMES[i]
        info = results["phases"][i]
        header = f"{phase} | Score: {int(info['score_percent'])}%"
        cv2.putText(img, header, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        y = 90
        for name in ANGLE_NAMES:
            pj = info["per_joint"][name]
            color = (0,255,0) if pj["status"] == "Good" else (0,0,255)
            cv2.putText(img, f"{name}: {int(pj['diff_deg'])}deg {pj['status']}", (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            y += 30
        win = f"{phase} comparison"
        cv2.imshow(win, img)
        cv2.waitKey(500)
        if out_dir:
            fname = os.path.join(out_dir, f"{phase}_comparison.png")
            cv2.imwrite(fname, img)
            saved.append(fname)
    cv2.destroyAllWindows()
    return saved

# ---------- MAIN ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ideal", required=True, help="Path to ideal .npy (3x7 or Tx7)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--webcam", action="store_true")
    g.add_argument("--video", help="user video path")
    p.add_argument("--outdir", default="compare_out", help="output folder")
    p.add_argument("--max_frames", type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print("Loading ideal from:", args.ideal)
    ideal_key_angles = load_ideal(args.ideal)
    print("Ideal key angles (3 x N):\n", ideal_key_angles)

    if args.webcam:
        user_matrix, frames = extract_user_angle_sequence_from_video(None, use_webcam=True, max_frames=args.max_frames)
    else:
        user_matrix, frames = extract_user_angle_sequence_from_video(args.video, use_webcam=False, max_frames=args.max_frames)

    user_smoothed = smooth_matrix(user_matrix, window=SMOOTH_WINDOW)
    user_key_idxs = pick_key_indices_from_series(user_smoothed)
    print("User key indices:", user_key_idxs)

    user_key_angles = np.array([user_smoothed[idx] for idx in user_key_idxs], dtype=float)
    compare_results = compare_keyframes(ideal_key_angles, user_key_angles)
    print("Comparison:\n", json.dumps(compare_results, indent=2))

    saved_images = overlay_feedback_on_frames(frames, user_key_idxs, compare_results, out_dir=args.outdir)
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ideal_path": os.path.abspath(args.ideal),
        "user_source": "webcam" if args.webcam else os.path.abspath(args.video),
        "user_key_indices_valid_frames": [int(i) for i in user_key_idxs],
        "compare_results": compare_results,
        "saved_images": saved_images
    }
    summary_path = os.path.join(args.outdir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved summary to:", summary_path)
    print("Overall score:", compare_results["overall_score_percent"])

if __name__ == "__main__":
    main()
