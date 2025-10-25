import cv2
import mediapipe as mp
import numpy as np

# set up things
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
VIDEO_PATH = "test_video.mp4"
OUTPUT_FILE = "ideal_angles.npy"

# computes angle between 3 points
def angle_3d(a, b, c):
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))


# extract angles
cap = cv2.VideoCapture(VIDEO_PATH)
all_angles = []
arm_angle_series = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        continue
    lm = results.pose_landmarks.landmark
    def pt(i): return [lm[i].x, lm[i].y]

    # Compute representative angles
    lm = results.pose_world_landmarks.landmark
    L_SH, L_EL, L_WR = lm[11], lm[13], lm[15]
    R_SH, R_EL, R_WR = lm[12], lm[14], lm[16]
    L_HIP, L_KNEE, L_ANK = lm[23], lm[25], lm[27]

    left_arm = angle_3d(L_SH, L_EL, L_WR)
    right_arm = angle_3d(R_SH, R_EL, R_WR)
    back = angle_3d(L_SH, L_HIP, L_KNEE)
    
    arm_angle_series.append(left_arm)

    all_angles.append([left_arm, right_arm, back])

cap.release()
all_angles = np.array(all_angles)

# get key frames
arm_angle_series = np.array(arm_angle_series)
key_idxs = [
    np.argmax(arm_angle_series),  # top of backswing
    np.argmin(arm_angle_series),  # impact
    len(arm_angle_series)//2,     # mid-swing approx
]
key_angles = all_angles[key_idxs]

# save values to output
np.save(OUTPUT_FILE, key_angles)
print(f"Ideal swing angles saved to {OUTPUT_FILE}")
