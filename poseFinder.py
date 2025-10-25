import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# ---------------------------
# Setup MediaPipe
# ---------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6)

# ---------------------------
# 3D Angle function
# ---------------------------
def angle_3d(a, b, c):
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

# ---------------------------
# Ideal swing angles (example 3 key frames)
# ---------------------------
# For simplicity, we use one "frame" here, you can expand to multiple frames
ideal_angles = {
    "L_Elbow": 160,
    "R_Elbow": 160,
    "L_Shoulder": 45,
    "R_Shoulder": 45,
    "L_Knee": 175,
    "R_Knee": 175,
    "Torso": 40,
}

tolerance = 10  # degrees

# ---------------------------
# Smoothing buffers
# ---------------------------
angle_history = {joint: deque(maxlen=5) for joint in ideal_angles.keys()}

# ---------------------------
# Capture
# ---------------------------
cap = cv2.VideoCapture(0)  #cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks and results.pose_world_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = results.pose_world_landmarks.landmark

        # ---------------------------
        # Compute angles
        # ---------------------------
        angles = {}
        angles["L_Elbow"] = angle_3d(lm[11], lm[13], lm[15])
        angles["R_Elbow"] = angle_3d(lm[12], lm[14], lm[16])
        angles["L_Shoulder"] = angle_3d(lm[23], lm[11], lm[13])  # hip-shoulder-elbow
        angles["R_Shoulder"] = angle_3d(lm[24], lm[12], lm[14])
        angles["L_Knee"] = angle_3d(lm[23], lm[25], lm[27])
        angles["R_Knee"] = angle_3d(lm[24], lm[26], lm[28])
        angles["Torso"] = angle_3d(lm[11], lm[23], lm[25])  # shoulder-hip-knee

        # ---------------------------
        # Smooth angles
        # ---------------------------
        for joint in angles:
            angle_history[joint].append(angles[joint])
            angles[joint] = np.mean(angle_history[joint])

        # ---------------------------
        # Provide feedback
        # ---------------------------
        y_offset = 30
        for joint, val in angles.items():
            diff = abs(val - ideal_angles[joint])
            color = (0, 255, 0) if diff <= tolerance else (0, 0, 255)
            text = f"{joint}: {int(val)}°"
            if diff > tolerance:
                text += " <- Adjust"
            cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30

    else:
        cv2.putText(image, "No pose detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Golf Swing Assistant MVP", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
