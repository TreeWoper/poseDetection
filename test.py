import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    if angle > 180:
        angle = 360 - angle
    return angle

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark
            h, w, _ = image.shape

            # Get 2D coordinates for each relevant joint
            def coord(point): return [lm[point].x * w, lm[point].y * h]

            joints = {
                "L_shoulder": coord(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
                "L_elbow": coord(mp_pose.PoseLandmark.LEFT_ELBOW.value),
                "L_wrist": coord(mp_pose.PoseLandmark.LEFT_WRIST.value),
                "R_shoulder": coord(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
                "R_elbow": coord(mp_pose.PoseLandmark.RIGHT_ELBOW.value),
                "R_wrist": coord(mp_pose.PoseLandmark.RIGHT_WRIST.value),
                "L_hip": coord(mp_pose.PoseLandmark.LEFT_HIP.value),
                "R_hip": coord(mp_pose.PoseLandmark.RIGHT_HIP.value),
                "L_knee": coord(mp_pose.PoseLandmark.LEFT_KNEE.value),
                "R_knee": coord(mp_pose.PoseLandmark.RIGHT_KNEE.value),
                "L_ankle": coord(mp_pose.PoseLandmark.LEFT_ANKLE.value),
                "R_ankle": coord(mp_pose.PoseLandmark.RIGHT_ANKLE.value),
            }

            # Compute multiple angles
            angles = {
                "L_Elbow": calculate_angle(joints["L_shoulder"], joints["L_elbow"], joints["L_wrist"]),
                "R_Elbow": calculate_angle(joints["R_shoulder"], joints["R_elbow"], joints["R_wrist"]),
                "L_Knee": calculate_angle(joints["L_hip"], joints["L_knee"], joints["L_ankle"]),
                "R_Knee": calculate_angle(joints["R_hip"], joints["R_knee"], joints["R_ankle"]),
                "Torso": calculate_angle(joints["L_shoulder"], joints["L_hip"], joints["L_ankle"]),
            }

            # Display all angles
            y_offset = 30
            for name, ang in angles.items():
                cv2.putText(image, f"{name}: {int(ang)}°", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30

        else:
            cv2.putText(image, "No pose detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Pose Angles", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
