
import cv2
import mediapipe as mp

path = "data/raw/real/my_video.mp4"
cap = cv2.VideoCapture(path)

mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)

total = 0
detected = 0
saved = False

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        total += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mesh.process(rgb)

        if result.multi_face_landmarks:
            detected += 1

            if not saved:
                annotated = frame.copy()
                h, w = frame.shape[:2]

                for landmark in result.multi_face_landmarks[0].landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(
                            annotated, (x, y), 1, (0, 255, 0), -1
                        )

                cv2.imwrite("cache/face_check.jpg", annotated)
                saved = True
finally:
    cap.release()
    mesh.close()

print("Total frames:", total)
print("Face detected:", detected)
print("Detection rate:", round(detected / total * 100, 1) if total else 0, "%")
print("Preview saved to cache/face_check.jpg")
