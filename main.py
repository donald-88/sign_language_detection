import cv2
import mediapipe as mp
from matplotlib import pyplot as plt
import numpy as np

# init
mp_holistic = mp.solutions.holistic  # holistic model
mp_drawing = mp.solutions.drawing_utils  # drawing utilities
mp_drawing_styles = mp.solutions.drawing_styles  # drawing styles

# detection function


def detection(image, model):
    # converting color from bgr to rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)  # make detection with mediapipe
    image.flags.writeable = True
    # converting color from rgb to bgr
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# draw landmarks function


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, mp_drawing.DrawingSpec(color=(
        149, 183, 58), thickness=2, circle_radius=1), mp_drawing.DrawingSpec(color=(86, 207, 255), thickness=2, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())


def extract_keypoints(results):
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
    ) if results.face_landmarks else np.zeros(468*3)
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten(
    ) if results.pose_landmarks else np.zeros(33*3)
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([face, pose, left_hand, right_hand])


cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        plt.imshow(frame)

        # detect feed
        image, results = detection(frame, holistic)
        draw_landmarks(frame, results)
        extract_keypoints(results)

        cv2.imshow('Video Feed', frame)
        if cv2.waitKey(5) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
