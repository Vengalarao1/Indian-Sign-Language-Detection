import cv2
import mediapipe as mp
import copy
import itertools
from tensorflow import keras
import numpy as np
import pandas as pd
import string
import time
from collections import deque

# Load model
model = keras.models.load_model("model.h5")

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Alphabet classes
alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

# Smooth prediction buffer
prediction_buffer = deque(maxlen=5)

# Functions
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    return [[min(int(lm.x * image_width), image_width - 1),
             min(int(lm.y * image_height), image_height - 1)]
            for lm in landmarks.landmark]

def pre_process_landmark(landmark_list):
    temp_landmarks = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmarks[0]
    
    for point in temp_landmarks:
        point[0] -= base_x
        point[1] -= base_y

    flat_landmarks = list(itertools.chain.from_iterable(temp_landmarks))
    max_val = max(map(abs, flat_landmarks)) or 1  # avoid div by zero
    normalized = [x / max_val for x in flat_landmarks]
    
    return normalized

def get_smoothed_prediction(buffer):
    if not buffer:
        return None
    counts = np.bincount(buffer)
    return np.argmax(counts)

# Webcam input
cap = cv2.VideoCapture(0)
prev_time = 0

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        image = cv2.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        label = ""
        confidence = 0

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(image, hand_landmarks)
                if landmark_list:
                    processed = pre_process_landmark(landmark_list)
                    df = pd.DataFrame([processed])  # shape: (1, 42)
                    
                    predictions = model.predict(df, verbose=0)
                    predicted_class = np.argmax(predictions)
                    confidence = predictions[0][predicted_class]

                    # Save prediction to buffer
                    prediction_buffer.append(predicted_class)

                    smoothed_class = get_smoothed_prediction(prediction_buffer)
                    label = alphabet[smoothed_class]

                    # Draw annotations
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    # Show handedness
                    handed_label = handedness.classification[0].label
                    cv2.putText(image, f"{handed_label}", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Display result
        if label:
            cv2.putText(image, f"{label} ({confidence*100:.1f}%)", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time
        cv2.putText(image, f"FPS: {int(fps)}", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow('Indian Sign Language Detector', image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
