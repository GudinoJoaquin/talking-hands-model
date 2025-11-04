# recognize.py
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import time

MODEL_PATH = "model.h5"
LABEL_MAP_PATH = "label_map.npy"
SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 63

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_hand_landmarks(results):
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0]
    coords = []
    for p in lm.landmark:
        coords.extend([p.x, p.y, p.z])
    return np.array(coords, dtype=np.float32)

def load_label_map(path):
    d = np.load(path, allow_pickle=True).item()
    # invertir diccionario
    inv = {v:k for k,v in d.items()}
    return inv

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    label_map = load_label_map(LABEL_MAP_PATH)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    seq_buffer = []
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
        last_prediction = None
        last_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            img = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            lm = extract_hand_landmarks(results)
            if lm is None:
                seq_buffer.append(np.zeros(NUM_LANDMARKS, dtype=np.float32))
            else:
                seq_buffer.append(lm)
            if len(seq_buffer) > SEQUENCE_LENGTH:
                seq_buffer.pop(0)
            # predicción solo si buffer completo
            pred_text = ""
            if len(seq_buffer) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(np.array(seq_buffer), axis=0)  # (1, seq_len, feats)
                probs = model.predict(input_data, verbose=0)[0]
                idx = np.argmax(probs)
                score = probs[idx]
                pred_text = f"{label_map[idx]} ({score:.2f})"
                # filtro simple: mostrar pred solo si confianza suficiente y con cooldown
                if score > 0.7 and (time.time() - last_time) > 0.6:
                    last_prediction = label_map[idx]
                    last_time = time.time()
            # dibujar y mostrar
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            display_text = last_prediction if last_prediction is not None else pred_text
            cv2.putText(img, f"Pred: {display_text}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.imshow("Sign Recognition", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
