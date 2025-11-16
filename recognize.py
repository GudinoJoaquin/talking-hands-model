# recognize.py
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import time

MODEL_PATH = "model.h5"
LABEL_MAP_PATH = "label_map.npy"
NUM_LANDMARKS = 1629


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Ajusta según tus datos (63 para manos, 132 para cara, 33*3=99 para pose)
# Si tu modelo original solo usa manos, dejá NUM_LANDMARKS = 63.
# Si vas a reentrenar con todo el cuerpo, actualizalo al total de landmarks * 3.


def extract_holistic_landmarks(results):
    """Extrae landmarks de pose, rostro y manos."""
    data = []

    # Cara
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0.0] * (132 * 3))

    # Pose
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0.0] * (33 * 3))

    # Mano izquierda
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0.0] * (21 * 3))

    # Mano derecha
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0.0] * (21 * 3))

    return np.array(data, dtype=np.float32)


def load_label_map(path):
    d = np.load(path, allow_pickle=True).item()
    return {v: k for k, v in d.items()}


def draw_confidence_window(probs, label_map, top_k=5):
    bar_width = 400
    bar_height = 30
    img_height = 200 + (top_k * 40)
    img = np.zeros((img_height, 500, 3), dtype=np.uint8)

    sorted_indices = np.argsort(probs)[::-1]
    top_indices = sorted_indices[:top_k]

    top_label = label_map[top_indices[0]]
    top_score = probs[top_indices[0]]
    cv2.putText(img, f"Prediccion principal: {top_label} ({top_score*100:.1f}%)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.rectangle(img, (20, 60), (20 + bar_width, 60 + bar_height), (50,50,50), -1)
    filled = int(bar_width * top_score)
    cv2.rectangle(img, (20, 60), (20 + filled, 60 + bar_height), (0,255,0), -1)
    cv2.putText(img, f"{top_score*100:.1f}%", (bar_width + 40, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    start_y = 120
    cv2.putText(img, "Otras posibles predicciones:", (20, start_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
    y = start_y + 30
    for i in range(1, len(top_indices)):
        lbl = label_map[top_indices[i]]
        sc = probs[top_indices[i]]
        color = (100, 255 - int(150*i/top_k), 255 - int(255*i/top_k))
        cv2.putText(img, f"{lbl}: {sc*100:.1f}%", (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 35

    return img


if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    label_map = load_label_map(LABEL_MAP_PATH)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    seq_buffer = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as holistic:
        last_prediction = None
        last_score = 0.0
        last_probs = None
        last_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            img = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(img_rgb)

            lm = extract_holistic_landmarks(results)
            seq_buffer.append(lm)
            if len(seq_buffer) > SEQUENCE_LENGTH:
                seq_buffer.pop(0)

            if len(seq_buffer) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(np.array(seq_buffer), axis=0)
                probs = model.predict(input_data, verbose=0)[0]
                idx = np.argmax(probs)
                score = probs[idx]
                pred_label = label_map[idx]
                last_probs = probs

                if score > 0.7 and (time.time() - last_time) > 0.6:
                    last_prediction = pred_label
                    last_score = score
                    last_time = time.time()
                else:
                    last_score = score

            # Dibujar landmarks de cuerpo, cara y manos
            mp_drawing.draw_landmarks(
                img, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
            )
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
            )
            mp_drawing.draw_landmarks(
                img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            )
            mp_drawing.draw_landmarks(
                img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

            display_text = last_prediction if last_prediction is not None else "Esperando"
            cv2.putText(img, f"Pred: {display_text}", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            cv2.imshow("Sign Recognition", img)

            if last_probs is not None:
                conf_window = draw_confidence_window(last_probs, label_map, top_k=5)
            else:
                conf_window = np.zeros((200, 500, 3), dtype=np.uint8)
                cv2.putText(conf_window, "Esperando predicciones...", (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            cv2.imshow("Confidence Monitor", conf_window)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
