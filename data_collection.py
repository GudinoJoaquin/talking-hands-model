import cv2
import mediapipe as mp
import numpy as np
import os
import time

# =======================
# CONFIGURACIÓN PRINCIPAL
# =======================
DATA_PATH = "data_holistic"
LABELS = ["A", "B", "C", "CH", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
SEQUENCE_LENGTH = 50
SAMPLES_PER_LABEL = 100
DELAY_BETWEEN_SAMPLES = 0.05
CAMERA_SOURCE = 0

# =======================
# CONFIGURACIÓN MEDIAPIPE
# =======================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 

# 🔸 Usaremos una lista reducida de índices del rostro
# (algunos puntos clave: ojos, cejas, nariz, boca, mentón)
FACE_LANDMARKS_REDUCED = [
    1, 33, 61, 199, 263, 291,   # contorno general
    5, 45, 65, 295, 324, 355,   # mejillas
    0, 4, 9, 94, 164, 168,      # nariz y frente
    11, 13, 14, 17, 37, 39, 82, 87, 178, 400, 402, 435  # ojos y boca
]

REDUCED_FACE_COUNT = len(FACE_LANDMARKS_REDUCED)
NUM_LANDMARKS = (REDUCED_FACE_COUNT + 33 + 21 + 21) * 3  # cara reducida + cuerpo + manos


def extract_holistic_landmarks(results):
    """Extrae cara reducida, cuerpo y manos."""
    data = []

    # Cara reducida
    if results.face_landmarks:
        face = results.face_landmarks.landmark
        for idx in FACE_LANDMARKS_REDUCED:
            if idx < len(face):
                lm = face[idx]
                data.extend([lm.x, lm.y, lm.z])
            else:
                data.extend([0.0, 0.0, 0.0])
    else:
        data.extend([0.0] * (REDUCED_FACE_COUNT * 3))

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


def ensure_dirs():
    os.makedirs(DATA_PATH, exist_ok=True)
    for label in LABELS:
        os.makedirs(os.path.join(DATA_PATH, label), exist_ok=True)


def collect():
    ensure_dirs()
    cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("⚠️ No se pudo abrir la cámara.")
        return

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        refine_face_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as holistic:

        for label in LABELS:
            print(f"\n🖐 Grabando muestras para '{label}' automáticamente...")
            saved = 0
            recording = False
            seq = []

            while saved < SAMPLES_PER_LABEL:
                ret, frame = cap.read()
                if not ret:
                    continue
                img = cv2.flip(frame, 1)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = holistic.process(img_rgb)

                # Dibujar landmarks (solo para visualización)
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # Dibuja la cara reducida (solo algunos puntos)
                if results.face_landmarks:
                    for idx in FACE_LANDMARKS_REDUCED:
                        if idx < len(results.face_landmarks.landmark):
                            lm = results.face_landmarks.landmark[idx]
                            h, w, _ = img.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.circle(img, (cx, cy), 1, (255, 200, 0), -1)

                # Detectar si hay manos
                hands_visible = (results.left_hand_landmarks or results.right_hand_landmarks)

                if hands_visible:
                    if not recording:
                        print(f"🎬 Grabando muestra {saved+1}/{SAMPLES_PER_LABEL} para '{label}'...")
                        recording = True
                        seq = []

                    lm = extract_holistic_landmarks(results)
                    seq.append(lm)

                    # Barra de progreso
                    progress = int((len(seq) / SEQUENCE_LENGTH) * 200)
                    cv2.rectangle(img, (10, 10), (210, 30), (50, 50, 50), -1)
                    cv2.rectangle(img, (10, 10), (10 + progress, 30), (0, 255, 0), -1)
                    cv2.putText(img, f"{label} [{len(seq)}/{SEQUENCE_LENGTH}]",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if len(seq) == SEQUENCE_LENGTH:
                        arr = np.array(seq)
                        filename = os.path.join(DATA_PATH, label, f"{label}_{saved:03d}.npy")
                        np.save(filename, arr)
                        saved += 1
                        print(f"💾 Guardado: {filename} (shape {arr.shape})")
                        recording = False
                        seq = []
                        time.sleep(DELAY_BETWEEN_SAMPLES)
                else:
                    if recording and len(seq) < SEQUENCE_LENGTH:
                        print("⏸ Manos perdidas, cancelando grabación actual.")
                        recording = False
                        seq = []

                    cv2.putText(img, f"Esperando manos... {label} ({saved+1}/{SAMPLES_PER_LABEL})",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Data Collection (Face-Lite)", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            print(f"✅ Completadas todas las muestras para '{label}'.")

    cap.release()
    cv2.destroyAllWindows()
    print("🎉 Recolección finalizada correctamente.")


if __name__ == "__main__":
    collect()
