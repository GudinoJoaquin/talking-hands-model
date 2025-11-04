import cv2
import mediapipe as mp
import numpy as np
import os
import time

# =======================
# CONFIGURACIÓN PRINCIPAL
# =======================
DATA_PATH = "data"
LABELS = ["NN", "Q", "W", "X"]   # cambiá o agregá tus gestos
SEQUENCE_LENGTH = 30        # frames por muestra
SAMPLES_PER_LABEL = 100     # cuántas secuencias grabar por etiqueta
DELAY_BETWEEN_SAMPLES = 0.5 # segundos de pausa entre muestras

# =======================
# CONFIGURACIÓN CÁMARA
# =======================
# OPCIÓN 1: usar la cámara DroidCam como fuente IP (recomendado)
# EJEMPLO: "http://192.168.0.105:4747/video"
# Revisá la IP que te muestra la app DroidCam
CAMERA_SOURCE = 0  # o reemplazá por "http://<tu_ip>:4747/video"

# =======================
# MEDIAPIPE CONFIG
# =======================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_hand_landmarks(results):
    """Extrae los landmarks de la mano en un vector plano."""
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0]
    coords = []
    for p in lm.landmark:
        coords.extend([p.x, p.y, p.z])
    return np.array(coords, dtype=np.float32)


def ensure_dirs():
    """Crea los directorios de datos si no existen."""
    os.makedirs(DATA_PATH, exist_ok=True)
    for label in LABELS:
        os.makedirs(os.path.join(DATA_PATH, label), exist_ok=True)


def fix_green_frame(frame):
    """
    Detecta y corrige frames verdes provenientes de DroidCam.
    Si el frame es YUYV o NV12, los convierte a BGR.
    """
    try:
        # Algunos droidcams envían frames de 2 canales YUYV
        if len(frame.shape) == 2 or frame.shape[2] == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
        # Algunos otros usan NV12 (verde azulado)
        elif frame.shape[2] == 3 and np.mean(frame[..., 1]) > 150 and np.mean(frame[..., 0]) < 80:
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
    except:
        pass
    return frame


def collect():
    ensure_dirs()

    cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("⚠️ No se pudo abrir la cámara. Verifica el índice o la IP de DroidCam.")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as hands:

        for label in LABELS:
            print(f"\n🖐 Preparado para grabar label: '{label}'.")
            print("Presiona 's' para comenzar o 'q' para salir.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = fix_green_frame(frame)
                img = cv2.flip(frame, 1)
                cv2.putText(img, f"Label: {label} - presiona S para grabar",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Data Collection", img)

                k = cv2.waitKey(1) & 0xFF
                if k == ord('s'):
                    break
                elif k == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            saved = 0
            while saved < SAMPLES_PER_LABEL:
                seq = []
                while len(seq) < SEQUENCE_LENGTH:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    frame = fix_green_frame(frame)
                    img = cv2.flip(frame, 1)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = hands.process(img_rgb)

                    lm = extract_hand_landmarks(results)
                    if lm is None:
                        seq.append(np.zeros(63, dtype=np.float32))
                    else:
                        seq.append(lm)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    cv2.putText(
                        img,
                        f"{label} muestra {saved+1}/{SAMPLES_PER_LABEL} frame {len(seq)}/{SEQUENCE_LENGTH}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )
                    cv2.imshow("Data Collection", img)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                arr = np.array(seq)
                filename = os.path.join(DATA_PATH, label, f"{label}_{saved:03d}.npy")
                np.save(filename, arr)
                saved += 1
                print(f"💾 Guardado: {filename} (shape {arr.shape})")
                time.sleep(DELAY_BETWEEN_SAMPLES)

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Recolección finalizada.")


if __name__ == "__main__":
    collect()
