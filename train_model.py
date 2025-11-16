# train_model_holistic.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

DATA_PATH = "data_holistic"
SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 1629  # cara + pose + manos


def load_data():
    sequences = []
    labels = []
    label_names = sorted(os.listdir(DATA_PATH))
    label_map = {name: i for i, name in enumerate(label_names)}
    print("Etiquetas encontradas:", label_map)
    for label in label_names:
        folder = os.path.join(DATA_PATH, label)
        for f in os.listdir(folder):
            if f.endswith(".npy"):
                arr = np.load(os.path.join(folder, f))
                if arr.shape != (SEQUENCE_LENGTH, NUM_LANDMARKS):
                    print("Advertencia: forma inesperada", f, arr.shape)
                    continue
                sequences.append(arr)
                labels.append(label_map[label])
    X = np.array(sequences)
    y = to_categorical(labels)
    return X, y, label_map


def build_model(input_shape, num_classes):
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=input_shape),
        Dropout(0.4),
        LSTM(128),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    X, y, label_map = load_data()
    print("Datos cargados:", X.shape, y.shape)
    if X.shape[0] == 0:
        raise SystemExit("No hay muestras. Ejecuta data_collection_holistic.py primero.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )

    model = build_model((SEQUENCE_LENGTH, NUM_LANDMARKS), y.shape[1])
    model.summary()

    callbacks = [
        ModelCheckpoint("model_holistic.h5", monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    ]

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, batch_size=8, callbacks=callbacks)

    np.save("label_map_holistic.npy", label_map)
    print("✅ Entrenamiento completado. Modelo guardado como model_holistic.h5 y label_map_holistic.npy")
