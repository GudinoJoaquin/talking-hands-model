# train_model.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

DATA_PATH = "data"
SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 63  # 21 * (x,y,z)

def load_data():
    sequences = []
    labels = []
    label_names = sorted(os.listdir(DATA_PATH))
    label_map = {name:i for i,name in enumerate(label_names)}
    print("Etiquetas encontradas:", label_map)
    for label in label_names:
        folder = os.path.join(DATA_PATH, label)
        for f in os.listdir(folder):
            if f.endswith(".npy"):
                arr = np.load(os.path.join(folder, f))
                # validar forma
                if arr.shape != (SEQUENCE_LENGTH, NUM_LANDMARKS):
                    print("Advertencia: archivo con forma inesperada", f, arr.shape)
                    continue
                sequences.append(arr)
                labels.append(label_map[label])
    X = np.array(sequences)  # (n_samples, seq_len, features)
    y = to_categorical(labels)
    return X, y, label_map

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.4))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    X, y, label_map = load_data()
    print("Datos cargados:", X.shape, y.shape)
    if X.shape[0] == 0:
        raise SystemExit("No hay muestras. Ejecuta data_collection.py primero.")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1))
    model = build_model((SEQUENCE_LENGTH, NUM_LANDMARKS), y.shape[1])
    model.summary()
    callbacks = [
        ModelCheckpoint("model.h5", monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    ]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=16, callbacks=callbacks)
    # guardar mapa de etiquetas
    np.save("label_map.npy", label_map)
    print("Entrenamiento completado. Modelo guardado como model.h5 y label_map.npy")
