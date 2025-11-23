import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from data_generator import VideoSequenceGenerator

# -----------------------------
# GPU SETUP
# -----------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

# -----------------------------
# SETTINGS
# -----------------------------
SEQ_LEN = 16
IMG_SIZE = (128, 128)  # You can reduce to 96x96 or 64x64 if needed
BATCH_SIZE = 3   # Adjust to 2 if OOM
EPOCHS = 4         # Start small for testing
CLASSES = ["fire", "smoke", "no_fire"]

# -----------------------------
# DATASET LOADERS
# -----------------------------
train_path = "../data/sequences"

train_gen = VideoSequenceGenerator(
    base_path=train_path,
    classes=CLASSES,
    batch_size=BATCH_SIZE,
    seq_len=SEQ_LEN,
    img_size=IMG_SIZE
)

print("Total batches per epoch:", len(train_gen))

# -----------------------------
# MODEL ARCHITECTURE (ConvLSTM)
# -----------------------------
model = Sequential()

# First ConvLSTM
model.add(ConvLSTM2D(
    filters=64,
    kernel_size=(3, 3),
    padding="same",
    input_shape=(SEQ_LEN, IMG_SIZE[0], IMG_SIZE[1], 3),
    activation="tanh",
    return_sequences=True
))
model.add(BatchNormalization())

# Second ConvLSTM
model.add(ConvLSTM2D(
    filters=32,
    kernel_size=(3, 3),
    padding="same",
    activation="tanh",
    return_sequences=False
))
model.add(BatchNormalization())

# Optional 3D Conv layer for spatio-temporal features
model.add(tf.keras.layers.Reshape((1, IMG_SIZE[0], IMG_SIZE[1], 32)))
model.add(Conv3D(filters=32, kernel_size=(1,3,3), activation='relu'))

# Dense layers for classification
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(len(CLASSES), activation="softmax"))

# -----------------------------
# COMPILE MODEL
# -----------------------------
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(learning_rate=0.0001),
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# TRAIN
# -----------------------------
history = model.fit(
    train_gen,
    epochs=EPOCHS
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("best.pt")
print("\nModel saved as fire_smoke_hazard_convlstm_gpu.h5")
