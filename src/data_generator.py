import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence

class VideoSequenceGenerator(Sequence):
    def __init__(self, base_path, classes, batch_size=4, seq_len=16, img_size=(128,128)):
        self.base_path = base_path
        self.classes = classes
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.img_size = img_size

        self.samples = []
        for label, cls in enumerate(classes):
            cls_path = os.path.join(base_path, cls)
            for seq in os.listdir(cls_path):
                self.samples.append((os.path.join(cls_path, seq), label))

    def __len__(self):
        return len(self.samples) // self.batch_size

    def __getitem__(self, index):
        batch_samples = self.samples[index * self.batch_size : (index+1) * self.batch_size]

        X = np.zeros((self.batch_size, self.seq_len, self.img_size[0], self.img_size[1], 3), dtype=np.float32)
        y = np.zeros((self.batch_size,), dtype=np.int32)

        for i, (seq_folder, label) in enumerate(batch_samples):
            frames = sorted(os.listdir(seq_folder))
            for t in range(self.seq_len):
                img_path = os.path.join(seq_folder, frames[t])
                img = cv2.imread(img_path)
                img = cv2.resize(img, self.img_size)
                img = img.astype(np.float32) / 255.0
                X[i, t] = img
            y[i] = label

        return X, y
