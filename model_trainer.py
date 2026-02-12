import pandas as pd
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from typing import Tuple, Optional, Any, List, Union
from numpy.typing import NDArray


class AcousticModelTrainer:
    clean_csv: str
    noise_csv: str
    model_dir: str
    img_size: Tuple[int, int]

    def __init__(self, clean_csv: str = "metadata_pro_train_clean.csv",
                 noise_csv: str = "metadata_pro_test_noise.csv",
                 model_dir: str = "model_output",
                 img_size: Tuple[int, int] = (128, 64)) -> None:
        self.clean_csv = clean_csv
        self.noise_csv = noise_csv
        self.model_dir = model_dir
        self.img_size = img_size
        os.makedirs(self.model_dir, exist_ok=True)

    def _load_img_func(self, path: str) -> NDArray[np.float32]:
        img_name: str = os.path.basename(path).replace(".wav", ".png")
        img_path: str = os.path.join("spectrograms_pro", img_name)

        try:
            if not os.path.exists(img_path):
                return np.zeros((self.img_size[1], self.img_size[0], 1), dtype=np.float32)

            img: Any = load_img(img_path, target_size=(self.img_size[1], self.img_size[0]), color_mode="grayscale")
            return img_to_array(img) / 255.0
        except Exception:
            return np.zeros((self.img_size[1], self.img_size[0], 1), dtype=np.float32)

    def prepare_data(self) -> Tuple[
        Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64], NDArray[np.int64]], int]:
        if not os.path.exists(self.clean_csv):
            os._exit(1)

        df_raw: pd.DataFrame = pd.read_csv(self.clean_csv)
        min_samples_threshold: int = 20
        counts: pd.Series = df_raw['key_id'].value_counts()
        valid_keys: pd.Index = counts[counts >= min_samples_threshold].index

        df_clean: pd.DataFrame = df_raw[df_raw['key_id'].isin(valid_keys)]
        #df_noise: pd.DataFrame = pd.read_csv(self.noise_csv) if os.path.exists(self.noise_csv) else pd.DataFrame()

        key_samples: pd.DataFrame = df_clean[df_clean['key_id'] != 'unknown']
        df_unkn: pd.DataFrame = df_clean[df_clean['key_id'] == 'unknown']

        if not key_samples.empty:
            limit_unkn: int = int(key_samples['key_id'].value_counts().max() * 2)
            df_unkn = df_unkn.sample(n=min(len(df_unkn), limit_unkn), random_state=42)

        df_train_base: pd.DataFrame = pd.concat([key_samples, df_unkn], ignore_index=True)

        le: LabelEncoder = LabelEncoder()
        le.fit(df_train_base['key_id'])

        MAX_SAMPLES: int = 100
        df_balanced: pd.DataFrame = df_train_base.groupby('key_id', group_keys=False).sample(
            n=MAX_SAMPLES,
            replace=True,
            random_state=42
        ).reset_index(drop=True)

        x_train: NDArray[np.float32] = np.stack(df_balanced['wav_path'].map(self._load_img_func).values)
        y_train: NDArray[np.int64] = le.transform(df_balanced['key_id'].values)

        df_val: pd.DataFrame = df_train_base.sample(frac=0.20, random_state=42)
        x_val: NDArray[np.float32] = np.stack(df_val['wav_path'].map(self._load_img_func).values)
        y_val: NDArray[np.int64] = le.transform(df_val['key_id'].values)

        with open(os.path.join(self.model_dir, "label_encoder.pickle"), "wb") as f:
            pickle.dump(le, f)

        return (x_train, x_val, y_train, y_val), len(le.classes_)

    def build_model(self, num_classes: int) -> models.Sequential:
        data_aug: models.Sequential = models.Sequential([
            layers.RandomTranslation(height_factor=0, width_factor=0.05),
            layers.RandomContrast(0.15),
        ])

        model: models.Sequential = models.Sequential([
            layers.Input(shape=(self.img_size[1], self.img_size[0], 1)),
            data_aug,
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, epochs: int = 20, batch_size: int = 32) -> None:
        data_pack: Tuple[Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64], NDArray[np.int64]], int]
        data_pack, num_classes = self.prepare_data()
        (x_train, x_val, y_train, y_val) = data_pack

        model: models.Sequential = self.build_model(num_classes)

        cbs: List[callbacks.Callback] = [
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
            callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            callbacks.ModelCheckpoint(os.path.join(self.model_dir, "best_model.keras"), save_best_only=True)
        ]

        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cbs,
            verbose=1
        )

        model.save(os.path.join(self.model_dir, "acoustic_key_model.keras"))


if __name__ == "__main__":
    trainer: AcousticModelTrainer = AcousticModelTrainer()
    trainer.train(epochs=50, batch_size=32)