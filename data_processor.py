import os
import numpy as np
import librosa
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, Optional, List, Union, Any
from numpy.typing import NDArray


class AcousticDataProcessor:
    input_csv: str
    output_dir: str
    n_mels: int
    img_size: Tuple[int, int]

    def __init__(self, input_csv: str = "metadata_pro.csv") -> None:
        self.input_csv = input_csv
        self.output_dir = "spectrograms_pro"
        os.makedirs(self.output_dir, exist_ok=True)

        self.n_mels = 128
        self.img_size = (128, 64)

    def analyze_and_convert(self, row: pd.Series) -> Tuple[Optional[str], bool]:
        wav_path: str = row['wav_path']
        if not os.path.exists(wav_path):
            return None, False

        try:
            y: NDArray[np.float32]
            sr: int
            y, sr = librosa.load(wav_path, sr=None)

            if len(y) == 0:
                return None, False

            rms: NDArray[np.float32] = librosa.feature.rms(y=y)[0]

            impulsiveness: float = float(np.max(rms) / (np.mean(rms) + 1e-9))
            flatness: float = float(np.mean(librosa.feature.spectral_flatness(y=y)))
            signal_power: float = float(np.max(rms) ** 2)
            noise_power: float = float(np.mean(rms) ** 2)
            snr_proxy: float = 10 * np.log10(signal_power / (noise_power + 1e-9))

            is_clean: bool = (
                    impulsiveness > 1.2 and
                    flatness < 0.30 and
                    snr_proxy > 3.0
            )

            y = librosa.effects.preemphasis(y)
            peak_idx: int = int(np.argmax(np.abs(y)))

            start: int = max(0, peak_idx - int(sr * 0.05))
            end: int = min(len(y), peak_idx + int(sr * 0.15))
            y_focus: NDArray[np.float32] = y[start:end]

            target_len: int = int(sr * 0.20)
            if len(y_focus) < target_len:
                y_focus = np.pad(y_focus, (0, target_len - len(y_focus)), mode='constant')

            S: NDArray[np.float32] = librosa.feature.melspectrogram(
                y=y_focus,
                sr=sr,
                n_mels=self.n_mels,
                fmax=sr // 2,
                n_fft=2048,
                hop_length=512
            )
            S_db: NDArray[np.float32] = librosa.power_to_db(S, ref=1.0)

            img_name: str = os.path.basename(wav_path).replace(".wav", ".png")
            img_path: str = os.path.join(self.output_dir, img_name)

            plt.imsave(img_path, S_db, cmap='magma', format='png')

            return img_path, is_clean

        except Exception as e:
            print(f"Błąd przetwarzania {wav_path}: {e}")
            return None, False

    def run(self) -> None:
        if not os.path.exists(self.input_csv):
            print(f"BŁĄD: Nie znaleziono pliku {self.input_csv}")
            return

        print("=" * 60)
        print("PROCESOR DANYCH AKUSTYCZNYCH")
        print("=" * 60)

        df: pd.DataFrame = pd.read_csv(self.input_csv)

        if 'rms_val' not in df.columns:
            df['rms_val'] = 0.0

        total: int = len(df)
        print(f"Znaleziono: {total} próbek")

        results: List[Tuple[Optional[str], bool]] = []
        for _, row in tqdm(df.iterrows(), total=total, desc="Przetwarzanie"):
            results.append(self.analyze_and_convert(row))

        res_zip: Tuple[Any, ...] = zip(*results)
        df['spectrogram_path'], df['is_clean'] = res_zip

        df_valid: pd.DataFrame = df[df['spectrogram_path'].notna()]
        df_clean: pd.DataFrame = df_valid[df_valid['is_clean'] == True]
        df_noise: pd.DataFrame = df_valid[df_valid['is_clean'] == False]

        clean_csv: str = "metadata_pro_train_clean.csv"
        noise_csv: str = "metadata_pro_test_noise.csv"

        df_clean.to_csv(clean_csv, index=False)
        df_noise.to_csv(noise_csv, index=False)

        print(f"Zapisano: {clean_csv} oraz {noise_csv}")


if __name__ == "__main__":
    processor: AcousticDataProcessor = AcousticDataProcessor()
    processor.run()