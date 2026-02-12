import sounddevice as sd
import numpy as np
import librosa
import os
import pickle
import tensorflow as tf
import threading
import signal
from cv2 import resize, INTER_CUBIC
from scipy.signal import butter, lfilter
from collections import deque
from typing import Optional, List, Any, Tuple, Deque
from numpy.typing import NDArray

class AcousticKeyPredictor:
    fs: int
    device_id: Optional[int]
    threshold: float
    confidence_threshold: float
    is_processing: bool
    stop_event: threading.Event
    audio_buffer: Deque[float]
    img_size: Tuple[int, int]
    b: NDArray[np.float64]
    a: NDArray[np.float64]
    window_size: int
    model: tf.keras.Model
    label_encoder: Any

    def __init__(self, model_dir: str = "model_output", fs: int = 48000, device_id: Optional[int] = None,
                 threshold: float = 0.015, confidence_threshold: float = 0.35) -> None:
        self.fs = fs
        self.device_id = device_id
        self.threshold = threshold
        self.confidence_threshold = confidence_threshold
        self.is_processing = False
        self.stop_event = threading.Event()

        self.audio_buffer = deque(maxlen=int(self.fs * 1.5))
        self.img_size = (128, 64)
        self.b, self.a = butter(4, [200 / (fs / 2), 10000 / (fs / 2)], btype='band')
        self.window_size = int(self.fs * 0.20)

        model_path: str = os.path.join(model_dir, "acoustic_key_model.keras")
        encoder_path: str = os.path.join(model_dir, "label_encoder.pickle")

        self.model = tf.keras.models.load_model(model_path)
        with open(encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

    def _apply_filter(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        return lfilter(self.b, self.a, data)

    def _process_event(self, audio_snapshot: List[float]) -> None:
        try:
            y: NDArray[np.float64] = np.array(audio_snapshot).flatten()
            y_filtered: NDArray[np.float64] = self._apply_filter(y)
            y_preemph: NDArray[np.float64] = librosa.effects.preemphasis(y_filtered)
            peak_idx: int = int(np.argmax(np.abs(y_preemph)))

            start: int = max(0, peak_idx - int(self.fs * 0.05))
            end: int = min(len(y_preemph), peak_idx + int(self.fs * 0.15))
            y_focus: NDArray[np.float64] = y_preemph[start:end]

            pad_width: int = max(0, self.window_size - len(y_focus))
            y_focus = np.pad(y_focus, (0, pad_width), mode='constant')[:self.window_size]

            S: NDArray[np.float32] = librosa.feature.melspectrogram(
                y=y_focus.astype(np.float32), sr=self.fs, n_mels=128, fmax=self.fs // 2, n_fft=2048, hop_length=512
            )
            S_db: NDArray[np.float32] = librosa.power_to_db(S, ref=1.0)

            img: NDArray[np.float32] = resize(S_db, (self.img_size[1], self.img_size[0]), interpolation=INTER_CUBIC)
            img_norm: NDArray[np.float32] = (img - img.min()) / (img.max() - img.min() + 1e-9)

            prediction: NDArray[np.float32] = self.model.predict(img_norm[np.newaxis, ..., np.newaxis], verbose=0)
            class_idx: int = int(np.argmax(prediction))
            confidence: float = float(np.max(prediction))
            key: str = str(self.label_encoder.inverse_transform([class_idx])[0])

            def display_result() -> None:
                key_display: str = key.upper() if len(key) == 1 else key.capitalize()
                conf_bar: str = "█" * int(confidence * 20)
                print(f"[{key_display:8}] {conf_bar:20} {confidence * 100:5.1f}%", flush=True)

            is_valid: bool = confidence >= self.confidence_threshold and not (key == 'unknown' and confidence < 0.6)
            if is_valid: display_result()

        except Exception:
            pass
        finally:
            self.is_processing = False

    def _callback(self, indata: NDArray[np.float32], frames: int,
                  time_info: Any, status: sd.CallbackFlags) -> None:
        mono_chunk: NDArray[np.float32] = indata.mean(axis=1) if indata.ndim > 1 else indata.flatten()
        self.audio_buffer.extend(mono_chunk.tolist())

        if not self.is_processing and float(np.max(np.abs(mono_chunk))) > self.threshold:
            self.is_processing = True
            snapshot: List[float] = list(self.audio_buffer)[-int(0.5 * self.fs):]
            threading.Thread(target=self._process_event, args=(snapshot,), daemon=True).start()

    def run(self) -> None:
        signal.signal(signal.SIGINT, lambda sig, frame: self.stop_event.set())
        try:
            with sd.InputStream(device=self.device_id, channels=1, samplerate=self.fs, callback=self._callback):
                self.stop_event.wait()
        except Exception:
            pass

if __name__ == "__main__":
    predictor: AcousticKeyPredictor = AcousticKeyPredictor()
    predictor.run()