import sounddevice as sd
import numpy as np
import os
import threading
import csv
import struct
import time
from pynput import keyboard
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from datetime import datetime
from collections import deque
from typing import Optional, Union, Deque, Any
from numpy.typing import NDArray


class AudioLogger:
    fs: int
    device_id: Optional[int]
    output_dir: str
    csv_path: str
    kbd_output: str
    window_size: int
    ring_buffer: Deque[float]
    threshold: float
    last_key: Optional[Union[keyboard.Key, keyboard.KeyCode]]
    last_key_time: Optional[float]
    is_processing: bool
    processing_lock: threading.Lock
    key_timeout: float
    b: NDArray[np.float64]
    a: NDArray[np.float64]

    def __init__(self, device_id: Optional[int] = None, fs: int = 48000) -> None:
        self.fs = fs
        self.device_id = device_id
        self.output_dir = "samples_pro"
        self.csv_path = "metadata_pro.csv"
        self.kbd_output = "training_data.kbd"
        self.window_size = int(self.fs * 0.20)
        self.ring_buffer = deque(maxlen=int(1.5 * fs))
        self.threshold = 0.015
        self.last_key = None
        self.last_key_time = None
        self.is_processing = False
        self.processing_lock = threading.Lock()
        self.key_timeout = 0.2
        self._prepare_storage()
        self.b, self.a = butter(4, [200 / (fs / 2), 10000 / (fs / 2)], btype='band')

    def _prepare_storage(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(['timestamp', 'key_id', 'wav_path', 'peak_val', 'rms_val'])
        with open(self.kbd_output, "wb") as f:
            f.write(struct.pack('i', self.window_size))

    def _apply_filter(self, data: NDArray[Any]) -> NDArray[np.float64]:
        return lfilter(self.b, self.a, data)

    def _audio_callback(self, indata: NDArray[np.float32], frames: int,
                        time_info: Any, status: sd.CallbackFlags) -> None:
        if status:
            print(f"Audio warning: {status}")

        mono_chunk: NDArray[np.float32] = np.mean(indata, axis=1) if indata.ndim > 1 else indata.flatten()
        self.ring_buffer.extend(mono_chunk.tolist())

        with self.processing_lock:
            if self.last_key is not None and not self.is_processing:
                if self.last_key_time and (time.time() - self.last_key_time) > self.key_timeout:
                    self.last_key = None
                    self.last_key_time = None
                    return

                current_energy: float = float(np.max(np.abs(mono_chunk)))
                if current_energy > self.threshold:
                    self.is_processing = True
                    threading.Thread(
                        target=self._process_detection,
                        args=(self.last_key,),
                        daemon=True
                    ).start()

    def _process_detection(self, key_id: Union[keyboard.Key, keyboard.KeyCode]) -> None:
        try:
            raw_data: NDArray[np.float32] = np.array(self.ring_buffer)
            filtered_data: NDArray[np.float64] = self._apply_filter(raw_data)
            search_window: int = min(len(filtered_data), int(self.fs * 0.5))
            search_data: NDArray[np.float64] = filtered_data[-search_window:]
            peak_idx_relative: int = int(np.argmax(np.abs(search_data)))
            peak_idx: int = len(filtered_data) - search_window + peak_idx_relative
            start_offset: int = int(self.fs * 0.05)
            start: int = peak_idx - start_offset
            end: int = start + self.window_size

            segment: NDArray[np.float64]
            if start < 0:
                segment = np.concatenate([np.zeros(-start), filtered_data[0:end]])
            elif end > len(filtered_data):
                segment = np.concatenate([filtered_data[start:], np.zeros(end - len(filtered_data))])
            else:
                segment = filtered_data[start:end]

            segment = segment[:self.window_size]
            peak_val: float = float(np.max(np.abs(segment)))
            rms_val: float = float(np.sqrt(np.mean(segment ** 2)))

            if len(segment) == self.window_size and peak_val > self.threshold:
                self._save_sample(key_id, segment, peak_val, rms_val)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            with self.processing_lock:
                self.last_key = None
                self.last_key_time = None
                self.is_processing = False

    def _save_sample(self, key_id: Union[keyboard.Key, keyboard.KeyCode],
                     data: NDArray[np.float64], peak: float, rms: float) -> None:
        ts: str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        key_name: str = str(key_id).replace("'", "").replace("Key.", "").lower()

        if not key_name.strip() or key_name == "key.space":
            key_name = "space"
        elif key_name.startswith("key."):
            key_name = key_name.replace("key.", "")

        filename: str = f"{key_name}_{ts}.wav"
        path: str = os.path.join(self.output_dir, filename)
        wav_data: NDArray[np.int16] = (data * 32767).astype(np.int16)
        wavfile.write(path, self.fs, wav_data)

        try:
            char_code: int = ord(key_name[0]) if len(key_name) == 1 else 0
            with open(self.kbd_output, "ab") as f:
                f.write(struct.pack('i', char_code))
                f.write(data.astype(np.float32).tobytes())
        except Exception:
            pass

        with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([ts, key_name, path, f"{peak:.6f}", f"{rms:.6f}"])

    def on_press(self, key: Union[keyboard.Key, keyboard.KeyCode]) -> None:
        with self.processing_lock:
            if not self.is_processing:
                self.last_key = key
                self.last_key_time = time.time()

    def run(self) -> None:
        try:
            with sd.InputStream(
                    channels=1,
                    samplerate=self.fs,
                    callback=self._audio_callback,
                    device=self.device_id
            ):
                with keyboard.Listener(on_press=self.on_press) as listener:
                    listener.join()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    logger: AudioLogger = AudioLogger(fs=48000)
    logger.run()