import sounddevice as sd
import numpy as np
import os
import csv
import struct
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from datetime import datetime
from typing import Optional, Any
from numpy.typing import NDArray


class UnknownDataCollector:
    fs: int
    device_id: Optional[int]
    block_size: int
    output_dir: str
    csv_path: str
    kbd_output: str
    window_size: int
    b: NDArray[np.float64]
    a: NDArray[np.float64]
    collected_count: int

    def __init__(self, device_id: Optional[int] = None, fs: int = 48000, interval_sec: float = 1.0) -> None:
        self.fs = fs
        self.device_id = device_id
        self.block_size = int(fs * interval_sec)
        self.output_dir = "samples_pro"
        self.csv_path = "metadata_pro.csv"
        self.kbd_output = "training_data.kbd"
        self.window_size = int(self.fs * 0.20)

        self.b, self.a = butter(4, [200 / (fs / 2), 10000 / (fs / 2)], btype='band')
        self.collected_count = 0

        os.makedirs(self.output_dir, exist_ok=True)
        self._prepare_files()

    def _prepare_files(self) -> None:
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(['timestamp', 'key_id', 'wav_path', 'peak_val', 'rms_val'])

        if not os.path.exists(self.kbd_output):
            with open(self.kbd_output, "wb") as f:
                f.write(struct.pack('i', self.window_size))

    def _apply_filter(self, data: NDArray[np.float32]) -> NDArray[np.float64]:
        return lfilter(self.b, self.a, data)

    def _audio_callback(self, indata: NDArray[np.float32], frames: int,
                        time_info: Any, status: sd.CallbackFlags) -> None:
        if status:
            print(f"Status: {status}")

        data_to_process: NDArray[np.float32] = indata[:self.window_size, 0].flatten()
        self._save_sample(data_to_process)

        self.collected_count += 1
        if self.collected_count % 10 == 0:
            print(f"Zebrano: {self.collected_count} próbek tła")

    def capture_noise(self) -> None:
        print("=" * 60)
        print("KOLEKTOR ASYNCHRONICZNY")
        print("=" * 60)

        with sd.InputStream(
                samplerate=self.fs,
                device=self.device_id,
                channels=1,
                callback=self._audio_callback,
                blocksize=self.block_size
        ):
            print("Nagrywanie trwa... Naciśnij Enter, aby zakończyć.")
            input()

    def _save_sample(self, data: NDArray[np.float32]) -> None:
        filtered_data: NDArray[np.float64] = self._apply_filter(data)
        peak: float = float(np.max(np.abs(filtered_data)))
        rms: float = float(np.sqrt(np.mean(filtered_data ** 2)))

        ts: str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        key_id_str: str = "unknown"
        bin_key_id: int = 0
        filepath: str = os.path.join(self.output_dir, f"{key_id_str}_{ts}.wav")

        wavfile.write(filepath, self.fs, (filtered_data * 32767).astype(np.int16))

        with open(self.kbd_output, "ab") as f:
            f.write(struct.pack('i', bin_key_id))
            f.write(filtered_data.astype(np.float32).tobytes())

        with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([ts, key_id_str, filepath, f"{peak:.6f}", f"{rms:.6f}"])


if __name__ == "__main__":
    collector: UnknownDataCollector = UnknownDataCollector(interval_sec=1.0)
    collector.capture_noise()