import logging
import threading
import time
import queue
from typing import Callable, List, Optional, Union

import numpy as np
import sounddevice as sd
from scipy.signal import resample
from openwakeword.model import Model

import sys

if sys.platform == 'win32':
    from openwakeword.utils import download_models

class WakeWordDetector:
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1280

    def __init__(
        self,
        wakeword_models: Union[str, List[str]] = "alexa",
        threshold: float = 0.5,
        cooldown_sec: float = 2.0,
        init_delay_sec: float = 0.0,
        detection_callback: Optional[Callable[[str], None]] = None,
        input_device: Optional[Union[int, str]] = None,
        custom_logger: Optional[logging.Logger] = None,
    ) -> None:
        self.wakeword_models = (
            [wakeword_models] if isinstance(wakeword_models, str) else wakeword_models
        )
            
        self.threshold = threshold
        self.cooldown_sec = cooldown_sec
        self.init_delay_sec = init_delay_sec
        self.callback = detection_callback
        self.input_device = input_device
        
        self.logger = custom_logger if custom_logger is not None else logging.getLogger(self.__class__.__name__)
        
        self._is_paused = False
        self._last_detection_time = 0.0
        self._stream_start_time = 0.0
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._actual_sample_rate = self.SAMPLE_RATE
        self._audio_queue: queue.Queue = queue.Queue(maxsize=50)
        self._stream: Optional[sd.InputStream] = None

        if sys.platform == 'win32':
            try:
                from openwakeword.utils import download_models
                self.logger.info("Проверка наличия базовых моделей (может потребоваться загрузка)...")
                download_models()
            except ImportError:
                self.logger.warning("Не удалось импортировать download_models. Убедитесь, что модели скачаны вручную.")

        try:
            # openwakeword >= 0.5.0
            self.oww_model = Model(wakeword_models=self.wakeword_models, inference_framework="onnx")
        except TypeError:
            # openwakeword 0.4.x
            self.oww_model = Model(wakeword_model_paths=self.wakeword_models)
        self.logger.info("Модели успешно загружены. Детектор готов к работе.")

    def pause(self) -> None:
        self._is_paused = True
        self._close_stream()
        self.logger.debug("Распознавание поставлено на паузу. Микрофон освобождён.")

    def unpause(self) -> None:
        self._is_paused = False
        self.oww_model.reset()
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        self._pause_event.set()
        self.logger.debug("Распознавание возобновлено.")

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.set()
        self._close_stream()

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _open_stream(self, actual_blocksize: int) -> None:
        self._stream = sd.InputStream(
            device=self.input_device,
            samplerate=self._actual_sample_rate,
            channels=1,
            dtype='int16',
            blocksize=actual_blocksize,
            callback=self._audio_callback
        )
        self._stream.start()

    def _audio_callback(
        self, 
        indata: np.ndarray, 
        frames: int, 
        time_info: dict, 
        status: sd.CallbackFlags
    ) -> None:
        if self._is_paused:
            return
        try:
            self._audio_queue.put_nowait(indata.copy())
        except queue.Full:
            pass

    def _prediction_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                indata = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            current_time = time.time()

            if self._stream_start_time > 0 and (current_time - self._stream_start_time) < self.init_delay_sec:
                continue

            if current_time - self._last_detection_time < self.cooldown_sec:
                continue

            audio_data = indata.flatten().astype(np.int16)

            if self._actual_sample_rate != self.SAMPLE_RATE:
                num_samples = int(len(audio_data) * self.SAMPLE_RATE / self._actual_sample_rate)
                audio_data = resample(audio_data, num_samples).astype(np.int16)

            prediction = self.oww_model.predict(audio_data)

            best_wakeword = max(prediction, key=prediction.get)
            prob = prediction[best_wakeword]

            self.logger.debug(f"Текущая вероятность ('{best_wakeword}'): {prob * 100:.2f}%")

            if prob > self.threshold:
                self._last_detection_time = current_time
                self.oww_model.reset()

                if self.callback:
                    threading.Thread(target=self.callback, args=(best_wakeword,), daemon=True).start()

    def _detect_sample_rate(self) -> int:
        for rate in [self.SAMPLE_RATE, 44100, 48000]:
            try:
                sd.check_input_settings(device=self.input_device, samplerate=rate, channels=1, dtype='int16')
                return int(rate)
            except Exception:
                continue
        try:
            info = sd.query_devices(self.input_device, 'input')
            return int(info['default_samplerate'])
        except Exception:
            return self.SAMPLE_RATE

    def start(self) -> None:
        self._stop_event.clear()
        try:
            self._actual_sample_rate = self._detect_sample_rate()
            actual_blocksize = int(self.CHUNK_SIZE * self._actual_sample_rate / self.SAMPLE_RATE)

            if self._actual_sample_rate != self.SAMPLE_RATE:
                self.logger.info(f"Устройство не поддерживает {self.SAMPLE_RATE} Hz. "
                                 f"Используется {self._actual_sample_rate} Hz с ресемплированием.")

            worker = threading.Thread(target=self._prediction_worker, daemon=True)
            worker.start()

            while not self._stop_event.is_set():
                self._pause_event.clear()
                self.logger.info("Запуск захвата аудио...")
                try:
                    self._open_stream(actual_blocksize)
                    self._stream_start_time = time.time()
                    self.logger.info("Слушаю... Нажмите Ctrl+C для выхода.")

                    while not self._stop_event.is_set() and not self._is_paused:
                        time.sleep(0.1)

                    if self._is_paused and not self._stop_event.is_set():
                        self._pause_event.wait()
                except Exception as e:
                    self.logger.error(f"Ошибка в аудиопотоке: {e}", exc_info=True)
                    break

        except KeyboardInterrupt:
            self.logger.info("Остановлено пользователем.")
        finally:
            self.stop()
            self.logger.info("Детектор остановлен.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    def my_action(detected_word: str) -> None:
        detector.pause()
        
        print(f"\n[!] Внимание: Обнаружено слово {detected_word}. Выполняем паузу.")
        time.sleep(2)
        print("[!] Команда завершена. Возобновляем прослушивание.\n")
        
        detector.unpause()

    detector = WakeWordDetector(
        wakeword_models="alexa", 
        threshold=0.55, 
        cooldown_sec=2.0, 
        init_delay_sec=2.0,
        detection_callback=my_action,
        input_device=None
    )
    
    detector.logger.setLevel(logging.DEBUG)
    
    detector.start()