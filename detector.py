import logging
import threading
import time
from typing import Callable, List, Optional, Union

import numpy as np
import sounddevice as sd
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

        if sys.platform == 'win32':
            self.logger.info("Инициализация моделей openWakeWord (может потребоваться загрузка)...")
            download_models()

        self.oww_model = Model(wakeword_models=self.wakeword_models, inference_framework="onnx")
        self.logger.info("Модели успешно загружены. Детектор готов к работе.")

    def pause(self) -> None:
        self._is_paused = True
        self.logger.debug("Распознавание поставлено на паузу.")

    def unpause(self) -> None:
        self._is_paused = False
        self.oww_model.reset()
        self.logger.debug("Распознавание возобновлено.")

    def stop(self) -> None:
        self._stop_event.set()

    def _process_audio(
        self, 
        indata: np.ndarray, 
        frames: int, 
        time_info: dict, 
        status: sd.CallbackFlags
    ) -> None:
        if status:
            self.logger.warning(f"Статус аудиопотока: {status}")
            
        if self._is_paused:
            return

        current_time = time.time()
        
        if self._stream_start_time > 0 and (current_time - self._stream_start_time) < self.init_delay_sec:
            return

        if current_time - self._last_detection_time < self.cooldown_sec:
            return

        audio_data = indata.flatten()
        
        prediction = self.oww_model.predict(audio_data)
        
        best_wakeword = max(prediction, key=prediction.get)
        prob = prediction[best_wakeword]
        
        self.logger.debug(f"Текущая вероятность ('{best_wakeword}'): {prob * 100:.2f}%")
        
        if prob > self.threshold:
            self._last_detection_time = current_time
            self.oww_model.reset()
            
            if self.callback:
                threading.Thread(target=self.callback, args=(best_wakeword,), daemon=True).start()

    def start(self) -> None:
        self._stop_event.clear()
        try:
            self.logger.info("Запуск захвата аудио...")
            with sd.InputStream(
                device=self.input_device, 
                samplerate=self.SAMPLE_RATE, 
                channels=1, 
                dtype='int16', 
                blocksize=self.CHUNK_SIZE, 
                callback=self._process_audio
            ):
                self._stream_start_time = time.time()
                self.logger.info("Слушаю... Нажмите Ctrl+C для выхода.")
                self._stop_event.wait()
        except KeyboardInterrupt:
            self.logger.info("Остановлено пользователем.")
        except Exception as e:
            self.logger.error(f"Ошибка в аудиопотоке: {e}", exc_info=True)
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