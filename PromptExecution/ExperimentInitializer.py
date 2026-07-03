"""實驗啟動前的環境設定：logger、隨機種子。"""
import logging
import os
import random
import sys

import numpy as np


class ExperimentInitializer:
    """實驗啟動前的環境設定：logger、隨機種子。"""

    def initializeGlobalLogger(self, logDir: str = "data/logs", logName: str = "experiment.log") -> None:
        """設定全域 Logger，同時輸出到檔案與標準輸出。

        httpx logger 拉到 WARNING，避免推論時被連線層 INFO 訊息淹沒。
        """
        os.makedirs(logDir, exist_ok=True)
        logPath = os.path.join(logDir, logName)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s:%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
            handlers=[
                logging.FileHandler(logPath, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.info(f"[Logger] 初始化完成 → {logPath}")

    def setupSeed(self, seed: int = 42) -> None:
        """固定 Python random / NumPy / PYTHONHASHSEED，確保實驗可重現。"""
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        logging.info(f"[Setup] 隨機種子設定為 {seed}")
