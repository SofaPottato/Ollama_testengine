import yaml
import logging
import os
import random
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any
from .schemas import LLMAppConfig
class LLMConfigManager:
    """專門負責讀取 LLM 設定檔並驗證型別與路徑"""
    def __init__(self, configPath: str):
        # 第一步：讀取原始的 YAML 字典
        raw_yaml_dict = self.loadYamlConfiguration(configPath)
        self.config: LLMAppConfig = LLMAppConfig(**raw_yaml_dict)
        
    def loadYamlConfiguration(self, path: str) -> Dict[str, Any]:
        """讀取 YAML 檔案"""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)


def initializeGlobalLogger(logDir="./logs", logName="experiment.log"):
    """
    設定全域 Logger，同時輸出到檔案與終端機
    :param logDir: Log 檔案存放目錄
    :param logName: Log 檔名
    """
    os.makedirs(logDir, exist_ok=True)
    logPath = os.path.join(logDir, logName)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s", 
        force=True, 
        handlers=[
            logging.FileHandler(logPath, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)               
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.info(f"📝 Logger initialized. Writing to {logPath}")

def setupSeed(seed=42):
    """
    固定隨機種子，確保實驗可重現 (Reproducibility)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"Random seed set to {seed}")