import yaml
import logging
import os
import random
import numpy as np
import sys
import functools
import time
from pathlib import Path
from typing import Dict, List, Any

class LLMConfigManager:
    """專門負責讀取 LLM 設定檔並處理型別與路徑"""
    
    def __init__(self, config_path: str):
        # 使用 Dict[str, Any] 確保型別清晰
        self.config: Dict[str, Any] = self._load_yaml(config_path)
        
        # 將 paths 區塊獨立抽出來，轉換為 Path 物件
        self.paths: Dict[str, Path] = self._parse_paths()
        
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """讀取 YAML 檔案"""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _parse_paths(self) -> Dict[str, Path]:
        """將 YAML 中的 paths 字串轉換為 pathlib.Path 物件"""
        parsed_paths: Dict[str, Path] = {}
        raw_paths: Dict[str, str] = self.config.get('paths', {})
        
        for key, path_str in raw_paths.items():
            path_obj = Path(path_str)
            parsed_paths[key] = path_obj
            
            # 如果是輸出資料夾 (結尾是 _dir)，自動幫忙建立目錄
            if key.endswith('_dir'):
                path_obj.mkdir(parents=True, exist_ok=True)
                
        return parsed_paths

    def get_models(self) -> List[str]:
        """取得模型清單 (回傳 List[str])"""
        return self.config.get('selected_models', [])
    
def load_config(path):
    """
    讀取 YAML 設定檔
    :param path: 設定檔路徑 (例如 'config/llm_config.yaml')
    :return: 字典格式的設定內容
    """
    if not os.path.exists(path):
        logging.critical(f"❌ Critical Error: Config file not found at {path}")
        raise RuntimeError("Data loading failed")
        
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            logging.info(f"✅ 設定檔已載入: {path}")
            return config
    except Exception as e:
        logging.error(f"❌ Error loading config: {e}")
        raise RuntimeError("Data loading failed")

def setup_logger(log_dir="./logs", log_name="experiment.log"):
    """
    設定全域 Logger，同時輸出到檔案與終端機
    :param log_dir: Log 檔案存放目錄
    :param log_name: Log 檔名
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s", 
        force=True, 
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'), 
            logging.StreamHandler(sys.stdout)               
        ]
    )
    logging.info(f"📝 Logger initialized. Writing to {log_path}")

def setup_seed(seed=42):
    """
    固定隨機種子，確保實驗可重現 (Reproducibility)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"Random seed set to {seed}")