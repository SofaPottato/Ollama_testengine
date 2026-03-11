import yaml
import logging
import os
import random
import numpy as np
import sys

def load_config(path):
    """
    讀取 YAML 設定檔
    :param path: 設定檔路徑 (例如 'config/llm_config.yaml')
    :return: 字典格式的設定內容
    """
    if not os.path.exists(path):
        print(f"❌ Critical Error: Config file not found at {path}")
        sys.exit(1)
        
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            print(f"✅ 設定檔已載入: {path}")
            return config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)

def setup_logger(log_dir="./logs", log_name="experiment.log"):
    """
    設定全域 Logger，同時輸出到檔案與終端機
    :param log_dir: Log 檔案存放目錄
    :param log_name: Log 檔名
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)

    # 設定 logging 格式
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'), # 輸出到檔案
            logging.StreamHandler(sys.stdout)               # 輸出到終端機
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