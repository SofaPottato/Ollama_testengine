import yaml
import logging
import os
import random
import numpy as np
import sys
import functools
import time


def llm_logger(func):#todo
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        func_name = func.__name__
        
        safe_args = []
        for a in args:
            if hasattr(a, 'shape'): 
                safe_args.append(f"DataFrame{a.shape}")
            elif isinstance(a, dict) and len(a) > 5:
                safe_args.append(f"LargeConfig(keys={list(a.keys())[:5]}...)")
            else:
                safe_args.append(a)
                
        logging.info(f"Running {self.__class__.__name__}.{func_name}")
        if safe_args or kwargs:
            logging.info(f"Params: {safe_args} {kwargs if kwargs else ''}")

        try:
            result = func(self, *args, **kwargs)
            logging.info(f"{func_name}() successfully completed")
            return result
        except Exception as e:
            logging.error(f"❌ Error in {func_name}: {e}", exc_info=True)
            raise e
    return wrapper

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