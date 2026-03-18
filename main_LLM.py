# main_LLM.py (主程式)
import argparse
import logging
import time
from llm_modules.utils import LLMConfigManager, setup_logger, setup_seed
from llm_modules.Pipeline import ExperimentPipeline

def main():
    parser = argparse.ArgumentParser(description="LLM Inference Runner")
    parser.add_argument('--config', type=str, default='configs/llm_config.yaml', help='Path to YAML config file')
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M")
    setup_logger(log_dir="./logs", log_name=f"llm_inference_{timestamp}.log")
    setup_seed(42)
    
    logging.info("========================================")
    logging.info("        Ollama            ")
    logging.info("========================================")

    try:
        # 1. 載入並解析設定檔與路徑
        config_manager = LLMConfigManager(args.config)
        
        # 2. 將 ConfigManager 傳給 Pipeline
        pipeline = ExperimentPipeline(config_manager.config)
        
        # 3. 執行！
        pipeline.run()
        
    except Exception as e:
        logging.critical(f"❌ 發生未預期的錯誤: {e}", exc_info=True)

if __name__ == "__main__":
    main()