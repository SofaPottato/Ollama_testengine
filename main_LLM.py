import argparse
import logging
import time
from modules.utils import load_config, setup_logger, setup_seed
from modules.Pipeline import ExperimentPipeline

def main():
    # 1. 解析命令列參數
    parser = argparse.ArgumentParser(description="LLM Experiment Runner")
    parser.add_argument('--config', type=str, default='config/llm_config.yaml', help='Path to YAML config file')
    args = parser.parse_args()

    # 2. 初始化環境
    timestamp = time.strftime("%Y%m%d_%H%M")
    setup_logger(log_dir="./logs", log_name=f"experiment{timestamp}.log")
    setup_seed(42)  # 固定種子，保證實驗可重現
    logging.info("========================================")
    logging.info("      LLM Experiment Framework v2.0     ")
    logging.info("========================================")

    # 3. 讀取設定檔
    cfg = load_config(args.config)

    # 4. 啟動 Pipeline
    try:
        pipeline = ExperimentPipeline(cfg)
        pipeline.run()
    except Exception as e:
        logging.critical(f"❌ Unhandled Exception in Pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    main()