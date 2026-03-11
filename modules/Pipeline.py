import logging
import os
import sys
from pathlib import Path
import pandas as pd

# 引入所有模組
from .PromptManager import PromptManager
from .LLM_Engine import InferenceManager
from .LLMResultProcessor import LLMResultProcessor
from .Evaluate import LLMEvaluationSystem

class ExperimentPipeline:
    def __init__(self, config):
        """
        初始化實驗流程
        :param config: 從 yaml 讀入的完整設定字典
        """
        self.cfg = config
        self.output_dir = Path(config['output_dir'])
        
        # 確保輸出目錄存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self):  
        """執行完整的實驗流程"""
        logging.info("🚀 Pipeline Started")
        
        # --- 步驟 1: 準備資料 ---
        df = self._load_data()
        
        # 防呆機制：如果讀取失敗，直接終止
        if df is None:
            logging.critical("❌ Data loading failed. Pipeline aborted.")
            sys.exit(1)

        # --- 步驟 2: 生成 Prompt ---
        logging.info("--- [Step 2] Generating Prompts ---")
        pm = PromptManager('config/prompts.yaml') #吃路徑
        prompts = pm.generate_combinations(self.cfg)
        
        if not prompts:
            logging.error("❌ No prompts generated. Aborting.")
            return
        
        # --- 步驟 3: 執行推論 (Inference) ---
        logging.info("--- [Step 3] Running Inference Engine ---")
        engine = InferenceManager(self.cfg)
        raw_csv_path = engine.run(df, prompts)
        
        if not raw_csv_path:
            logging.error("❌ Inference failed or produced no output.")
            return

        # --- 步驟 4: 資料清洗與格式轉換 (Processing) ---
        logging.info("--- [Step 4] Processing Results ---")
        proc_dir = self.output_dir / "processed_result"
        processor = LLMResultProcessor(raw_csv_path, str(proc_dir))
        processed_csv_path = processor.process()

        if not processed_csv_path:
            logging.error("❌ Data processing failed.")
            return

        # --- 步驟 5: 評估 (Evaluation) ---
        logging.info("--- [Step 5] Evaluating Performance ---")
        eval_dir = self.output_dir / "eval_results"
        evaluator = LLMEvaluationSystem(processed_csv_path, str(eval_dir))
        
        evaluator.run_evaluation()      # 計算指標
        evaluator.analyze_difficulty()  # 計算難題與上限
        evaluator.plot_confusion_matrices() # 畫混淆矩陣
        evaluator.plot_heatmap()        # 畫熱圖
        evaluator.save_results()        # 存檔
        
        logging.info(f"Pipeline Completed Successfully! Results at: {eval_dir}")

    def _load_data(self):
        """讀取並預處理原始資料"""
        data_path = self.cfg.get('data_path')
        
        logging.info(f"--- [Step 1] Loading Data from: {data_path} ---")
        
        if not data_path or not os.path.exists(data_path):
            logging.error(f"❌ Data file not found: {data_path}")
            return None
            
        try:
            df = pd.read_csv(data_path)
            
            if self.cfg.get('test_limit'):
                limit = self.cfg['test_limit']
                df = df.head(limit)
                logging.warning(f"⚠️ Test Mode: Using only first {limit} rows.")
            
            logging.info(f"✅ Data loaded. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logging.error(f"❌ Failed to load data: {e}")
            return None

