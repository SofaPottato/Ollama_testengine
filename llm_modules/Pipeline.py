import logging
import os
import sys
from pathlib import Path
import pandas as pd

# 注意：這裡我們移除了對 PromptManager 的 import，降低耦合！
from .LLM_Engine import InferenceManager
from .LLMResultProcessor import LLMResultProcessor
from .Evaluate import LLMEvaluationSystem

class ExperimentPipeline:
    
    def __init__(self, config):
        """
        初始化實驗流程
        :param config: 從 yaml 讀入的完整設定字典
        """
        logging.info("Initializing ExperimentPipeline()")
        self.cfg = config
        
        # 🌟 關鍵修改：先取得 'paths' 這個字典區塊
        paths_cfg = config.get("paths", {})
        
        # 從 paths_cfg 中提取所有需要的路徑
        self.data_path = Path(paths_cfg.get("dataPath", "data/bcvcdr_raw/BCVCDR_Processed.csv"))
        
        # 你的 yaml 輸出目錄叫做 mainOutputDir
        self.output_dir = Path(paths_cfg.get("mainOutputDir", "data/llm_output/"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 讀取 Prompts 路徑
        self.prompts_path = Path(paths_cfg.get("promptsPath", "data/prompt_output/generatedPromptList.csv"))
        
        logging.info(f"ExperimentPipeline initialized, output_dir='{self.output_dir}'")
    
    def run(self):  
        """執行實驗流程"""
               
        logging.info(f"==== [Step 1] Loading Data from: {self.data_path} ====")
        df = self._load_data()
        if df is None:
            logging.critical("❌ Data loading failed. Pipeline aborted.")
            raise RuntimeError("Data loading failed")

        # ---------------------------------------------------------
        # 修改區塊：[Step 2] 改為直接從 CSV 讀取 Prompts
        # ---------------------------------------------------------
        logging.info(f"==== [Step 2] Loading Prompts from: {self.prompts_path} ====")
        prompts = self._load_prompts(self.prompts_path)
        
        if not prompts:
            logging.error("❌ No prompts loaded. Aborting.")
            return

        logging.info("==== [Step 3] Running LLM ====")
        engine = InferenceManager(self.cfg)
        # 這裡傳入的 prompts 已經是 [{'Prompt_ID': '...', 'Prompt_Text': '...'}, ...] 的格式
        raw_csv_path = engine.run(df, prompts)
        
        if not raw_csv_path:
            logging.error("❌ Inference failed or produced no output.")
            return

        logging.info("==== [Step 4] Processing Results ====")
        proc_dir = self.output_dir / "processed_result"
        processor = LLMResultProcessor(raw_csv_path, str(proc_dir))
        processed_csv_path = processor.process()

        if not processed_csv_path:
            logging.error("❌ Data processing failed.")
            return

        logging.info("==== [Step 5] Evaluate ====")
        eval_dir = self.output_dir / "eval_results"
        evaluator = LLMEvaluationSystem(processed_csv_path, str(eval_dir))
        
        evaluator.run_evaluation()     
        evaluator.analyze_difficulty()  
        evaluator.plot_confusion_matrices() 
        evaluator.plot_heatmap()        
        evaluator.save_results()        
        
    def _load_data(self):
        """讀取原始資料"""
        if not self.data_path or not os.path.exists(self.data_path):
            logging.error(f"❌ Data file not found: {self.data_path}")
            return None
            
        try:
            df = pd.read_csv(self.data_path)
            
            if self.cfg.get('test_limit'):
                limit = self.cfg['test_limit']
                df = df.head(limit)
                logging.warning(f"⚠️test:Using only first {limit} pairs.")
            
            logging.info(f"✅ Data loaded. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logging.error(f"❌ Failed to load data: {e}")
            return None

    # ---------------------------------------------------------
    # 新增區塊：專屬的 Prompt 讀取模組
    # ---------------------------------------------------------
    def _load_prompts(self, path: Path) -> list:
        """
        讀取 Prompt CSV 檔案並轉換為字典列表。
        這樣做可以確保 Pipeline 與特定的 Prompt 生成邏輯解耦。
        """
        if not path.exists():
            logging.error(f"❌ Prompt CSV file not found: {path}")
            return []
            
        try:
            # 讀取您提供的 test_manual_prompts.csv
            df_prompts = pd.read_csv(path)
            
            # 檢查是否包含預期的欄位，避免後續 LLM Engine 抓不到資料
            if 'Prompt_ID' not in df_prompts.columns or 'Prompt_Text' not in df_prompts.columns:
                logging.error("❌ CSV missing required columns: 'Prompt_ID' or 'Prompt_Text'")
                return []

            # 將 DataFrame 轉換為 list of dictionaries
            # 格式範例：[{'Prompt_ID': 'EMO01 + RAR02', 'Prompt_Text': 'This is very important...'}, ...]
            prompts_list = df_prompts.to_dict(orient='records')
            logging.info(f"✅ Successfully loaded {len(prompts_list)} prompts.")
            
            return prompts_list
            
        except Exception as e:
            logging.error(f"❌ Failed to load prompt CSV: {e}")
            return []