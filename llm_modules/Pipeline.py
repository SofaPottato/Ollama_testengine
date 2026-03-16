import logging
import os
import sys
from pathlib import Path
import pandas as pd
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
        logging.info("Initializing ExperimentPipeline()")
        self.cfg = config
        self.data_path = Path(config["data_path"])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"ExperimentPipeline(config_keys={list(config.keys())}, output_dir='{self.output_dir}')")
    
    def run(self):  
        """執行實驗流程"""
               
        logging.info(f"==== [Step 1] Loading Data from: {self.data_path} ====")
        df = self._load_data()
        if df is None:
            logging.critical("❌ Data loading failed. Pipeline aborted.")
            raise RuntimeError("Data loading failed")

        logging.info("==== [Step 2] Generating Prompts ====")
        pm = PromptManager('config/prompts.yaml')
        prompts = pm.generate_combinations(self.cfg)
        
        if not prompts:
            logging.error("❌ No prompts generated. Aborting.")
            return

        logging.info("==== [Step 3] Running LLM ====")
        engine = InferenceManager(self.cfg)
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
        data_path = self.cfg.get('data_path')

        if not data_path or not os.path.exists(data_path):
            logging.error(f"❌ Data file not found: {data_path}")
            return None
            
        try:
            df = pd.read_csv(data_path)
            
            if self.cfg.get('test_limit'):
                limit = self.cfg['test_limit']
                df = df.head(limit)
                logging.warning(f"⚠️test:Using only first {limit} pairs.")
            
            logging.info(f"✅ Data loaded. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logging.error(f"❌ Failed to load data: {e}")
            return None