import pandas as pd
import os
import time
import re
import logging 

class LLMResultProcessor:
    def __init__(self, input_csv_path, output_dir):
        """
        初始化資料處理器
        :param input_csv_path: 原始 LLM 輸出結果 (Raw Output)
        :param output_dir: 處理後檔案的儲存目錄
        """
        self.input_csv_path = input_csv_path
        self.output_dir = output_dir
        self.required_cols = ['Model', 'Prompt_ID', 'Pred_Label', 'True_Label']
        self.index_cols = ['Data_ID', 'PMID', 'E1', 'E2']
        self.raw_df = None
        self.pivot_df = None
        
        logging.info(f"LLMResultProcessor(input_csv_path='{self.input_csv_path}', output_dir='{self.output_dir}')")

    def process(self):
        """
        [Public] 執行完整的處理流程：讀取 -> 解析 -> 轉置 -> 存檔
        :return: 處理後的檔案路徑 (str) 或 None
        """
        logging.info(f"process(self=<{self.__module__}.{self.__class__.__name__} object at {hex(id(self))}>)")
        
        logging.info(f"Processing data: {self.input_csv_path}")
        
        if not self._load_data():
            return None
 
        self.raw_df['Pred_Numeric'] = self.raw_df['Pred_Label'].apply(self._parse_response)
        
        logging.info("Processing True Labels")
        self.raw_df['True_Numeric'] = self.raw_df['True_Label'].apply(
            lambda x: 1 if str(x).strip().lower() == 'cid' else 0
        )

        logging.info("Creating Feature Names")
        self.raw_df['Feature_Name'] = self.raw_df['Model'].astype(str) + "_" + self.raw_df['Prompt_ID'].astype(str)
        

        if not self._pivot_data():
            return None
            
        result_path = self._save_data()  
        return result_path

    def _load_data(self):
        """[Private] 讀取並驗證 CSV"""
        logging.info("Loading Raw CSV Data...")
        if not os.path.exists(self.input_csv_path):
            logging.error(f"❌ Error: File not found: {self.input_csv_path}")
            return False
            
        try:
            self.raw_df = pd.read_csv(self.input_csv_path)
            logging.info(f"Data loaded successfully. Shape: {self.raw_df.shape}")
            
            missing = [c for c in self.required_cols + self.index_cols if c not in self.raw_df.columns]
            if missing:
                logging.error(f"❌ Error: Missing columns: {missing}")
                return False
            return True
            
        except Exception as e:
            logging.error(f"❌ Error reading CSV: {e}")
            return False

    def _parse_response(self, text):
        """
        [Private] 解析單一回應字串
        :return: 1 (Yes), 0 (No), -1 (Unknown)
        """
        if pd.isna(text):
            return -1
            
        s = str(text).lower().strip()
        
        s = s.replace('*', '').replace('`', '').replace('#', '').replace('"', '').replace("'", "")
        s = s.rstrip('.,!')

        if s in ['yes', '1']: return 1
        if s in ['no', '0']: return 0

        if re.search(r'\byes\b', s): return 1
        if re.search(r'\bno\b', s): return 0

        if 'positive' in s: return 1
        if 'negative' in s: return 0

        return -1 # 解析失敗

    def _pivot_data(self):
        """[Private] 將長表格轉為寬表格"""
        logging.info("Pivoting table (Long to Wide)")
        try:
            self.pivot_df = self.raw_df.pivot_table(
                index=self.index_cols + ['True_Numeric'], 
                columns='Feature_Name', 
                values='Pred_Numeric',
                aggfunc='first' 
            )
            
            # 整理表格
            self.pivot_df = self.pivot_df.reset_index()
            self.pivot_df = self.pivot_df.fillna(-1) 
            logging.info(f"Pivot completed. New Shape: {self.pivot_df.shape}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Pivot failed: {e}")
            return False

    def _save_data(self):
        """[Private] 儲存結果"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M")
            save_path = os.path.join(self.output_dir, f"LLM_result_{timestamp}.csv")
            
            self.pivot_df.to_csv(save_path, index=False, encoding='utf-8-sig')
            
            # 統計解析成功率
            valid_count = (self.raw_df['Pred_Numeric'] != -1).sum()
            total_count = len(self.raw_df)
            
            logging.info("✅ Data processed successfully!")
            logging.info(f"   - Shape: {self.pivot_df.shape}")
            logging.info(f"   - Parse Success Rate: {valid_count}/{total_count} ({valid_count/total_count:.1%})")
            logging.info(f"   - Saved to: {save_path}")
            
            return save_path
            
        except Exception as e:
            logging.error(f"❌ Error saving file: {e}")
            return None