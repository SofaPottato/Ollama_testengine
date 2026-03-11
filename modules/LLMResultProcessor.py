import pandas as pd
import os
import time
import re

class LLMResultProcessor:
    def __init__(self, input_csv_path, output_dir):
        """
        初始化資料處理器
        :param input_csv_path: 原始 LLM 輸出結果 (Raw Output)
        :param output_dir: 處理後檔案的儲存目錄
        """
        self.input_csv_path = input_csv_path
        self.output_dir = output_dir
        
        # 定義必要欄位
        self.required_cols = ['Model', 'Prompt_ID', 'Pred_Label', 'True_Label']
        self.index_cols = ['Data_ID', 'PMID', 'E1', 'E2']
        
        # 狀態變數
        self.raw_df = None
        self.pivot_df = None
        
    def process(self):
        """
        [Public] 執行完整的處理流程：讀取 -> 解析 -> 轉置 -> 存檔
        :return: 處理後的檔案路徑 (str) 或 None
        """
        print(f"🔄 Processing data: {self.input_csv_path}")
        
        if not self._load_data():
            return None
            
        # 1. 解析預測結果 (Text -> 0/1/-1)
        print("🔍 Parsing LLM responses...")
        self.raw_df['Pred_Numeric'] = self.raw_df['Pred_Label'].apply(self._parse_response)
        
        # 2. 處理真實標籤 (True Label -> 0/1)
        # 假設 'cid' 為正樣本 (1)，其餘為負樣本 (0)
        self.raw_df['True_Numeric'] = self.raw_df['True_Label'].apply(
            lambda x: 1 if str(x).strip().lower() == 'cid' else 0
        )
        
        # 3. 建立特徵名稱 (Feature Name)
        self.raw_df['Feature_Name'] = self.raw_df['Model'].astype(str) + "_" + self.raw_df['Prompt_ID'].astype(str)
        
        # 4. 執行轉置 
        if not self._pivot_data():
            return None
            
        # 5. 存檔
        return self._save_data()

    def _load_data(self):
        """[Private] 讀取並驗證 CSV"""
        if not os.path.exists(self.input_csv_path):
            print(f"❌ Error: File not found: {self.input_csv_path}")
            return False
            
        try:
            self.raw_df = pd.read_csv(self.input_csv_path)
            
            # 檢查欄位
            missing = [c for c in self.required_cols + self.index_cols if c not in self.raw_df.columns]
            if missing:
                print(f"❌ Error: Missing columns: {missing}")
                return False
            return True
            
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return False

    def _parse_response(self, text):
        """
        [Private] 解析單一回應字串
        :return: 1 (Yes), 0 (No), -1 (Unknown)
        """
        if pd.isna(text):
            return -1
            
        # 1. 轉小寫並去除前後空白
        s = str(text).lower().strip()
        
        # 2. 去除 Markdown 與標點
        s = s.replace('*', '').replace('`', '').replace('#', '').replace('"', '').replace("'", "")
        s = s.rstrip('.,!')

        # 3. 精確匹配 (最優先)
        if s in ['yes', '1']: return 1
        if s in ['no', '0']: return 0

        # 4. 模糊匹配 (Regex)
        # \b 代表單字邊界，避免匹配到 "yesterday"
        if re.search(r'\byes\b', s): return 1
        if re.search(r'\bno\b', s): return 0
            
        # 5. 特殊語義匹配
        if 'positive' in s: return 1
        if 'negative' in s: return 0

        return -1 # 解析失敗

    def _pivot_data(self):
        """[Private] 將長表格轉為寬表格"""
        print("🔄 Pivoting table (Long to Wide)...")
        try:
            self.pivot_df = self.raw_df.pivot_table(
                index=self.index_cols + ['True_Numeric'], 
                columns='Feature_Name', 
                values='Pred_Numeric',
                aggfunc='first' 
            )
            
            # 整理表格
            self.pivot_df = self.pivot_df.reset_index()
            self.pivot_df = self.pivot_df.fillna(-1) # 缺值填 -1
            return True
            
        except Exception as e:
            print(f"❌ Pivot failed: {e}")
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
            
            print(f"✅ Data processed successfully!")
            print(f"   - Shape: {self.pivot_df.shape}")
            print(f"   - Parse Success Rate: {valid_count}/{total_count} ({valid_count/total_count:.1%})")
            print(f"   - Saved to: {save_path}")
            
            return save_path
            
        except Exception as e:
            print(f"❌ Error saving file: {e}")
            return None

