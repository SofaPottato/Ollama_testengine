import pandas as pd
import os
import time
import re
import logging 

class LLMResultProcessor:
    def __init__(self, inputCsvPath: str, outputCsvPath: str, mergedPath: str, originalDf: pd.DataFrame):
        """
        初始化資料處理器
        :param inputCsvPath: 原始 LLM 輸出結果的檔案路徑 (Raw Output)
        :param outputCsvPath: 處理後乾淨檔案的儲存【檔案路徑】(包含檔名，解決原本 outputDir 的命名混淆)
        :param mergedPath: 包含完整資訊的合併檔案儲存【檔案路徑】
        :param originalDf: 原始的 DataFrame
        """
        self.inputCsvPath = inputCsvPath
        self.outputCsvPath = outputCsvPath 
        self.mergedPath = mergedPath
        self.originalDf = originalDf 
        
        # 保留原有的領域邏輯與欄位設定
        self.requiredCols = ['Model', 'promptID', 'Pred_Label', 'True_Label']
        self.index_cols = ['Data_ID', 'PMID', 'E1', 'E2']
        self.rawDf = None
        self.pivotDf = None
        
        logging.info(f"LLMResultProcessor(inputCsvPath='{self.inputCsvPath}', outputCsvPath='{self.outputCsvPath}')")

    def cleanAndMergeOriginalData(self):
        """
        [Public] 執行完整的處理流程：讀取 -> 解析 -> 轉置 -> 存檔
        :return: 處理後的檔案路徑 (str) 或 None
        """
        logging.info(f"process(self=<{self.__module__}.{self.__class__.__name__} object at {hex(id(self))}>)")
        
        logging.info(f"Processing data: {self.inputCsvPath}")
        
        if not self.loadData():
            return None
 
        self.rawDf['Pred_Label'] = self.rawDf['Pred_Label'].apply(self.parseResponse)
        
        logging.info("Processing True Labels")
        self.rawDf['True_Label'] = self.rawDf['True_Label'].apply(
            lambda x: 1 if str(x).strip().lower() == 'cid' else 0
        )

        logging.info("Creating Feature Names")
        self.rawDf['Feature_Name'] = self.rawDf['Model'].astype(str) + "_" + self.rawDf['promptID'].astype(str)
        
        if not self.pivotData():
            return None
            
        result_path = self.saveData()
        return result_path

    def loadData(self):
        """[Private] 讀取並驗證 CSV"""
        logging.info("Loading Raw CSV Data...")
        if not os.path.exists(self.inputCsvPath):
            logging.error(f"❌ Error: File not found: {self.inputCsvPath}")
            return False
            
        try:
            self.rawDf = pd.read_csv(self.inputCsvPath)
            logging.info(f"Data loaded successfully. Shape: {self.rawDf.shape}")
            
            missing = [c for c in self.requiredCols + self.index_cols if c not in self.rawDf.columns]
            if missing:
                logging.error(f"❌ Error: Missing columns: {missing}")
                return False
            return True
            
        except Exception as e:
            logging.error(f"❌ Error reading CSV: {e}")
            return False

    def parseResponse(self, text):
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

    def pivotData(self):
        """[Private] 將長表格轉為寬表格"""
        logging.info("Pivoting table (Long to Wide)")
        try:
            self.pivotDf = self.rawDf.pivot_table(
                index=self.index_cols + ['True_Label'], 
                columns='Feature_Name', 
                values='Pred_Label',
                aggfunc='first' 
            )
            
            # 整理表格
            self.pivotDf = self.pivotDf.reset_index()
            self.pivotDf = self.pivotDf.fillna(-1)
            logging.info(f"Pivot completed. New Shape: {self.pivotDf.shape}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Pivot failed: {e}")
            return False

    def saveData(self):
        """[Private] 儲存結果 (分流：一份簡單乾淨，一份豐富資訊)"""
        try:   
            savePath = str(self.outputCsvPath)
            
            saveDir = os.path.dirname(savePath)
            if saveDir:
                os.makedirs(saveDir, exist_ok=True)
                
            self.pivotDf.to_csv(savePath, index=False, encoding='utf-8-sig')
            
            if self.mergedPath and self.originalDf is not None:
                logging.info("Generating rich merged table for human review...")
                
                mergedDfDir = os.path.dirname(str(self.mergedPath))
                if mergedDfDir: 
                    os.makedirs(mergedDfDir, exist_ok=True)
                    
                columnsToAdd = ['Title', 'Abstract', 'Full_Text', 'E1_Type', 'E1_MeSH', 'E2_Type', 'E2_MeSH']
                validCols = [c for c in columnsToAdd if c in self.originalDf.columns]
                
                origSubset = self.originalDf[validCols].copy()
                origSubset['Data_ID'] = origSubset.index
                
                mergeDf = pd.merge(self.pivotDf, origSubset, on='Data_ID', how='left')
                frontCols = ['Data_ID', 'PMID',
                            'E1', 'E1_Type', 'E2', 'E2_Type',
                            'True_Label', 'Title', 'Abstract']
                frontCols = [c for c in frontCols if c in mergeDf.columns]
                predCols = [c for c in mergeDf.columns if c not in frontCols]
                mergeDf = mergeDf[frontCols + predCols]
                
                mergeDf.to_csv(self.mergedPath, index=False, encoding='utf-8-sig')
                logging.info(f"   -資訊總成已儲存至: {self.mergedPath}")
            # ==========================================
            
            validCount = (self.rawDf['Pred_Label'] != -1).sum()
            totalCount = len(self.rawDf)
            
            logging.info("✅ Data processed successfully!")
            logging.info(f"   - Clean Shape: {self.pivotDf.shape}")
            logging.info(f"   - Parse Success Rate: {validCount}/{totalCount} ({validCount/totalCount:.1%})")
            logging.info(f"   - Clean Pipeline Data Saved to: {savePath}")
            
            return savePath
            
        except Exception as e:
            logging.error(f"❌ Error saving file: {e}")
            return None