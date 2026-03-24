import pandas as pd
import json
import logging
import re
from pathlib import Path
from typing import List

class RegexOutputParser:
    def __init__(self, rawCsvPath: Path, csvOutputPath: Path, singlePromptDir: Path):
        """
        初始化輸出解析器。
        專門負責將 LLM 的 Raw Output (字串) 透過 Regex 拆解成結構化的 DataFrame。
        """
        self.rawCsvPath = Path(rawCsvPath)
        self.csvOutputPath = Path(csvOutputPath)
        self.singlePromptDir = Path(singlePromptDir)
        self.singlePromptDir.mkdir(parents=True, exist_ok=True)
        
        logging.info("RegexOutputParser Initialized.")

    def extractAnswersFromLLMText(self, text: str, batchSize: int) -> List[str]:
        """
        核心解析邏輯：從 LLM 的文字回應中，精準切分出每一題的答案。
        """
        cleanResultsList = ["Parse_Error"] * batchSize
        if not text or "Error:" in text:
            return cleanResultsList

        # 1. 把粗體星號清掉
        text = text.replace('*', '')
        
        # 2. 在字串最前面補上換行符號
        text = "\n" + text.strip()
        
        # 3. 分隔符號必須以 \n (換行) 開頭，徹底避免把答案的 "No" 吃掉
        blocksList = re.split(r'\n\s*(?:Item\s+|No\.?\s*)?\d+\s*[:.)-]', text, flags=re.IGNORECASE)
        
        # 移除 split 產生的第一個空字串
        blocksList = blocksList[1:]

        for i in range(batchSize):
            if i < len(blocksList):
                blockText = blocksList[i].lower()
                # Fallback 掃描
                if 'yes' in blockText or 'cid' in blockText:
                    cleanResultsList[i] = 'Yes'
                elif 'no' in blockText or 'none' in blockText:
                    cleanResultsList[i] = 'No'
        
        return cleanResultsList

    def parse(self) -> str:
        """
        讀取暫存 CSV，套用 Regex 解析，展開成最終的 DataFrame 並存成 CSV
        """
        logging.info("==== [OutputParser] Parsing LLM Outputs & Building CSV ====")
        try:
            if not self.rawCsvPath.exists():
                logging.error(f"❌ 找不到暫存結果檔案: {self.rawCsvPath}")
                return ""

            tempDf = pd.read_csv(str(self.rawCsvPath), encoding='utf-8-sig')
            resultsList = []

            # 遍歷每一筆完成的任務
            for _, taskResultDict in tempDf.iterrows():
                model = taskResultDict.get('model')
                promptID = taskResultDict.get('promptID')
                rawOutput = str(taskResultDict.get('rawOutput', ''))
                
                batchDataStr = taskResultDict.get('batchData', '{}')
                
                if pd.isna(batchDataStr):
                    batchDataStr = '{}'
                    
                batchData = {} 
                
                try:
                    if isinstance(batchDataStr, str):
                        batchData = json.loads(batchDataStr)
                    elif isinstance(batchDataStr, dict):
                        batchData = batchDataStr
                except Exception as e:
                    logging.warning(f"⚠️ 解析 batchData 失敗: {e} (原始資料: {batchDataStr})")
                    batchData = {}
                
                batchPairsList = batchData.get('batchPairsList', [])
                if not batchPairsList:
                    logging.error(f"❌ 警告：這筆任務的 Batch Data 遺失！(Model: {model}, Prompt: {promptID})")

                pmid = batchData.get('pmid', '')

                # 呼叫 Regex 解析
                parsedAnswers = self.extractAnswersFromLLMText(rawOutput, len(batchPairsList))

                # 將 Batch 展開為原本的一對一資料列 (Row)
                for j, pairInfo in enumerate(batchPairsList):
                    ans = parsedAnswers[j] if j < len(parsedAnswers) else "Index_Error"
                    resultsList.append({
                        "Data_ID": pairInfo.get('orig_idx', ''),
                        "PMID": pmid,
                        "Model": model,
                        "promptID": promptID,
                        "E1": pairInfo.get('E1_Name', ''),
                        "E2": pairInfo.get('E2_Name', ''),
                        "True_Label": pairInfo.get('True_Label', ''),
                        "Pred_Label": ans,
                        "Raw_Output": rawOutput # 保留原始輸出方便 Debug
                    })

            finalDf = pd.DataFrame(resultsList)
            
            if finalDf.empty:
                logging.error("❌ 解析後沒有產生任何有效資料，無法儲存 CSV。")
                return ""
                
            finalDf = finalDf.sort_values(['Model', 'promptID', 'Data_ID'])
            
            csvPathStr = str(self.csvOutputPath)
            finalDf.to_csv(csvPathStr, index=False, encoding='utf-8-sig')
            
            for promptID, groupDf in finalDf.groupby('promptID'):
                safeName = str(promptID).replace(":", "_").replace("+", "_").replace(" ", "_").replace("/", "_")
                singlePromptOutputPath = self.singlePromptDir / f"{safeName}_result.csv"
                groupDf.to_csv(singlePromptOutputPath, index=False, encoding='utf-8-sig')
            
            logging.info(f"✅ 額外儲存完成！單獨 Prompt 結果已放入: {self.singlePromptDir}")
            logging.info(f"✅ 解析完成！格式化資料已儲存至: {csvPathStr}")
            
            return csvPathStr
            
        except Exception as e:
            logging.error(f"❌ 解析暫存檔轉最終 CSV 時發生錯誤: {e}", exc_info=True)
            return ""