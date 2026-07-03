"""解析 LLM structured JSON 回應 → 長表 result.csv（response.csv → result.csv）。

response.csv 一列是「一次推論（一個 batch，可能多 item）」，這裡攤平成「一 item 一列」的 long format，
下游才好 pivot 與評估。responseAns → predLabels 的解碼委由獨立的 ResponseDecoder。
無狀態：所需變數於呼叫時傳入，中間結果以回傳值串接，不存於 self。
"""
import logging
from pathlib import Path
from typing import List

import pandas as pd

from .ResponseDecoder import ResponseDecoder
from .schemas import LabelSet, ParsingError, RESERVED_ITEM_FIELDS, parseJsonCell, safeFileStem


class ResponseParser:
    """把 response.csv 解析成長表 resultDf（每 item 一列）。

    result.csv 欄位（前面核心欄固定，之後動態疊 item 其他欄與 sentence 欄）：
      sentID     ：樣本唯一識別；優先用 item 自帶 sentID，多 item 時用 taskID_序號，單 item 直接用 taskID。
      model / promptCmbID：來源任務的模型與組合（runKey 在下游 pivot 時才合成，此表不存）。
      originalAns：該樣本 gold label 的原始字串（未轉碼，供人工檢視）。
      trueLabel  ：gold label 轉成 classes 索引 labelCode（與 predLabel 同階段、同一份 labelSet）。
      predLabel  ：模型預測轉成 labelCode（0..N-1）；JSON 壞、label 不在 classes、"Error:" 一律 -1。
      responseAns：模型原始回應字串（同一 batch 的多 item 共用）。
      + item 其他欄（如 e1/e2，排除 RESERVED_ITEM_FIELDS）與 sentence 欄（只補空缺、不覆蓋前面的欄）。
    另按 promptCmbID 分檔輸出 <singleOutputDir>/<promptCmbID>_result.csv，供快速檢視單一組合。
    """

    def loadResponse(self, responsePath: Path) -> pd.DataFrame:
        """讀 response.csv；找不到檔案即 raise。"""
        responsePath = Path(responsePath)
        if not responsePath.exists():
            raise ParsingError(f"找不到暫存結果檔案: {responsePath}")
        return pd.read_csv(str(responsePath), encoding='utf-8-sig')

    def parseToResultDf(self, rawDf: pd.DataFrame, labelSet: LabelSet) -> pd.DataFrame:
        """response.csv DataFrame → 排序後的長表 resultDf（每列 batch 展開成多列，每 item 一列）。

        解碼 responseAns → predLabels 交給 ResponseDecoder。解析後無任何有效資料即 raise。
        """
        decoder = ResponseDecoder()
        # itertuples 比 iterrows 快；response.csv schema 已於 Step 2 loadCompletedTaskRunIDs 驗證，欄位必齊。
        sentRowList: List[dict] = []
        for taskRow in rawDf.itertuples(index=False):
            model    = taskRow.model
            promptCmbID = taskRow.promptCmbID
            taskID   = str(taskRow.taskID)

            # items 是寫檔時序列化的 JSON，解析回 list。空 list → 跳過此 task 並 warning（而非 raise），
            # 避免單列異常打斷整批解析。
            itemList = parseJsonCell(taskRow.items) or []
            if not itemList:
                logging.warning(f"[Parser] 跳過任務: items 為空 (model={model}, promptCmbID={promptCmbID})")
                continue

            responseAns  = str(taskRow.responseAns)
            predLabels   = decoder.decodePredLabels(responseAns, len(itemList), labelSet)
            sentenceDict = parseJsonCell(taskRow.sentence) or {}

            # 一 item 一列（sentRow）：欄序固定核心欄 → item 其他欄 → sentence 欄。
            for j, itemDict in enumerate(itemList):
                # sentID 來源優先序：item 自帶 sentID > 多 item 時 taskID_序號 > 單 item 直接用 taskID。
                sentID = itemDict.get('sentID') or (f"{taskID}_{j}" if len(itemList) > 1 else taskID)
                # trueLabel 在此就轉 labelCode（與 predLabel 同階段、同一份 labelSet），originalAns 保留原始字串；
                # 下游 LLMResultProcessor 不再重複轉碼，避免 labelSet 不一致。
                rawLabel = itemDict.get('label', '')
                sentRow = {
                    "sentID":      sentID,
                    "model":       model,
                    "promptCmbID": promptCmbID,
                    "originalAns": rawLabel,
                    "trueLabel":   labelSet.labelToLabelCode(rawLabel),
                    "predLabel":   predLabels[j],
                    "responseAns": responseAns,
                }
                # 疊上 item 的其他欄位（如 e1/e2），濾掉 RESERVED_ITEM_FIELDS。
                for otherColName, otherColVal in itemDict.items():
                    if otherColName not in RESERVED_ITEM_FIELDS:
                        sentRow[otherColName] = otherColVal
                # 最後疊 sentence 欄，且不覆蓋已存在的欄——核心欄與 item 欄優先，sentence 只補空缺。
                for otherColName, otherColVal in sentenceDict.items():
                    if otherColName not in sentRow:
                        sentRow[otherColName] = otherColVal
                sentRowList.append(sentRow)

        # 全部 task 的 items 都空（整批被跳過）→ 沒東西可評估，raise。
        if not sentRowList:
            raise ParsingError("解析後沒有產生任何有效資料。")
        resultDf = pd.DataFrame(sentRowList)
        # 依 (model, promptCmbID, sentID) 排序：讓同組合的資料連續，方便下游 groupby/pivot 與人工對照。
        return resultDf.sort_values(['model', 'promptCmbID', 'sentID'])

    def saveResults(self, resultDf: pd.DataFrame, resultPath: Path, singleOutputDir: Path) -> None:
        """輸出合併版 result.csv，並按 promptCmbID 分檔（供快速檢視單一組合）。

        分檔用 safeFileStem 把 promptCmbID 洗成跨平台安全的檔名。此檔為存查產物；
        Step 6 直接吃記憶體中的 resultDf，不從這裡讀回。
        """
        resultPath = Path(resultPath)
        singleOutputDir = Path(singleOutputDir)
        resultDf.to_csv(str(resultPath), index=False, encoding='utf-8-sig')
        for promptCmbID, groupDf in resultDf.groupby('promptCmbID'):
            singleCsvPath = singleOutputDir / f"{safeFileStem(promptCmbID)}_result.csv"
            groupDf.to_csv(singleCsvPath, index=False, encoding='utf-8-sig')
        logging.info(f"[Parser] 解析完成: {len(resultDf)} samples → {resultPath}")
