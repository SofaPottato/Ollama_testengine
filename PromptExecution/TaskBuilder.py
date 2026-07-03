"""Dataset CSV → promptInfoDf：切批/渲染成 userPromptDf，再排列 models × prompts、排除已完成組合。"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd

from .DataLoader import TaskRunID
from .PromptFormatter import PromptFormatter
from .schemas import TaskBuildError, isBlankCell, parseJsonCell


# userPromptDf 欄位（buildUserPromptBatches 的產物）：Dataset CSV 每列攤平/切批後的樣子。
#   taskID / itemList（該批樣本 list）/ userPrompt（已渲染）/ sentenceDict（sentenceColumns 原值 dict）。
USER_PROMPT_COLS: List[str] = ["taskID", "itemList", "userPrompt", "sentenceDict"]

# promptInfoDf 欄位（一列 = 一個待送 LLM 的推論任務；推論引擎依此逐列送一次 API）。
# items / sentence 在記憶體中是 list / dict；存 CSV 時才由 savePromptInfo 轉成 JSON 字串。
# items 兩模式共用同一 schema：BC5CDR 為多個 {sentID,label,e1,e2}（一篇多 pair）；
#   PPI 退化成單元素 [{'label': ...}]（句子在 sentence，每筆只剩 label），下游才能共用一條路徑。
PROMPT_INFO_COLS: List[str] = [
    "taskID", "model", "promptCmbID", "sysPrompt", "userPrompt", "items", "sentence",
]


class TaskBuilder:
    """把 Dataset CSV 攤平成 userPromptDf，再與 models × prompts 排列組合成待執行的 promptInfoDf。

    無狀態：輸入由呼叫端傳入、輸出以 return 交回（不藏進 self）。渲染委由獨立的 PromptFormatter。

    三個 public 步驟，由 Main 依序串起來：
      1. buildUserPromptBatches：Dataset CSV → userPromptDf（PPI：label 包成單元素 itemList；
         BC5CDR：解析 items JSON 並依 maxItemsPerBatch 切批）。
      2. assemblePromptInfo：× models × prompts，跳過已完成的 (model, promptCmbID, taskID) → promptInfoDf。
      3. savePromptInfo：把 promptInfoDf 寫成 datasetPromptInfo.csv 供檢視。
    """

    def buildUserPromptBatches(self, taskDf: pd.DataFrame, taskType: str, sentenceColumns: List[str],
                               taskTemplate: str, labelColumn: Optional[str] = None,
                               maxItemsPerBatch: int = 1, itemTemplate: Optional[str] = None,
                               itemColumns: Optional[List[str]] = None) -> pd.DataFrame:
        """把 Dataset CSV 每列預處理成 userPromptDf 一列（parse JSON、依 maxItemsPerBatch 切批、渲染 userPrompt）。

        依 taskType 各走完整路徑（PPI / BC5CDR），組態不合即 fail-fast（raise TaskBuildError）。
        兩模式都攤平成同一種 itemList 結構，下游才能共用同一條路徑。回傳 userPromptDf（欄位見 USER_PROMPT_COLS）。
        """
        formatter = PromptFormatter()

        if taskType == "PPI":
            if not labelColumn:
                raise TaskBuildError("PPI 模式必須設定 labelColumn（true label 的來源欄）。")
            # PPI 一篇一 item，不切批（忽略外部傳入的 maxItemsPerBatch）。

            recordList: List[Dict[str, Any]] = []
            for _, row in taskDf.iterrows():
                baseTaskID = str(row['taskID'])
                taskSentenceDict: Dict[str, Any] = {col: row[col] for col in sentenceColumns}
                # 單一 labelColumn 包成單元素 [{'label': ...}]（刻意退化，與 BC5CDR 的 itemList 同型）。
                itemList: List[Dict[str, Any]] = [{'label': row[labelColumn]}]
                userPrompt = formatter.formatSinglePrompt(taskSentenceDict, itemList, taskTemplate, itemColumns)
                recordList.append({
                    "taskID":      baseTaskID,
                    "itemList":    itemList,
                    "userPrompt":  userPrompt,
                    "sentenceDict": taskSentenceDict,
                })
            # 指定 columns：空資料集時 df 仍保有正確欄位，assemblePromptInfo 的 itertuples 不會 KeyError。
            return pd.DataFrame(recordList, columns=USER_PROMPT_COLS)

        elif taskType == "BC5CDR":
            # BC5CDR 組態把關：缺 itemTemplate 或 {items} 佔位符都會讓 item 默默消失在 userPrompt 外。
            if not itemTemplate:
                raise TaskBuildError("BC5CDR 模式必須提供 itemTemplate，否則 item 無法渲染進 userPrompt。")
            if "{items}" not in taskTemplate:
                raise TaskBuildError("BC5CDR 模式的 taskTemplate 必須包含 {items} 佔位符。")

            recordList = []
            for _, row in taskDf.iterrows():
                baseTaskID = str(row['taskID'])
                taskSentenceDict = {col: row[col] for col in sentenceColumns}
                # 解析 items JSON 欄成 list（每筆含 sentID/label/e1/e2）；空欄代表沒有 pair → fail-fast。
                if isBlankCell(row['items']):
                    raise TaskBuildError(f"Task {baseTaskID} 的欄位 'items' 為空。")
                allItemList = parseJsonCell(row['items'])
                # 依 maxItemsPerBatch 切片，每片產生一個 batch。
                for offset in range(0, len(allItemList), maxItemsPerBatch):
                    batchItemList = allItemList[offset:offset + maxItemsPerBatch]
                    # 有切片（item 數 > 一批容量）才在 taskID 加 _offset 區分；否則沿用原 taskID。
                    batchTaskID = (
                        f"{baseTaskID}_{offset}"
                        if len(allItemList) > maxItemsPerBatch
                        else baseTaskID
                    )
                    userPrompt = formatter.formatBatchPrompt(
                        taskSentenceDict, batchItemList, taskTemplate, itemTemplate, itemColumns)
                    recordList.append({
                        "taskID":      batchTaskID,
                        "itemList":    batchItemList,
                        "userPrompt":  userPrompt,
                        "sentenceDict": taskSentenceDict,
                    })
            return pd.DataFrame(recordList, columns=USER_PROMPT_COLS)

        else:
            # taskType 誤拼在此擋下（DataLoader 亦有把關，此處再防一次，避免直接呼叫時漏檢）。
            raise TaskBuildError(f"taskType 必須為 'PPI' 或 'BC5CDR'，收到 '{taskType}'。")

    def assemblePromptInfo(self, userPromptDf: pd.DataFrame, promptCmbDf: pd.DataFrame,
                           selectedModels: List[str], completedTaskRunIDSet: Set[TaskRunID]) -> pd.DataFrame:
        """userPromptDf × selectedModels × promptCmbDf，跳過已完成的 TaskRunID，回傳待執行的 promptInfoDf。

        欄位見 PROMPT_INFO_COLS，PPI / BC5CDR 共用。
        """
        # 空模型 / 空 prompt 都是無意義的執行，提前 raise 比讓下游產出空結果好。
        if not selectedModels:
            raise TaskBuildError("selectedModels 為空，無可執行模型。")
        if promptCmbDf.empty:
            raise TaskBuildError("Prompt 組合清單為空。")

        pendingRecordList: List[Dict[str, Any]] = []
        skippedCount = 0

        # 三層迴圈展開 model × prompt × userPrompt 列，逐一比對 checkpoint 跳過已完成的。
        for modelName in selectedModels:
            for prompt in promptCmbDf.itertuples(index=False):
                for batch in userPromptDf.itertuples(index=False):
                    taskRunID = TaskRunID(modelName, prompt.promptCmbID, batch.taskID)

                    if taskRunID in completedTaskRunIDSet:
                        skippedCount += 1
                        continue

                    pendingRecordList.append({
                        "taskID":      batch.taskID,
                        "model":       modelName,
                        "promptCmbID": prompt.promptCmbID,
                        "sysPrompt":   prompt.promptText,
                        "userPrompt":  batch.userPrompt,
                        "items":       batch.itemList,
                        "sentence":     batch.sentenceDict,
                    })

        if skippedCount > 0:
            logging.info(f"[Builder] 跳過已完成任務: {skippedCount} 筆")
        logging.info(f"[Builder] 待執行任務: {len(pendingRecordList)} 筆")
        # 指定 columns：即使 pendingRecordList 為空，df 仍保有正確欄位，下游 groupby/empty 檢查不會 KeyError。
        return pd.DataFrame(pendingRecordList, columns=PROMPT_INFO_COLS)

    def savePromptInfo(self, promptInfoDf: pd.DataFrame, promptInfoPath: Union[str, Path]) -> None:
        """把 promptInfoDf（待執行任務）存成 CSV 供檢視（items/sentence 序列化成 JSON 字串）。"""
        # 不動原 df（引擎稍後要用記憶體中的 list/dict）：複製一份，把巢狀欄轉成 JSON 字串再寫檔，
        # 否則 to_csv 會寫出 Python repr（單引號）而非合法 JSON，重讀時無法 json.loads。
        outDf = promptInfoDf.copy()
        for col in ("items", "sentence"):
            if col in outDf.columns:
                outDf[col] = outDf[col].map(lambda v: json.dumps(v, ensure_ascii=False))
        csvPath = Path(promptInfoPath)
        outDf.to_csv(str(csvPath), index=False, encoding='utf-8-sig')
        logging.info(f"[Builder] promptInfo 已寫入: {len(outDf)} 筆 → {csvPath}")
