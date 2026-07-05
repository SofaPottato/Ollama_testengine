"""把 Dataset CSV 一路加工成 promptInfoDf：先切批、組裝成 userPromptDf，再和 models × prompts 排列組合，並剔除已經跑過的組合。"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd

from .schemas import RESERVED_ITEM_FIELDS, TaskBuildError, TaskRunID, isBlankCell, parseJsonCell


# userPromptDf 的欄位（buildUserPromptPPI / buildUserPromptBC5CDR 產出的東西）：Dataset CSV 每一列攤平、切批後長這樣。
#   taskID：任務編號 / itemList：這一批的樣本 list / userPrompt：組裝好的提示 / sentenceDict：sentenceColumns 的原始值（dict）。
USER_PROMPT_COLS: List[str] = ["taskID", "itemList", "userPrompt", "sentenceDict"]

# promptInfoDf 的欄位：一列就是一個要送去 LLM 的推論任務，推論引擎會照著一列送一次 API。
# items / sentence 放在記憶體時是 list / dict，等到 savePromptInfo 要存成 CSV 時才轉成 JSON 字串。
# items 在兩種模式下用同一種結構：BC5CDR 是一篇文章配多個 pair，所以是多個 {sentID,label,e1,e2}；
#   PPI 只有一句話，就變成只有一個元素的 [{'label': ...}]（句子放在 sentence，每筆只留 label），這樣下游才能走同一條路。
PROMPT_INFO_COLS: List[str] = [
    "taskID", "model", "promptCmbID", "sysPrompt", "userPrompt", "items", "sentence",
]


class TaskBuilder:
    """把 Dataset CSV 攤平成 userPromptDf，再和 models × prompts 排列組合，變成一份待執行的 promptInfoDf。

    字串組裝（把 sentence / items 套進模板）直接寫在兩個 buildUserPrompt* 方法內，各自完整、不共用 helper。

    對外有三個步驟，由 Main 照順序接起來（第 1 步依 taskType 分成 PPI / BC5CDR 兩個方法，Main 擇一呼叫）：
      1a. buildUserPromptPPI   ：Dataset CSV → userPromptDf（把 label 包成單元素 itemList，一篇一 item 不切批）。
      1b. buildUserPromptBC5CDR：Dataset CSV → userPromptDf（解析 items JSON，再依 maxItemsPerBatch 切成一批批）。
      2. buildPromptInfo：把 userPromptDf 乘上 models × prompts，跳過已經跑完的 (model, promptCmbID, taskID)，得到待處理的任務清單 promptInfoDf。
      3. savePromptInfo：把 promptInfoDf 寫成 datasetPromptInfo.csv，方便人工檢視。
    """

    #============================================================================#
    #   Step 1（擇一）: Dataset CSV → userPromptDf（PPI / BC5CDR 兩條路）#
    #============================================================================#
    def buildUserPromptPPI(self, taskDf: pd.DataFrame, sentenceColumns: List[str],
                           taskTemplate: str, labelColumn: str) -> pd.DataFrame:
        """PPI：把 Dataset CSV 的每一列預處理成 userPromptDf 的一列（一篇一 item，不切批）。

        label 從 labelColumn 取出、包成單元素 itemList（與 BC5CDR 的 itemList 同型，下游才能共用同一套流程）。
        labelColumn 未設定就當場 raise TaskBuildError（fail-fast）。回傳 userPromptDf（欄位見 USER_PROMPT_COLS）。
        """
        if not labelColumn:
            raise TaskBuildError("PPI 模式必須設定 labelColumn（true label 的來源欄）。")

        userPromptList: List[Dict[str, Any]] = []
        for _, row in taskDf.iterrows():
            taskID = str(row['taskID'])
            taskSentenceDict: Dict[str, Any] = {col: row[col] for col in sentenceColumns}
            # 把單一 labelColumn 包成只有一個元素的 [{'label': ...}]，刻意做成和 BC5CDR 的 itemList 同一種型別。
            itemList: List[Dict[str, Any]] = [{'label': row[labelColumn]}]
            # 組裝 userPrompt：PPI 只把 sentence 欄填進 taskTemplate；label 是 gold 答案，不進 prompt（留在 itemList 供下游對答案）。
            try:
                userPrompt = taskTemplate.format_map(taskSentenceDict)
            except KeyError as e:
                raise TaskBuildError(f"taskTemplate 佔位符 {e} 在資料中不存在，請確認欄位與模板一致。") from e
            userPromptList.append({
                "taskID":      taskID,
                "itemList":    itemList,
                "userPrompt":  userPrompt,
                "sentenceDict": taskSentenceDict,
            })
        # 明確指定 columns：就算資料集是空的，df 也還是保有正確欄位，buildPromptInfo 用 itertuples 時才不會 KeyError。
        return pd.DataFrame(userPromptList, columns=USER_PROMPT_COLS)

    def buildUserPromptBC5CDR(self, taskDf: pd.DataFrame, sentenceColumns: List[str],
                              taskTemplate: str, maxItemsPerBatch: int,
                              itemTemplate: str, itemColumns: Optional[List[str]] = None) -> pd.DataFrame:
        """BC5CDR：把 Dataset CSV 的每一列預處理成 userPromptDf 的多列（解析 items JSON、依 maxItemsPerBatch 切批）。

        缺 itemTemplate、或 taskTemplate 少了 {items} 佔位符，都會讓 item 被悄悄漏掉，故先當場 raise（fail-fast）。
        每個 batch 的 taskID 統一為 taskID_offset。回傳 userPromptDf（欄位見 USER_PROMPT_COLS）。
        """
        # BC5CDR 設定檢查：少了 itemTemplate 或 {items} 佔位符，item 都會被悄悄漏掉、組裝不進 userPrompt。
        if not itemTemplate:
            raise TaskBuildError("BC5CDR 模式必須提供 itemTemplate，否則 item 無法組裝進 userPrompt。")
        if "{items}" not in taskTemplate:
            raise TaskBuildError("BC5CDR 模式的 taskTemplate 必須包含 {items} 佔位符。")

        recordList: List[Dict[str, Any]] = []
        for _, row in taskDf.iterrows():
            taskID = str(row['taskID'])
            taskSentenceDict: Dict[str, Any] = {col: row[col] for col in sentenceColumns}
            # 把 items 這個 JSON 欄解析成 list（每筆含 sentID/label/e1/e2）；欄位是空的就代表沒有 pair，直接 fail-fast。
            if isBlankCell(row['items']):
                raise TaskBuildError(f"Task {taskID} 的欄位 'items' 為空。")
            allItemList = parseJsonCell(row['items'])

            for offset in range(0, len(allItemList), maxItemsPerBatch):
                # 1) 切批：依 maxItemsPerBatch 切出這一批；taskID 一律用 taskID_offset（單批就是 _0），格式統一。
                batchItemList = allItemList[offset:offset + maxItemsPerBatch]
                batchTaskID = f"{taskID}_{offset}"

                # 2) 拼 {items}：逐 item 套 itemTemplate，串成一整段文字。
                itemsText = ""
                for i, itemDict in enumerate(batchItemList, 1):
                    # 要進模板的欄位：itemColumns 有指定就用指定的，否則用 item 的全部欄位；一律排除內部欄。
                    candidateNameList = itemColumns if itemColumns else list(itemDict.keys())
                    itemFields = {name: itemDict[name] for name in candidateNameList
                                  if name in itemDict and name not in RESERVED_ITEM_FIELDS}
                    try:
                        itemsText += itemTemplate.format_map({'i': i, **itemFields})  # {i} 為 1-based 序號
                    except KeyError as e:
                        raise TaskBuildError(f"itemTemplate 佔位符 {e} 在資料中不存在，請確認欄位與模板一致。") from e

                # 3) 填 taskTemplate：sentence 欄 + 拼好的 {items}。
                try:
                    userPrompt = taskTemplate.format_map({**taskSentenceDict, 'items': itemsText})
                except KeyError as e:
                    raise TaskBuildError(f"taskTemplate 佔位符 {e} 在資料中不存在，請確認欄位與模板一致。") from e

                # 4) 收列。
                recordList.append({
                    "taskID":      batchTaskID,
                    "itemList":    batchItemList,
                    "userPrompt":  userPrompt,
                    "sentenceDict": taskSentenceDict,
                })
        return pd.DataFrame(recordList, columns=USER_PROMPT_COLS)

    #============================================================================#
    #   Step 2: userPromptDf × models × prompts → promptInfoDf#
    #============================================================================#
    def buildPromptInfo(self, userPromptDf: pd.DataFrame, promptCmbDf: pd.DataFrame,
                        selectedModels: List[str], completedTaskRunIDSet: Set[TaskRunID]) -> pd.DataFrame:
        """把 userPromptDf 乘上 selectedModels × promptCmbDf，跳過已經跑完的 TaskRunID，回傳還要執行的 promptInfoDf。

        欄位見 PROMPT_INFO_COLS，PPI 和 BC5CDR 共用同一套。
        """
        # 沒有模型或沒有 prompt 都跑不出東西，與其讓下游拿到空結果，不如在這裡提前 raise 擋掉。
        if not selectedModels:
            raise TaskBuildError("selectedModels 為空，無可執行模型。")
        if promptCmbDf.empty:
            raise TaskBuildError("Prompt 組合清單為空。")

        pendingRecordList: List[Dict[str, Any]] = []
        skippedCount = 0

        # 三層迴圈把 model × prompt × userPrompt 全部展開，一筆一筆對照 checkpoint，跳過已經跑完的。
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
        # 明確指定 columns：就算 pendingRecordList 是空的，df 也還是保有正確欄位，下游做 groupby 或 empty 檢查時才不會 KeyError。
        return pd.DataFrame(pendingRecordList, columns=PROMPT_INFO_COLS)

    #============================================================================#
    #   Step 3: promptInfoDf → datasetPromptInfo.csv（存檔供檢視）#
    #============================================================================#
    def savePromptInfo(self, promptInfoDf: pd.DataFrame, promptInfoPath: Union[str, Path]) -> None:
        """把 promptInfoDf（待執行的任務）存成 CSV 方便檢視，其中 items / sentence 會序列化成 JSON 字串。"""
        outDf = promptInfoDf.copy()
        for col in ("items", "sentence"):
            if col in outDf.columns:
                outDf[col] = outDf[col].map(lambda v: json.dumps(v, ensure_ascii=False))
        csvPath = Path(promptInfoPath)
        outDf.to_csv(str(csvPath), index=False, encoding='utf-8-sig')
        logging.info(f"[Builder] promptInfo 已寫入: {len(outDf)} 筆 → {csvPath}")
