"""Dataset CSV 載入與 checkpoint 還原：產出 (datasetDf, labelSet) 與已完成的 TaskRunID set。"""
import logging
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import pandas as pd

from .schemas import DataLoadError, LabelSet, RESPONSE_CSV_COLS, TaskRunID, parseJsonCell


# response.csv 用來判斷任務是否已完成的 composite key 欄位（對應 TaskRunID 三元組）；只有本檔讀 checkpoint 時用到。
TASK_RUN_ID_COLUMNS: Tuple[str, str, str] = ("model", "promptCmbID", "taskID")


class DataLoader:
    """載入 Dataset CSV（並推導 labelSet），及從 response.csv 還原已完成的 TaskRunID set 供斷點續跑。

    無狀態：輸入由呼叫端傳入、輸出以 return 交回（不藏進 self），資料流在 Main 一眼可見。

    對外步驟，由 Main 照順序接起來（第 1 步依 taskType 分成 PPI / BC5CDR 兩個方法，Main 擇一呼叫）：
      1a. loadDatasetPPI   ：Dataset CSV → (datasetDf, labelSet)；label 直接在 labelColumn 欄。
      1b. loadDatasetBC5CDR：Dataset CSV → (datasetDf, labelSet)；label 藏在 items JSON 每筆內。
      2.  loadCompletedTaskRunIDs：response.csv → 已完成的 TaskRunID set（斷點續跑用）。

    Dataset CSV 必要欄位：
      - PPI   ：taskID + labelColumn(true label) + sentenceColumns（如 passage）。
      - BC5CDR：taskID + items(JSON array，每筆含 sentID/label/e1/e2) + sentenceColumns（如 title/abstract）。
    """

    #============================================================================#
    #   Step 1（擇一）: Dataset CSV → (datasetDf, labelSet)（PPI / BC5CDR 兩條路）#
    #============================================================================#
    def loadDatasetPPI(self, datasetCsvPath: Union[str, Path], labelColumn: str,
                       sentenceColumns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, LabelSet]:
        """PPI：載入 Dataset CSV、驗證必要欄位，從 labelColumn 推導 labelSet，回傳 (datasetDf, labelSet)。

        找不到檔、缺必要欄位、可用類別 < 2 一律 raise DataLoadError（fail-fast），錯誤在載入期就浮現。
        """
        if not labelColumn:
            raise DataLoadError("PPI 模式必須設定 labelColumn（true label 的來源欄）。")

        # 1) 讀檔。
        csvPath = Path(datasetCsvPath)
        if not csvPath.exists():
            raise DataLoadError(f"找不到 Dataset CSV: {csvPath}")
        taskDf = pd.read_csv(csvPath, encoding='utf-8-sig')

        # 2) 驗證必要欄位：taskID + labelColumn + 所有 sentenceColumns，缺哪些直接列出來。
        requiredColSet = {'taskID', labelColumn} | set(sentenceColumns or [])
        missingColSet = requiredColSet - set(taskDf.columns)
        if missingColSet:
            raise DataLoadError(f"Dataset CSV 缺少必要欄位: {missingColSet}")

        # 3) 從 labelColumn 推導 classes：去空白、以小寫為 key 去重（保留首見寫法）、字母排序定 labelCode 順序。
        seenLowerSet: Set[str] = set()
        classes: List[str] = []
        for rawLabel in taskDf[labelColumn]:
            if rawLabel is None or (isinstance(rawLabel, float) and pd.isna(rawLabel)):
                continue
            cleaned = str(rawLabel).strip()
            if cleaned and cleaned.lower() not in seenLowerSet:
                seenLowerSet.add(cleaned.lower())
                classes.append(cleaned)
        # 少於 2 類無法評估（也過不了 LabelSet 自身檢查），視為資料錯誤直接擋下。
        if len(classes) < 2:
            raise DataLoadError(
                f"資料中可用的 label 不足 2 種（偵測到 {classes}），無法建立 labelSet/評估。請檢查 labelColumn 與前處理輸出。"
            )
        classes.sort(key=str.lower)

        labelSet = LabelSet(classes=classes)
        logging.info(f"[Loader] Dataset 載入完成: {len(taskDf)} 筆、labelSet={labelSet.classes} from {csvPath}")
        return taskDf, labelSet

    def loadDatasetBC5CDR(self, datasetCsvPath: Union[str, Path],
                          sentenceColumns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, LabelSet]:
        """BC5CDR：載入 Dataset CSV、驗證必要欄位，逐列解析 items JSON 收集 label 推導 labelSet，回傳 (datasetDf, labelSet)。

        找不到檔、缺必要欄位、可用類別 < 2 一律 raise DataLoadError（fail-fast），錯誤在載入期就浮現。
        """
        # 1) 讀檔。
        csvPath = Path(datasetCsvPath)
        if not csvPath.exists():
            raise DataLoadError(f"找不到 Dataset CSV: {csvPath}")
        taskDf = pd.read_csv(csvPath, encoding='utf-8-sig')

        # 2) 驗證必要欄位：taskID + items + 所有 sentenceColumns，缺哪些直接列出來。
        requiredColSet = {'taskID', 'items'} | set(sentenceColumns or [])
        missingColSet = requiredColSet - set(taskDf.columns)
        if missingColSet:
            raise DataLoadError(f"Dataset CSV 缺少必要欄位: {missingColSet}")

        # 3) label 不在單一欄：逐列解析 items JSON，收集每筆的 label（壞 JSON 由 parseJsonCell 往上拋）。
        rawLabelList: List = []
        for itemsCell in taskDf['items']:
            parsed = parseJsonCell(itemsCell)
            if not isinstance(parsed, list):
                continue
            for item in parsed:
                if isinstance(item, dict):
                    rawLabelList.append(item.get('label'))

        # 4) 推導 classes：去空白、以小寫為 key 去重（保留首見寫法）、字母排序定 labelCode 順序。
        seenLowerSet: Set[str] = set()
        classes: List[str] = []
        for rawLabel in rawLabelList:
            if rawLabel is None or (isinstance(rawLabel, float) and pd.isna(rawLabel)):
                continue
            cleaned = str(rawLabel).strip()
            if cleaned and cleaned.lower() not in seenLowerSet:
                seenLowerSet.add(cleaned.lower())
                classes.append(cleaned)
        # 少於 2 類無法評估（也過不了 LabelSet 自身檢查），視為資料錯誤直接擋下。
        if len(classes) < 2:
            raise DataLoadError(
                f"資料中可用的 label 不足 2 種（偵測到 {classes}），無法建立 labelSet/評估。請檢查 items 欄與前處理輸出。"
            )
        classes.sort(key=str.lower)

        labelSet = LabelSet(classes=classes)
        logging.info(f"[Loader] Dataset 載入完成: {len(taskDf)} 筆、labelSet={labelSet.classes} from {csvPath}")
        return taskDf, labelSet

    #============================================================================#
    #   Step 2: response.csv → 已完成的 TaskRunID set（斷點續跑）#
    #============================================================================#
    def loadCompletedTaskRunIDs(self, responsePath: Union[str, Path]) -> Set[TaskRunID]:
        """讀 response.csv 取得已完成任務的 TaskRunID set，供斷點續跑。

        檔案不存在 → 回空 set（全新一輪）；schema 不符 / 讀取失敗 → raise DataLoadError。
        """
        # 1) 檔案不存在就是全新一輪，直接回空 set。
        csvPath = Path(responsePath)
        if not csvPath.exists():
            return set()

        # 2) 讀檔 + schema 驗證：壞檔或缺欄位都 raise，提示使用者刪除/備份後重跑。
        try:
            rawDf = pd.read_csv(csvPath, encoding='utf-8-sig')
        except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError, UnicodeDecodeError) as e:
            raise DataLoadError(
                f"response.csv 讀取失敗（壞檔或編碼問題）: {e}。請刪除或備份 {csvPath} 後重跑。"
            ) from e
        missingColSet = set(RESPONSE_CSV_COLS) - set(rawDf.columns)
        if missingColSet:
            raise DataLoadError(
                f"response.csv schema 不符，缺欄位: {sorted(missingColSet)}。請刪除或備份 {csvPath} 後重跑。"
            )

        # 3) 只取三個 key 欄、dropna 後組成 set；strip 對齊寫入端格式，避免空白造成比對失準。
        taskRunIDDf = rawDf[list(TASK_RUN_ID_COLUMNS)].dropna()
        completedTaskRunIDSet: Set[TaskRunID] = {
            TaskRunID(str(row.model).strip(), str(row.promptCmbID).strip(), str(row.taskID).strip())
            for row in taskRunIDDf.itertuples(index=False)
        }
        logging.info(f"[Checkpoint] 已完成任務: {len(completedTaskRunIDSet)} 筆")
        return completedTaskRunIDSet
