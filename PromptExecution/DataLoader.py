"""Dataset CSV 載入與 checkpoint 還原：產出 (datasetDf, labelSet) 與已完成的 TaskRunID set。"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import pandas as pd

from .OllamaEngine import RESPONSE_CSV_COLS, TASK_RUN_ID_COLUMNS
from .schemas import DataLoadError, LabelSet, ModelName, PromptCmbID, TaskID, parseJsonCell


@dataclass(frozen=True)
class TaskRunID:
    """單次 LLM 推論的唯一識別三元組；response.csv checkpoint 以此比對是否已完成。"""
    model:    ModelName
    promptCmbID: PromptCmbID
    taskID:   TaskID


class DataLoader:
    """載入 Dataset CSV（並推導 labelSet），及從 response.csv 還原已完成的 TaskRunID set 供斷點續跑。

    無狀態：輸入由呼叫端傳入、輸出以 return 交回（不藏進 self），資料流在 Main 一眼可見。

    Dataset CSV 必要欄位：
      - PPI   ：taskID + labelColumn(true label) + sentenceColumns（如 passage）。
      - BC5CDR：taskID + items(JSON array，每筆含 sentID/label/e1/e2) + sentenceColumns（如 title/abstract）。
    """

    def loadDataset(self, datasetCsvPath: Union[str, Path], taskType: str,
                    labelColumn: Optional[str] = None,
                    sentenceColumns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, LabelSet]:
        """載入 Dataset CSV、驗證必要欄位，並從資料推導 labelSet，回傳 (datasetDf, labelSet)。

        依 taskType 各走完整路徑（PPI / BC5CDR）：兩者結構相同，僅 true label 的來源不同——
        PPI 的 label 在 labelColumn 欄；BC5CDR 的 label 藏在 items JSON 每筆內。
        找不到檔、缺必要欄位、可用類別 < 2 一律 raise DataLoadError（fail-fast），錯誤在載入期就浮現。
        """
        sentenceColumnSet = set(sentenceColumns or [])
        csvPath = Path(datasetCsvPath)
        if not csvPath.exists():
            raise DataLoadError(f"找不到 Dataset CSV: {csvPath}")
        taskDf = pd.read_csv(csvPath, encoding='utf-8-sig')

        if taskType == 'PPI':
            # PPI 必要欄位：taskID + labelColumn + 所有 sentenceColumns。
            requiredColSet = {'taskID', labelColumn} | sentenceColumnSet
            self._checkRequiredCols(taskDf, requiredColSet)
            # PPI：label 直接在 labelColumn 欄。
            rawLabelList = taskDf[labelColumn].tolist()
            classes = self._deriveClasses(rawLabelList, hint="請檢查 labelColumn 與前處理輸出。")

        elif taskType == 'BC5CDR':
            # BC5CDR 必要欄位：taskID + items + 所有 sentenceColumns。
            requiredColSet = {'taskID', 'items'} | sentenceColumnSet
            self._checkRequiredCols(taskDf, requiredColSet)
            # BC5CDR：label 不在單一欄，逐列解析 items JSON 取出每筆的 label（壞 JSON 由 parseJsonCell 往上拋）。
            rawLabelList = []
            for itemsCell in taskDf['items']:
                parsed = parseJsonCell(itemsCell)
                if not isinstance(parsed, list):
                    continue
                for item in parsed:
                    if isinstance(item, dict):
                        rawLabelList.append(item.get('label'))
            classes = self._deriveClasses(rawLabelList, hint="請檢查 items 欄與前處理輸出。")

        else:
            # taskType 誤拼（如 'ppi'）在此最早擋下，避免靜默走錯分支報難解的錯。
            raise DataLoadError(f"taskType 必須為 'PPI' 或 'BC5CDR'，收到 '{taskType}'。")

        labelSet = LabelSet(classes=classes)
        logging.info(f"[Loader] Dataset 載入完成: {len(taskDf)} 筆、labelSet={labelSet.classes} from {csvPath}")
        return taskDf, labelSet

    @staticmethod
    def _checkRequiredCols(taskDf: pd.DataFrame, requiredColSet: set) -> None:
        """缺任一必要欄位就 raise，並用差集列出缺哪些，方便對照前處理輸出。"""
        missingColSet = requiredColSet - set(taskDf.columns)
        if missingColSet:
            raise DataLoadError(f"Dataset CSV 缺少必要欄位: {missingColSet}")

    @staticmethod
    def _deriveClasses(rawLabelList: List, hint: str) -> List[str]:
        """從一串原始 label 推導 classes：去空白、以小寫為 key 去重（保留首見寫法）、字母排序定 labelCode 順序。

        少於 2 類無法評估（也過不了 LabelSet 自身檢查），視為資料錯誤直接擋下。hint 併入錯誤訊息指路。
        """
        seenLowerSet: Set[str] = set()
        classes: List[str] = []
        for rawLabel in rawLabelList:
            if rawLabel is None or (isinstance(rawLabel, float) and pd.isna(rawLabel)):
                continue
            cleaned = str(rawLabel).strip()
            if cleaned and cleaned.lower() not in seenLowerSet:
                seenLowerSet.add(cleaned.lower())
                classes.append(cleaned)
        if len(classes) < 2:
            raise DataLoadError(
                f"資料中可用的 label 不足 2 種（偵測到 {classes}），無法建立 labelSet/評估。{hint}"
            )
        classes.sort(key=str.lower)
        return classes

    def loadCompletedTaskRunIDs(self, responsePath: Union[str, Path]) -> Set[TaskRunID]:
        """讀 response.csv 取得已完成任務的 TaskRunID set，供斷點續跑。

        檔案不存在 → 回空 set（全新一輪）；schema 不符 / 讀取失敗 → raise DataLoadError。
        """
        completedTaskRunIDSet: Set[TaskRunID] = set()
        csvPath = Path(responsePath)

        if not csvPath.exists():
            return completedTaskRunIDSet

        try:
            rawDf = pd.read_csv(csvPath, encoding='utf-8-sig')
        except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError, UnicodeDecodeError) as e:
            raise DataLoadError(
                f"response.csv 讀取失敗（壞檔或編碼問題）: {e}。請刪除或備份 {csvPath} 後重跑。"
            ) from e

        # 欄位對不上代表 response.csv 格式錯誤 → raise。
        missingColSet = set(RESPONSE_CSV_COLS) - set(rawDf.columns)
        if missingColSet:
            raise DataLoadError(
                f"response.csv schema 不符，缺欄位: {sorted(missingColSet)}。請刪除或備份 {csvPath} 後重跑。"
            )

        # 只取三個 key 欄、dropna 後組成 set；strip 對齊寫入端格式，避免空白造成比對失準。
        taskRunIDDf = rawDf[list(TASK_RUN_ID_COLUMNS)].dropna()
        for row in taskRunIDDf.itertuples(index=False):
            completedTaskRunIDSet.add(
                TaskRunID(str(row.model).strip(), str(row.promptCmbID).strip(), str(row.taskID).strip())
            )
        logging.info(f"[Checkpoint] 已完成任務: {len(completedTaskRunIDSet)} 筆")

        return completedTaskRunIDSet
