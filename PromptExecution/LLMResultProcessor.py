"""長表 → 寬表：把 ResponseParser 的長表 pivot 成「每樣本一列、每 runKey 一組欄」的寬表。

result.csv（long）→ 兩張寬表：
  - mlTable          ：精簡寬表，只含 index 欄 + 各 runKey 的 __pred，給 Evaluate 吃。
  - fullResultInfo   ：在 __pred 之上再補 __raw（模型原文）與 __sysPrompt（system prompt），給人工 review。

runKey = makeRunKey(model, promptCmbID)（寬表的欄維度識別；格式定義見 schemas）。
所有衍生欄一律帶後綴（__pred / __raw / __sysPrompt）方便下游用後綴過濾。
無狀態：長表由呼叫端傳入，中間結果以回傳值串接，不存於 self。
"""
import logging
from typing import Dict, Optional

import pandas as pd

from .schemas import makeRunKey


class LLMResultProcessor:
    """把長表 resultDf pivot 成 mlTable / fullResultInfo 兩張寬表。"""

    # 衍生 runKey 欄位的後綴（下游可用後綴一次篩出同類欄）。
    _PRED_SUFFIX = '__pred'
    _RAW_SUFFIX = '__raw'
    _SYS_PROMPT_SUFFIX = '__sysPrompt'
    # pivot 時不能當 index 的欄：model/promptCmbID 是欄維度，predLabel/responseAns 是值。
    _NON_INDEX_COLS = {'model', 'promptCmbID', 'predLabel', 'responseAns'}

    def _pivotWide(self, resultDf: pd.DataFrame, valueCol: str,
                   suffix: str, fillValue) -> pd.DataFrame:
        """長表 → 寬表的共用 pivot：index=樣本欄、columns=(model, promptCmbID)、values=valueCol。

        index 欄動態偵測（排除 _NON_INDEX_COLS），上游新增資料欄不必改這裡。
        欄名攤平成 runKey + suffix；缺值補 fillValue。回傳仍以樣本欄為 MultiIndex（供對齊）。
        """
        indexCols = [c for c in resultDf.columns if c not in self._NON_INDEX_COLS]
        wideDf = resultDf.pivot_table(
            index=indexCols, columns=['model', 'promptCmbID'], values=valueCol, aggfunc='first'
        ).fillna(fillValue)
        wideDf.columns = [makeRunKey(m, p) + suffix for m, p in wideDf.columns]
        return wideDf

    def writeMLTable(self, resultDf: pd.DataFrame) -> pd.DataFrame:
        """long → wide（精簡）：每樣本一列、每 runKey 一欄 {runKey}__pred（值為 predLabel）。

        某樣本在某 runKey 沒資料 → 補 -1（與「無法解析」共用 sentinel）。
        另記錄 parse rate：須用長表算，寬表的 -1 分不出「無資料」與「解析失敗」會數不準。
        """
        predWideDf = self._pivotWide(resultDf, 'predLabel', self._PRED_SUFFIX, -1)

        validCount = int((resultDf['predLabel'] != -1).sum())
        totalCount = len(resultDf)
        rate = f"{validCount / totalCount:.1%}" if totalCount else "n/a"
        logging.info(f"[Processor] parse rate={validCount}/{totalCount} ({rate})")

        return predWideDf.reset_index()

    def writeFullResultInfo(self, resultDf: pd.DataFrame, mlTableDf: pd.DataFrame,
                            promptCmbDf: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """在 mlTableDf（__pred 寬表）之上補 __raw 與 __sysPrompt，供人工 review。

        __pred 直接沿用傳入的 mlTableDf，不重算 pivot；本方法只多做 __raw 一次 pivot。
          - {runKey}__raw      ：responseAns 原文（缺值補空字串）。
          - {runKey}__sysPrompt：該 runKey 當次用的 system prompt（promptCmbDf 為空則略過）。
        以樣本 index 欄 merge 對齊（1:1，不依賴列順序）。
        """
        indexCols = [c for c in resultDf.columns if c not in self._NON_INDEX_COLS]
        rawWideDf = self._pivotWide(resultDf, 'responseAns', self._RAW_SUFFIX, '').reset_index()
        # mlTableDf 已含 index 欄 + __pred；rawWideDf 含 index 欄 + __raw；以 index 欄 1:1 merge。
        fullResultInfoDf = mlTableDf.merge(rawWideDf, on=indexCols, how='left')

        # 每個 runKey 補一欄 __sysPrompt（該組合的 system prompt 原文）。
        # runKey↔promptCmbID 的對應只在長表裡（寬表已沒這層維度），故由長表查。
        if promptCmbDf is None or promptCmbDf.empty:
            return fullResultInfoDf
        promptCmbIDToText = dict(zip(promptCmbDf['promptCmbID'], promptCmbDf['promptText']))
        newColsDict: Dict[str, pd.Series] = {}
        for row in resultDf[['model', 'promptCmbID']].drop_duplicates().itertuples(index=False):
            colName = makeRunKey(row.model, row.promptCmbID) + self._SYS_PROMPT_SUFFIX
            newColsDict[colName] = pd.Series(
                promptCmbIDToText.get(row.promptCmbID, ''), index=fullResultInfoDf.index)
        if newColsDict:
            fullResultInfoDf = pd.concat([fullResultInfoDf, pd.DataFrame(newColsDict)], axis=1)
        return fullResultInfoDf
