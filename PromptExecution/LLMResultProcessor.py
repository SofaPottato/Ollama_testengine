"""長表 → 寬表：把 ResponseParser 的長表 pivot 成「每樣本一列、每 runKey 一組欄」的寬表。

result.csv（long）→ 兩張寬表：
  - mlTable          ：精簡寬表，只含 index 欄 + 各 runKey 的 __pred，給 Evaluate 吃。
  - fullResultInfo   ：在 __pred 之上再補 __raw（模型原文）、__reason（抽出的 reasoning）與 __sysPrompt（system prompt），給人工 review。

runKey = makeRunKey(model, promptCmbID)（寬表的欄維度識別）。runKey 格式只在本檔產生（寬表欄名），
故連同分隔符定義於此，非跨模組契約。
所有衍生欄一律帶後綴（__pred / __raw / __reason / __sysPrompt）方便下游用後綴過濾。
無狀態：長表由呼叫端傳入，中間結果以回傳值串接，不存於 self。
"""
import logging
from typing import Dict, Optional

import pandas as pd

# runKey：把一次「模型 × Prompt 組合」壓成單一識別字串（寬表的欄維度識別）。
RUN_KEY_SEPARATOR = '|'   # 用 '|' 而非 '_'，因模型名常含 '_'/':'

# 本檔獨用的衍生欄後綴。
RAW_SUFFIX = '__raw'                # 模型回應的答案部分（已移除 reasoning；完整原文在 response.csv）
REASON_SUFFIX = '__reason'          # 從原始回應抽出的 reasoning 文字（與 __raw 成對）
SYS_PROMPT_SUFFIX = '__sysPrompt'   # 該組合的 system prompt


def makeRunKey(model: str, promptCmbID: str) -> str:
    """model + promptCmbID → runKey。形如 "llama3.2:1b|EMO01 + Role01"。

    長表(result.csv)不存此欄，只在下列輸出出現：
      - mlTable / fullResultInfo 的欄名（再加 __pred / __raw / __sysPrompt 後綴）。
      - evalSummary 的 modelPromptCmbID 欄值（Evaluate 以 removesuffix('__pred') 還原）。
      - 混淆矩陣 PNG 檔名（各檔自行 sanitize 成跨平台安全字串）。
    """
    return f"{model}{RUN_KEY_SEPARATOR}{promptCmbID}"


class LLMResultProcessor:
    """把長表 resultDf pivot 成 mlTable / fullResultInfo 兩張寬表。

    對外有兩個步驟，由 Main 照順序接起來：
      1. writeMLTable       ：resultDf → mlTableDf（每樣本一列、每 runKey 一欄 __pred）。
      2. writeFullResultInfo：在 mlTableDf 之上補 __raw / __sysPrompt → fullResultInfoDf。
    """

    # pivot 時不能當 index 的欄：model/promptCmbID 是欄維度，predLabel/responseAns/responseReason 是值。
    _NON_INDEX_COLS = {'model', 'promptCmbID', 'predLabel', 'responseAns', 'responseReason'}

    #============================================================================#
    #   Step 1: resultDf → mlTableDf（精簡寬表，給 Evaluate 吃）#
    #============================================================================#
    def writeMLTable(self, resultDf: pd.DataFrame) -> pd.DataFrame:
        """long → wide（精簡）：每樣本一列、每 runKey 一欄 {runKey}__pred（值為 predLabel）。

        某樣本在某 runKey 沒資料 → 補 -1（與「無法解析」共用 sentinel）。
        另記錄 parse rate：須用長表算，寬表的 -1 分不出「無資料」與「解析失敗」會數不準。
        """
        # 1) pivot long → wide：index=樣本欄（動態偵測，排除 _NON_INDEX_COLS，上游新增資料欄不必改這裡）、
        #    columns=(model, promptCmbID)、values=predLabel；欄名攤平成 runKey__pred，缺值補 -1。
        indexCols = [c for c in resultDf.columns if c not in self._NON_INDEX_COLS]
        predWideDf = resultDf.pivot_table(
            index=indexCols, columns=['model', 'promptCmbID'], values='predLabel', aggfunc='first'
        ).fillna(-1)
        predWideDf.columns = [makeRunKey(m, p) + '__pred' for m, p in predWideDf.columns]

        # 2) 記錄 parse rate（用長表算才準）。
        validCount = int((resultDf['predLabel'] != -1).sum())
        totalCount = len(resultDf)
        rate = f"{validCount / totalCount:.1%}" if totalCount else "n/a"
        logging.info(f"[Processor] parse rate={validCount}/{totalCount} ({rate})")

        return predWideDf.reset_index()

    #============================================================================#
    #   Step 2: mlTableDf + resultDf → fullResultInfoDf（含原文的完整寬表）#
    #============================================================================#
    def writeFullResultInfo(self, resultDf: pd.DataFrame, mlTableDf: pd.DataFrame,
                            promptCmbDf: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """在 mlTableDf（__pred 寬表）之上補 __raw、__reason 與 __sysPrompt，供人工 review。

        __pred 直接沿用傳入的 mlTableDf，不重算 pivot；本方法多做 __raw 與 __reason 兩次 pivot。
          - {runKey}__raw      ：responseAns（答案部分，已移除 reasoning；缺值補空字串）。
          - {runKey}__reason   ：responseReason（從原文抽出的 reasoning；與 __raw 成對，缺值補空字串）。
          - {runKey}__sysPrompt：該 runKey 當次用的 system prompt（promptCmbDf 為空則略過）。
        以樣本 index 欄 merge 對齊（1:1，不依賴列順序）。
        """
        # 1) __raw 寬表：responseAns 做同款 pivot（index=樣本欄、columns=(model, promptCmbID)、缺值補空字串），
        #    欄名攤平成 runKey__raw，再以 index 欄 1:1 merge 進 mlTableDf。
        indexCols = [c for c in resultDf.columns if c not in self._NON_INDEX_COLS]
        rawWideDf = resultDf.pivot_table(
            index=indexCols, columns=['model', 'promptCmbID'], values='responseAns', aggfunc='first'
        ).fillna('')
        rawWideDf.columns = [makeRunKey(m, p) + RAW_SUFFIX for m, p in rawWideDf.columns]
        fullResultInfoDf = mlTableDf.merge(rawWideDf.reset_index(), on=indexCols, how='left')

        # 1b) __reason 寬表：responseReason 做同款 pivot（與 __raw 成對），再以 index 欄 1:1 merge 進來。
        #     凡有 responseAns（__raw）之處就補上對應的 responseReason（__reason）。
        reasonWideDf = resultDf.pivot_table(
            index=indexCols, columns=['model', 'promptCmbID'], values='responseReason', aggfunc='first'
        ).fillna('')
        reasonWideDf.columns = [makeRunKey(m, p) + REASON_SUFFIX for m, p in reasonWideDf.columns]
        fullResultInfoDf = fullResultInfoDf.merge(reasonWideDf.reset_index(), on=indexCols, how='left')

        # 2) 每個 runKey 補一欄 __sysPrompt（該組合的 system prompt 原文）。
        #    runKey↔promptCmbID 的對應只在長表裡（寬表已沒這層維度），故由長表查。
        if promptCmbDf is None or promptCmbDf.empty:
            return fullResultInfoDf
        promptCmbIDToText = dict(zip(promptCmbDf['promptCmbID'], promptCmbDf['promptText']))
        newColsDict: Dict[str, pd.Series] = {}
        for row in resultDf[['model', 'promptCmbID']].drop_duplicates().itertuples(index=False):
            colName = makeRunKey(row.model, row.promptCmbID) + SYS_PROMPT_SUFFIX
            newColsDict[colName] = pd.Series(
                promptCmbIDToText.get(row.promptCmbID, ''), index=fullResultInfoDf.index)
        # resultDf 非空（parseToResultDf 空即 raise）→ 至少一組 (model, promptCmbID) → newColsDict 必非空。
        fullResultInfoDf = pd.concat([fullResultInfoDf, pd.DataFrame(newColsDict)], axis=1)
        return fullResultInfoDf
