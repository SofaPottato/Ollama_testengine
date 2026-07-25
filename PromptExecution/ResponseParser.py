"""解析 LLM structured JSON 回應 → 長表 result.csv（response.csv → result.csv）。

response.csv 一列是「一次推論（一個 batch，可能多 item）」，這裡攤平成「一 item 一列」的 long format，
下游才好 pivot 與評估。responseAns → predLabels 的解碼寫成本檔的獨立函式 decodePredLabels
（非平凡邏輯但僅此檔使用，故不拆檔、也不塞進迴圈體）。
無狀態：所需變數於呼叫時傳入，中間結果以回傳值串接，不存於 self。
"""
import json
import logging
from pathlib import Path
from typing import List

import pandas as pd
from pathvalidate import sanitize_filename

from .util import LabelSet, ParsingError, RESERVED_ITEM_FIELDS, parseJsonCell


#============================================================================#
#   responseAns 解碼：LLM structured JSON 字串 → 每筆預測的 labelCode#
#============================================================================#
def decodePredLabels(text: str, batchSize: int, labelSet: LabelSet) -> List[int]:
    """解析 structured JSON 輸出，回傳長度為 batchSize 的預測碼 list（classes 索引 0..N-1 / -1）。

    "Error:" / 空 / 非合法 JSON 物件 → 全部 -1。
    single：{"label": ...} → [labelCode]；
    batch ：{"answers": [{"id", "label"}]} → 依 id（1-based）或順序回填。
    結構不符 → 對應位置維持 -1（下游評估排除）。
    """
    # 先全填 -1：任何「無法判定」的位置就停在 -1，後面只覆寫「能判定」的，省去缺漏處理。
    predLabels = [-1] * batchSize
    # 空字串或含 "Error:"（來自 OllamaEngine.tryGenerate 的失敗 marker）→ 整批維持 -1。
    if not text or "Error:" in text:
        return predLabels

    # structured 模式下 responseAns 應為合法 JSON 物件；解析失敗或非物件型 → 全標 -1。
    try:
        obj = json.loads(text)
    except Exception:
        obj = None
    if not isinstance(obj, dict):
        logging.warning("[Decoder] 輸出非合法 JSON 物件，全標 -1")
        return predLabels

    answers = obj.get("answers")
    if isinstance(answers, list):
        seenIdxSet = set()
        for answerOrderIdx, ans in enumerate(answers):
            if not isinstance(ans, dict):
                continue
            # id 優先（1-based 編號，能對抗亂序）；不合法/越界 → 退回出現順序 answerOrderIdx；再越界則丟棄。
            try:
                idNum = int(str(ans.get("id")).strip())
            except (TypeError, ValueError):
                idNum = None
            if idNum is not None and 1 <= idNum <= batchSize:
                idx = idNum - 1
            elif answerOrderIdx < batchSize:
                idx = answerOrderIdx
            else:
                continue
            if idx in seenIdxSet:
                logging.warning(f"[Decoder] answers 出現重複位置 idx={idx}（id={ans.get('id')}），後者覆蓋前者")
            seenIdxSet.add(idx)
            predLabels[idx] = labelSet.labelToLabelCode(ans.get("label"))
        return predLabels

    if "label" in obj:
        predLabels[0] = labelSet.labelToLabelCode(obj.get("label"))
    return predLabels


#============================================================================#
#   responseAns → reasoning 文字抽取（供人工檢視，不參與評估）#
#============================================================================#
def extractReason(text: str) -> str:
    """從 structured JSON 輸出抽出 reasoning 文字，回傳「一整批一格」的字串（不參與評估）。

    single（PPI）：{"reasoning": ...} → 該字串。
    batch（BC5CDR）：{"answers":[{"id","reasoning","label"}]} → 各筆 reasoning 併成單一 blob
      （逐筆以 [id] 標註、換行分隔）。刻意「不 demux 到各 item 列」：整批共用一格，維持整批狀態，
      之後不再加工，要看再自己看——與 responseAns 同一種「整批共用、複製到每列」的處理方式。
    空 / "Error:" / 非合法 JSON → ""（原文仍完整保留在 response.csv 的 response 欄，此欄只是方便檢視的抽取值）。
    """
    if not text or "Error:" in text:
        return ""
    try:
        obj = json.loads(text)
    except Exception:
        return ""
    if not isinstance(obj, dict):
        return ""
    # batch 先判：整批 reasoning 併成單一字串（逐筆 [id] 標註），此值會原樣複製到該批展開出的每個 item 列。
    answers = obj.get("answers")
    if isinstance(answers, list):
        return "\n".join(
            f"[{ans.get('id')}] {ans.get('reasoning', '')}"
            for ans in answers if isinstance(ans, dict)
        )
    # single（PPI）：直接取頂層 reasoning。
    return str(obj.get("reasoning") or "")


def stripReason(text: str) -> str:
    """回傳「移除 reasoning 後」的 responseAns，只留答案部分（reasoning 已改存 responseReason）。

    single（PPI）：{"reasoning","label"} → {"label"}。
    batch（BC5CDR）：{"answers":[{"id","reasoning","label"}]} → {"answers":[{"id","label"}]}。
    空 / "Error:" / 非合法 JSON → 原樣回傳（無法解析就不動它：資訊不遺失，下游照舊標 -1）。
    只動解析後的 result.csv／__raw；response.csv 仍保留含 reasoning 的完整原文（存檔用）。
    """
    if not text or "Error:" in text:
        return text
    try:
        obj = json.loads(text)
    except Exception:
        return text
    if not isinstance(obj, dict):
        return text
    # batch：逐筆刪掉 reasoning，其餘欄（id/label…）原序保留。
    answers = obj.get("answers")
    if isinstance(answers, list):
        obj["answers"] = [
            {k: v for k, v in ans.items() if k != "reasoning"}
            for ans in answers if isinstance(ans, dict)
        ]
        return json.dumps(obj, ensure_ascii=False)
    # single：刪頂層 reasoning，其餘欄（label…）保留。
    obj.pop("reasoning", None)
    return json.dumps(obj, ensure_ascii=False)


class ResponseParser:
    """把 response.csv 解析成長表 resultDf（每 item 一列）。

    result.csv 欄位（前面核心欄固定，之後動態疊 item 其他欄與 sentence 欄）：
      sentID     ：樣本唯一識別；優先用 item 自帶 sentID，多 item 時用 taskID_序號，單 item 直接用 taskID。
      model / promptCmbID：來源任務的模型與組合（runKey 在下游 pivot 時才合成，此表不存）。
      originalAns：該樣本 gold label 的原始字串（未轉碼，供人工檢視）。
      trueLabel  ：gold label 轉成 classes 索引 labelCode（與 predLabel 同階段、同一份 labelSet）。
      predLabel  ：模型預測轉成 labelCode（0..N-1）；JSON 壞、label 不在 classes、"Error:" 一律 -1。
      responseAns：模型回應的「答案部分」——已移除 reasoning（同一 batch 的多 item 共用；剝除見 stripReason）。原始含 reasoning 的完整字串保留在 response.csv。
      responseReason：從原始回應抽出的 reasoning 文字（同一 batch 的多 item 共用；抽取見 extractReason）。
      + item 其他欄（如 e1/e2，排除 RESERVED_ITEM_FIELDS）與 sentence 欄（只補空缺、不覆蓋前面的欄）。
    另按 promptCmbID 分檔輸出 <singlePromptCmbOutputDir>/<promptCmbID>_result.csv，供快速檢視單一組合。

    對外有三個步驟，由 Main 照順序接起來：
      1. loadResponse   ：讀 response.csv → responseDf。
      2. parseToResultDf：responseDf → 長表 resultDf（每 item 一列；解碼交給本檔的 decodePredLabels）。
      3. saveResults    ：resultDf → result.csv + 按 promptCmbID 分檔。
    """

    #============================================================================#
    #   Step 1: response.csv → responseDf#
    #============================================================================#
    def loadResponse(self, responsePath: Path) -> pd.DataFrame:
        """讀 response.csv；找不到檔案即 raise。"""
        responsePath = Path(responsePath)
        if not responsePath.exists():
            raise ParsingError(f"找不到暫存結果檔案: {responsePath}")
        return pd.read_csv(str(responsePath), encoding='utf-8-sig')

    #============================================================================#
    #   Step 2: responseDf → 長表 resultDf（每 item 一列）#
    #============================================================================#
    def parseToResultDf(self, rawDf: pd.DataFrame, labelSet: LabelSet) -> pd.DataFrame:
        """response.csv DataFrame → 排序後的長表 resultDf（每列 batch 展開成多列，每 item 一列）。

        解碼 responseAns → predLabels 交給本檔的 decodePredLabels。解析後無任何有效資料即 raise。
        """
        # itertuples 比 iterrows 快；response.csv schema 已於 Step 2 loadCompletedTaskRunIDs 驗證，欄位必齊。
        sentRowList: List[dict] = []
        for taskRow in rawDf.itertuples(index=False):
            # 1) 讀 task 層欄位。taskID 過 str()：CSV 重讀時純數字 ID 會被 pandas 轉成 int，補回字串型別。
            model    = taskRow.model
            promptCmbID = taskRow.promptCmbID
            taskID   = str(taskRow.taskID)

            # 2) 解析 items JSON + 解碼 predLabels。items 空 → 跳過此 task 並 warning（而非 raise），
            #    避免單列異常打斷整批解析。
            itemList = parseJsonCell(taskRow.items) or []
            if not itemList:
                logging.warning(f"[Parser] 跳過任務: items 為空 (model={model}, promptCmbID={promptCmbID})")
                continue
            # 原始回應（含 reasoning）：解碼與抽 reasoning 都以它為準；存進 result 前再把 reasoning 剝掉。
            responseAnsRaw = str(taskRow.response)
            predLabels   = decodePredLabels(responseAnsRaw, len(itemList), labelSet)
            # reasoning 一次抽好（整批一格），與 responseAns 一樣複製到本批展開出的每個 item 列。
            responseReason = extractReason(responseAnsRaw)
            # responseAns 去掉 reasoning，只留答案部分（reasoning 已獨立到 responseReason）。
            responseAns  = stripReason(responseAnsRaw)
            sentenceDict = parseJsonCell(taskRow.sentence) or {}

            # 3) 逐 item 展開成 sentRow（一 item 一列）：欄序固定核心欄 → item 其他欄 → sentence 欄。
            for j, itemDict in enumerate(itemList):
                # sentID 來源優先序：item 自帶 sentID > 多 item 時 taskID_序號 > 單 item 直接用 taskID。
                sentID = itemDict.get('sentID')
                if not sentID:
                    sentID = f"{taskID}_{j}" if len(itemList) > 1 else taskID
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
                    "responseReason": responseReason,
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

        # 4) 全部 task 的 items 都空（整批被跳過）→ 沒東西可評估，raise；
        #    否則依 (model, promptCmbID, sentID) 排序，讓同組合的資料連續，方便下游 groupby/pivot 與人工對照。
        if not sentRowList:
            raise ParsingError("解析後沒有產生任何有效資料。")
        resultDf = pd.DataFrame(sentRowList)
        return resultDf.sort_values(['model', 'promptCmbID', 'sentID'])

    #============================================================================#
    #   Step 3: resultDf → result.csv + 按 promptCmbID 分檔#
    #============================================================================#
    def saveResults(self, resultDf: pd.DataFrame, resultPath: Path, singlePromptCmbOutputDir: Path) -> None:
        """輸出合併版 result.csv，並按 promptCmbID 分檔（供快速檢視單一組合）。

        分檔用 promptCmbID 當檔名；promptCmbID 只含 '+' 與空白這類非法/易誤字元，
        故 sanitize_filename 後再把 '+'、空白換底線即可（此檔為存查產物，
        Step 6 直接吃記憶體中的 resultDf，不從這裡讀回）。
        """
        resultPath = Path(resultPath)
        singlePromptCmbOutputDir = Path(singlePromptCmbOutputDir)
        resultDf.to_csv(str(resultPath), index=False, encoding='utf-8-sig')
        for promptCmbID, groupDf in resultDf.groupby('promptCmbID'):
            fileStem = sanitize_filename(str(promptCmbID), replacement_text='_').replace('+', '_').replace(' ', '_')
            singleCsvPath = singlePromptCmbOutputDir / f"{fileStem}_result.csv"
            groupDf.to_csv(singleCsvPath, index=False, encoding='utf-8-sig')
        logging.info(f"[Parser] 解析完成: {len(resultDf)} samples → {resultPath}")
