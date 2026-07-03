"""LLM structured JSON 回應 → 每筆預測 labelCode 的解碼器。

ResponseParser 於 parseToResultDf 內就地建立 ResponseDecoder，對每個 task row 的 responseAns 解出 predLabels。
"""
import json
import logging
from typing import List

from .schemas import LabelSet


class ResponseDecoder:
    """把單一推論的 responseAns（structured JSON 字串）解碼成長度為 batchSize 的預測碼 list。無狀態。"""

    def decodePredLabels(self, text: str, batchSize: int, labelSet: LabelSet) -> List[int]:
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
