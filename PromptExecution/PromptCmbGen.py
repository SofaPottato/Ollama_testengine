"""Prompt 組合生成：讀方法池 YAML → 依窮舉/手動模式枚舉組合 → 產出 promptCmbDf（並寫 CSV 存查）。"""
import logging
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import yaml

from .schemas import DataLoadError


class PromptCmbGen:
    """Pipeline 前置階段：讀方法池 YAML，枚舉出所有 prompt 組合。

    無狀態：方法池路徑、生成模式參數、輸出路徑皆由呼叫端逐一傳入。

    對外有三個步驟，由 Main 照順序接起來：
      1. loadMethodPool：讀方法池 YAML → promptTechPoolDict。
      2. genPromptCmb  ：依窮舉(Auto)/手動(Manual)模式枚舉組合 → promptCmbDf。
      3. savePromptCmb ：把 promptCmbDf 寫成 CSV 存查。

    產出 promptCmbDf 兩欄（train/test 共用同一份）：
      - promptCmbID：組合內各方法 ID 以 ' + ' 串接（如 'EMO01 + Role01'），下游 prompt 維度的識別碼。
      - promptText ：組合內各方法的 systemPrompt 原文以換行串接，即該組合實際送模型的 system prompt。
    """

    #============================================================================#
    #   Step 1: 方法池 YAML → promptTechPoolDict#
    #============================================================================#
    def loadMethodPool(self, promptTechPath: Union[str, Path]) -> Dict[str, Dict]:
        """載入方法池 YAML，回傳 {method: {itemKey: text}}；值非 dict 的 method 略過、空池即 raise。"""
        promptTechPath = Path(promptTechPath)
        if not promptTechPath.exists():
            raise DataLoadError(f"找不到 Prompt 方法池檔案: {promptTechPath}")
        try:
            with open(promptTechPath, 'r', encoding='utf-8') as f:
                dataDict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise DataLoadError(f"Prompt 方法池 YAML 格式錯誤: {promptTechPath}: {e}") from e

        # 容許兩種寫法：頂層直接是 method dict，或包在 'prompts' 鍵底下。
        promptTechPoolDict = (dataDict or {}).get('prompts', dataDict) or {}
        # 只保留值為 dict 的 method；空項或型別寫錯的略過，避免污染後續組合。
        skippedList = [k for k, v in promptTechPoolDict.items() if not isinstance(v, dict)]
        if skippedList:
            logging.warning(f"[PromptGen] 略過格式不符的 method: {skippedList}")
        promptTechPoolDict = {k: v for k, v in promptTechPoolDict.items() if isinstance(v, dict)}
        if not promptTechPoolDict:
            raise DataLoadError(f"方法池 '{promptTechPath}' 沒有任何有效 method，請確認 YAML 格式。")
        return promptTechPoolDict

    #============================================================================#
    #   Step 2: promptTechPoolDict → promptCmbDf（Auto 窮舉 / Manual 手動）#
    #============================================================================#
    def genPromptCmb(self, promptTechPoolDict: Dict[str, Dict],
                       b_exhaustiveCmb: bool,
                       selectedPromptTechList: List[str],
                       maxCmbNum: Optional[int],
                       manualPromptCmbList: List[List[str]]) -> pd.DataFrame:
        """依窮舉(Auto)/手動(Manual)模式枚舉組合並排序，回傳 promptCmbDf（欄位 promptCmbID, promptText）。

        依 b_exhaustiveCmb 各走完整路徑：
          - Auto ：對選定方法分類做 1..maxCmbNum 層組合，再對各分類的項目做笛卡兒積。
          - Manual：只生成 manualPromptCmbList 明確列出的組合；找不到的 Prompt ID 略過。
        每組 record 皆為 {promptCmbID: id 以 ' + ' 串接, promptText: text 以換行串接}。
        最後統一排序（兩模式共用），結果為空即 raise。
        """
        recordList: List[Dict[str, str]] = []

        if b_exhaustiveCmb:
            logging.info("[PromptGen] 生成模式: Auto (Exhaustive)")
            # 1) 決定參與組合的方法分類：'ALLMethod' 代表全部；否則過濾掉方法池裡不存在的。
            if selectedPromptTechList == ['ALLMethod']:
                targetMethodList = list(promptTechPoolDict.keys())
            else:
                invalidList = [m for m in selectedPromptTechList if m not in promptTechPoolDict]
                if invalidList:
                    logging.warning(f"[PromptGen] 方法池中找不到，將略過: {invalidList}")
                targetMethodList = [m for m in selectedPromptTechList if m in promptTechPoolDict]

            if not targetMethodList:
                raise DataLoadError("沒有任何有效的 selectedPromptTechList，請檢查參數與方法池 YAML。")

            # 2) 一組最多含幾個分類：maxCmbNum 未設就以分類數為上限；再夾到實際分類數，避免 combinations 取超量。
            maxSize = maxCmbNum or len(targetMethodList)
            limitNum = min(maxSize, len(targetMethodList))

            # 3) 枚舉：先選 r 個分類（combinations），每個分類各自展開成 (id, text) 清單，
            #    再對這些清單做笛卡兒積（product），每一種取法就是一組完整組合。
            for r in range(1, limitNum + 1):
                for methodComboTuple in combinations(targetMethodList, r):
                    perMethodItemsList = []
                    for cat in methodComboTuple:
                        catItemList = [(f"{cat}{str(k).zfill(2)}", text) for k, text in promptTechPoolDict[cat].items()]
                        perMethodItemsList.append(catItemList)
                    for itemComboTuple in product(*perMethodItemsList):
                        idList = [itemID for itemID, _ in itemComboTuple]
                        textList = [text for _, text in itemComboTuple]
                        recordList.append({"promptCmbID": " + ".join(idList), "promptText": "\n".join(textList)})

        else:
            logging.info("[PromptGen] 生成模式: Manual")
            if not manualPromptCmbList:
                logging.warning("[PromptGen] manualPromptCmbList 為空，沒有組合可生成。")

            # 1) 先把方法池攤平成 {PromptID: text}，之後照 manualPromptCmbList 給的 ID 直接查。
            flatPoolDict: Dict[str, str] = {}
            for cat, itemsDict in promptTechPoolDict.items():
                for k, text in itemsDict.items():
                    flatPoolDict[f"{cat}{str(k).zfill(2)}"] = text.strip()

            # 2) 逐組合查 ID：找得到的收進來、找不到的略過並 warning；一組全找不到就整組不收。
            for comboKeyList in manualPromptCmbList:
                idList, textList = [], []
                for itemKey in comboKeyList:
                    if itemKey in flatPoolDict:
                        idList.append(itemKey)
                        textList.append(flatPoolDict[itemKey])
                    else:
                        logging.warning(f"[PromptGen] 找不到 Prompt ID '{itemKey}'，已略過。")
                if idList:
                    recordList.append({"promptCmbID": " + ".join(idList), "promptText": "\n".join(textList)})

        # 排序（兩模式共用）：先比組合內方法數、再比方法定義順序與項目編號，輸出穩定可讀。
        methodOrderList = list(promptTechPoolDict.keys())
        # 先比對較長的 method 名，避免前綴相同（如 'RE2' 與 'RE'）時誤判。
        sortedMethodList = sorted(methodOrderList, key=len, reverse=True)

        def parsePartId(part: str):
            for method in sortedMethodList:
                if part.startswith(method):
                    return (methodOrderList.index(method), int(part[len(method):]))
            return (999, 999)

        def sortKey(record: Dict[str, str]):
            partList = record['promptCmbID'].split(' + ')
            return (len(partList), tuple(parsePartId(p) for p in partList))

        # promptCmbID 一律由 f"{cat}{zfill(2)}" 產生，parsePartId 又有 (999,999) fallback，排序不會 raise。
        recordList = sorted(recordList, key=sortKey)

        if not recordList:
            raise DataLoadError("Prompt 生成結果為空，請檢查生成模式參數設定。")
        return pd.DataFrame(recordList, columns=['promptCmbID', 'promptText'])

    #============================================================================#
    #   Step 3: promptCmbDf → CSV 存查#
    #============================================================================#
    def savePromptCmb(self, promptCmbDf: pd.DataFrame, outputPath: Union[str, Path]) -> Path:
        """把生成的組合寫入 outputPath（一律覆蓋舊檔），回傳該路徑。

        每次執行都以最新生成的組合為準；Main 只吃 genPromptCmb 的回傳 df，不再讀回此檔（故可安全覆蓋）。
        """
        outputPath = Path(outputPath)
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        promptCmbDf.to_csv(outputPath, index=False, encoding='utf-8-sig')
        logging.info(f"[PromptGen] 已生成 {len(promptCmbDf)} 組 Prompt → {outputPath}")
        return outputPath
