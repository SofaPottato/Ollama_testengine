"""userPrompt 渲染：把 sentenceDict 與 items 套進 taskTemplate / itemTemplate。

自 TaskBuilder 拆出的獨立渲染單元：
  - 批次模式（BC5CDR）：每個 item 各自用 itemTemplate 渲染後拼接成一整段 {items}，再塞進 taskTemplate。
  - 單筆模式（PPI）  ：sentence 與 items[0] 的欄位合併後，直接填入 taskTemplate。
TaskBuilder 於 buildUserPromptBatches 內就地建立 PromptFormatter，逐列渲染 userPrompt。
"""
from typing import Dict, List, Optional

from .schemas import RESERVED_ITEM_FIELDS


class PromptFormatter:
    """把 sentenceDict + items 渲染成一段 userPrompt 字串。無狀態，可重複使用同一實例。

    formatBatchPrompt / formatSinglePrompt 各自完整寫到底：欄位過濾（排除 RESERVED_ITEM_FIELDS）與
    format_map 渲染都 inline 在方法內，兩者的 try/except 段刻意重複，不抽共用 helper。
    """

    def formatBatchPrompt(self, sentenceDict: Dict, items: List[Dict],
                          taskTemplate: str, itemTemplate: str, itemColumns: Optional[List[str]]) -> str:
        """批次模式：每個 item 各自渲染後拼接，整段當成單一 {items} 值塞進 taskTemplate。"""
        itemsText = ""
        for i, itemDict in enumerate(items, 1):
            # 抽取要送進模板的欄位，排除 RESERVED_ITEM_FIELDS（sentID/label 等內部欄位）。
            candidateNameList = itemColumns if itemColumns else list(itemDict.keys())
            itemFields = {
                name: itemDict[name]
                for name in candidateNameList
                if name in itemDict and name not in RESERVED_ITEM_FIELDS
            }
            # format_map 渲染單一 item；模板有佔位符而資料缺 → KeyError 附提示，方便定位不一致。
            itemFieldStrDict = {fieldName: str(fieldValue) for fieldName, fieldValue in {'i': i, **itemFields}.items()}
            try:
                itemsText += itemTemplate.format_map(itemFieldStrDict)
            except KeyError as e:
                raise KeyError(f"[Formatter] 模板佔位符 {e} 在資料中不存在") from e

        # 渲染整段 taskTemplate（sentence 欄 + 拼好的 {items}）。
        taskFieldStrDict = {fieldName: str(fieldValue) for fieldName, fieldValue in {**sentenceDict, 'items': itemsText}.items()}
        try:
            return taskTemplate.format_map(taskFieldStrDict)
        except KeyError as e:
            raise KeyError(f"[Formatter] 模板佔位符 {e} 在資料中不存在") from e

    def formatSinglePrompt(self, sentenceDict: Dict, items: List[Dict],
                           taskTemplate: str, itemColumns: Optional[List[str]]) -> str:
        """單筆模式：sentence 與 items[0] 合併後直接填入 taskTemplate。item 欄優先，同名時覆蓋 sentence。"""
        allFieldDict = dict(sentenceDict)
        if items:
            # 抽取要送進模板的欄位，排除 RESERVED_ITEM_FIELDS（sentID/label 等內部欄位）。
            itemDict = items[0]
            candidateNameList = itemColumns if itemColumns else list(itemDict.keys())
            itemFields = {
                name: itemDict[name]
                for name in candidateNameList
                if name in itemDict and name not in RESERVED_ITEM_FIELDS
            }
            allFieldDict.update(itemFields)

        # format_map 渲染；模板有佔位符而資料缺 → KeyError 附提示，方便定位不一致。
        allFieldStrDict = {fieldName: str(fieldValue) for fieldName, fieldValue in allFieldDict.items()}
        try:
            return taskTemplate.format_map(allFieldStrDict)
        except KeyError as e:
            raise KeyError(f"[Formatter] 模板佔位符 {e} 在資料中不存在") from e
