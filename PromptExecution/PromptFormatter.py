from string import Formatter
from typing import Dict, List, Optional
from .schemas import RESERVED_ITEM_FIELDS


def _safeFormat(template: str, fields: Dict) -> str:
    # 佔位符命名使用field
    # 如果模板佔位符對應的資料值是 None，代表來源資料缺值
    # fail-fast：raise 並指出哪些佔位符對應的值是 None。
    referencedFieldSet = {fieldName for _, fieldName, _, _ in Formatter().parse(template) if fieldName}
    noneFieldList = sorted(name for name in referencedFieldSet if name in fields and fields[name] is None)
    if noneFieldList:
        raise ValueError(f"[Formatter] 模板欄位值為 None（來源資料缺值）: {noneFieldList}")
    try:
        return template.format_map({fieldName: str(fieldValue) for fieldName, fieldValue in fields.items()})
    except KeyError as e:
        # 模板寫了但 fields 沒有 → 這是「模板佔位符與資料欄位對不上」的設定錯誤，而非資料問題，附上提示訊息幫助 debug。
        raise KeyError(f"[Formatter] 模板佔位符 {e} 在資料中不存在") from e


class PromptFormatter:
    """
    將 context 與 items 填入 taskTemplate / itemTemplate，產出 userPrompt。
    itemTemplate 存在
    → 批次模式（多 item 展開後塞入 {items}）。
    → 單筆模式（context + items[0] 合併填入）。
    """
    def __init__(self, taskTemplate: str, itemTemplate: Optional[str] = None,
                 itemColumns: Optional[List[str]] = None):
        self.taskTemplate = taskTemplate
        self.itemTemplate = itemTemplate
        self.itemColumns = itemColumns

    def format(self, contextDict: Dict, items: List[Dict]) -> str:
        """itemTemplate 存在 → 批次模式；否則 → 單筆模式。"""
        if self.itemTemplate:
            return self._formatBatch(contextDict, items)
        return self._formatSingle(contextDict, items)

    def _formatBatch(self, contextDict: Dict, items: List[Dict]) -> str:
        """批次模式：每個 item 渲染後拼接，整段填入 {items} 佔位符。"""
        # 先把每個 item 各自渲染成一段文字再串起來，最後當成單一 {items} 值塞進 taskTemplate。
        itemsText = ""
        for i, itemDict in enumerate(items, 1):
            itemsText += _safeFormat(self.itemTemplate, {'i': i, **self._extractItemFields(itemDict)})
        return _safeFormat(self.taskTemplate, {**contextDict, 'items': itemsText})

    def _formatSingle(self, contextDict: Dict, items: List[Dict]) -> str:
        """單筆模式：context 與 items[0] 合併後直接填入 taskTemplate。item 欄優先，同名時覆蓋 context。"""
        # PPI 只有一個 item，沒有獨立的 itemTemplate，直接把 context 與 item 欄位攤平餵進 taskTemplate。
        allFieldDict = dict(contextDict)
        if items:
            allFieldDict.update(self._extractItemFields(items[0]))
        return _safeFormat(self.taskTemplate, allFieldDict)

    def _extractItemFields(self, itemDict: Dict) -> Dict:
        """從 item dict 抽取要送進模板的欄位，一律排除 RESERVED_ITEM_FIELDS 中的內部欄位。"""
        candidateNameList = self.itemColumns if self.itemColumns else list(itemDict.keys())
        return {
            name: itemDict[name]
            for name in candidateNameList
            if name in itemDict and name not in RESERVED_ITEM_FIELDS
        }
