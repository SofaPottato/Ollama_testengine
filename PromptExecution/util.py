"""跨模組共用契約、型別與工具（多模組共享者的家）。

只放「被多個模組共享、或需保證多端一致」的東西：
  - 共享資料模型／型別：LabelSet、TaskRunID。
  - 跨階段例外體系：PipelineError 及其子類。
  - 多處引用的常數與工具：RESERVED_ITEM_FIELDS、parseJsonCell。
單一模組自用的東西一律留在該模組本地，不集中於此——集中它們只是多一層 indirection：
  runKey 格式（makeRunKey / RUN_KEY_SEPARATOR）只有 LLMResultProcessor 產生，定義在該檔。
"""
import json
from dataclasses import dataclass
from pydantic import BaseModel, Field, model_validator
from typing import Any, List, FrozenSet

import pandas as pd


# 任何處理 item 的模組都應引用此常數，避免 'sentID'/'label' 硬編碼散落多處
# sentID = sentenceID（樣本識別碼；PPI 為句子，BC5CDR 為 entity-pair）
RESERVED_ITEM_FIELDS: FrozenSet[str] = frozenset({'sentID', 'label'})


#============================================================================#
#   跨模組共用工具（JSON 欄解析）#
#============================================================================#
def parseJsonCell(value: Any) -> Any:
    """解析 CSV 讀進來的 JSON 欄位（items / sentence）：
    str → json.loads（失敗往上拋）；已是 list/dict 等 → 原樣回傳；空欄(None/NaN) → None。
    寫檔端一律用 json.dumps，故此處採嚴格解析，壞資料就 fail-fast。"""
    if isinstance(value, str):
        return json.loads(value)
    # 空欄（None 或 pandas NaN）→ None
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return value


#============================================================================#
#   TaskRunID：checkpoint 三元組（response.csv 欄位契約由 OllamaEngine 寫入端持有）#
#============================================================================#
@dataclass(frozen=True)
class TaskRunID:
    """單次 LLM 推論的唯一識別三元組；response.csv checkpoint 以此比對是否已完成。

    三欄依序為：model（Ollama 模型名，如 "llama3.2:1b"）、promptCmbID（Prompt 組合識別碼）、taskID（批次識別碼）。
    """
    model:    str
    promptCmbID: str
    taskID:   str


#============================================================================#
#   標籤集合（LabelSet）#
#============================================================================#
class LabelSet(BaseModel):
    """
    標籤集合設定（純標籤語意，與推論引擎無關）。classes 在清單中的索引即為整數標籤 labelCode（0..N-1），未命中一律 -1。
    gold label 與 classes 的對齊由前處理負責，須完全一致（比對時大小寫不敏感、去空白）。
    （classes 也被 OllamaEngine.buildOllamaOutputFormat 拿去產生 Ollama `format` schema，但那屬引擎細節，不放這裡。）
    """
    classes: List[str] = Field(
        default_factory=lambda: ["no", "yes"],
        description="分類類別清單；索引即整數 labelCode。二元 [no,yes] 或多分類 [negative,neutral,positive]"
    )

    @model_validator(mode='after')
    def checkLabelSet(self):
        """去空白、檢查非空 / 不重複（大小寫不敏感）/ 至少 2 類。"""
        # 先去空白，再做三項健全性檢查：空字串、大小寫不敏感重複、至少 2 類——
        # 這些都是會讓 labelCode 對應出錯卻不易察覺的設定問題，故在載入期就擋下。
        cleaned = [str(c).strip() for c in self.classes]
        if any(not c for c in cleaned):
            raise ValueError("labelSet 不可包含空字串。")
        lowered = [c.lower() for c in cleaned]
        if len(set(lowered)) != len(lowered):
            raise ValueError(f"labelSet 不可重複（大小寫不敏感）: {self.classes}")
        if len(cleaned) < 2:
            raise ValueError(f"labelSet 至少需 2 個類別，目前: {self.classes}")
        self.classes = cleaned
        return self

    def labelToLabelCode(self, label: Any) -> int:
        """字串標籤 → 索引 labelCode；大小寫不敏感、去空白；未命中回 -1。"""
        # None 直接 -1，避免對 None 做字串操作。其餘去空白+小寫後線性掃 classes（類別數通常 2-3，
        # 成本可忽略）；未命中（含拼錯、多餘空白外的差異）回 -1——「無法判定」的統一表示，下游據此排除。
        if label is None:
            return -1
        target = str(label).strip().lower()
        return next((i for i, c in enumerate(self.classes) if c.lower() == target), -1)


#============================================================================#
#   Pipeline 例外體系#
#============================================================================#
# 階層化例外：各階段內部明確 raise 對應子類，讓 traceback 一眼看出失敗在哪一步。
# Main 目前不攔（直接讓它 traceback，實驗情境足夠）；需要時上層可捕 PipelineError 一網打盡。
class PipelineError(Exception):
    """所有 Pipeline 錯誤的基底類別。"""
    pass

class DataLoadError(PipelineError):
    """資料或 Prompt 載入失敗：檔案不存在、欄位缺漏、格式錯誤等。"""
    pass

class TaskBuildError(PipelineError):
    """任務建構失敗：模型/prompt 清單為空、JSON 欄位無法解析等。"""
    pass

class ParsingError(PipelineError):
    """解析輸出失敗：找不到 response.csv、解析後無有效資料等。"""
    pass


# 待執行任務以 promptInfoDf 一列表示（欄位定義見 TaskBuilder.PROMPT_INFO_COLS）；
# 推論引擎直接迭代該 df，不經過額外的 pydantic 物件。
