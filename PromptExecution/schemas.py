"""Pydantic schema、自定義例外、資料模型的單一事實來源。"""
import json
from dataclasses import dataclass
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing import Dict, Any, List, FrozenSet, TypeAlias

import pandas as pd
from pathvalidate import sanitize_filename


# 任何處理 item 的模組都應引用此常數，避免 'sentID'/'label' 硬編碼散落多處
# sentID = sentenceID（樣本識別碼；PPI 為句子，BC5CDR 為 entity-pair）
RESERVED_ITEM_FIELDS: FrozenSet[str] = frozenset({'sentID', 'label'})


#============================================================================#
#   跨模組共用工具（去重：JSON 欄解析、檔名清洗）#
#============================================================================#
def isBlankCell(value: Any) -> bool:
    """CSV 空欄判斷：None 或 pandas NaN。"""
    return value is None or (isinstance(value, float) and pd.isna(value))


def parseJsonCell(value: Any) -> Any:
    """解析 CSV 讀進來的 JSON 欄位（items / sentence）：
    str → json.loads（失敗往上拋）；已是 list/dict 等 → 原樣回傳；空欄(None/NaN) → None。
    寫檔端一律用 json.dumps，故此處採嚴格解析，壞資料就 fail-fast。"""
    if isinstance(value, str):
        return json.loads(value)
    if isBlankCell(value):
        return None
    return value


def safeFileStem(name: Any) -> str:
    """把 promptCmbID / runKey 等轉成跨平台安全的檔名片段（'+' 與空白一律換底線）。"""
    return sanitize_filename(str(name), replacement_text='_').replace('+', '_').replace(' ', '_')


#============================================================================#
#   語意化型別別名：純為提升可讀性與 IDE 提示，型別檢查器仍視為 str#
#============================================================================#
ModelName:   TypeAlias = str   # Ollama 模型名稱，如 "llama3.2:1b"
PromptCmbID: TypeAlias = str   # Prompt 組合識別碼
TaskID:      TypeAlias = str   # Task 批次層級識別碼
ResponseAns: TypeAlias = str   # LLM 原始文字回應（未解析）


#============================================================================#
#   runKey：把一次「模型 × Prompt 組合」壓成單一識別字串（格式的唯一來源）#
#============================================================================#
RUN_KEY_SEPARATOR = '|'   # 用 '|' 而非 '_'，因模型名常含 '_'/':'
RunKey: TypeAlias = str   # 形如 "llama3.2:1b|EMO01 + Role01"


def makeRunKey(model: ModelName, promptCmbID: PromptCmbID) -> RunKey:
    """model + promptCmbID → runKey：把一次「模型 × Prompt 組合」壓成單一識別字串（格式唯一來源）。

    形如 "llama3.2:1b|EMO01 + Role01"。長表(result.csv)不存此欄，只在下列輸出出現：
      - mlTable / fullResultInfo 的欄名（再加 __pred / __raw / __sysPrompt 後綴）。
      - evalSummary 的 modelPromptCmbID 欄值。
      - 混淆矩陣 PNG 檔名（經 safeFileStem 清洗成跨平台安全字串）。
    """
    return f"{model}{RUN_KEY_SEPARATOR}{promptCmbID}"


# 預測欄後綴：欄名 = runKey + '__pred'。LLMResultProcessor 寫欄名、Evaluate 依此後綴篩欄，
# 跨模組契約故放 schemas（__raw / __sysPrompt 只有 LLMResultProcessor 用，定義在該檔）。
PRED_SUFFIX = '__pred'


#============================================================================#
#   TaskRunID 與 response.csv schema（checkpoint 的單一事實來源）#
#============================================================================#
@dataclass(frozen=True)
class TaskRunID:
    """單次 LLM 推論的唯一識別三元組；response.csv checkpoint 以此比對是否已完成。"""
    model:    ModelName
    promptCmbID: PromptCmbID
    taskID:   TaskID


# response.csv 欄位順序的單一事實來源（推論引擎逐筆 append 的輸出檔，同時是斷點續跑的 checkpoint）。
# 寫入端（OllamaEngine.appendCsv）與讀取端（DataLoader.loadCompletedTaskRunIDs / ResponseParser）都引用同一常數；
# 改 schema 只需改這裡一處——但既有 response.csv 就得刪除，否則欄位對不上會 raise。
#   model / promptCmbID / taskID：任務三元組（= checkpoint composite key，對應 TaskRunID）。
#   systemPrompt / userPrompt   ：本次送模型的兩段 prompt 原文。
#   responseAns                 ：模型原始文字回應（未解析）；失敗時為 "Error:..." marker，下游標 -1。
#   items / sentence            ：該任務的樣本清單與 sentence（JSON 字串），供 ResponseParser 攤平回每個 item。
RESPONSE_CSV_COLS: List[str] = [
    "model", "promptCmbID", "taskID",
    "systemPrompt", "userPrompt", "responseAns", "items", "sentence",
]


#============================================================================#
#   標籤集合（LabelSet）#
#============================================================================#
class LabelSet(BaseModel):
    """
    標籤集合設定。classes 在清單中的索引即為整數標籤 labelCode（0..N-1），未命中一律 -1。
    Ollama `format` JSON schema 由此產生，強制模型只能輸出 classes 之一。
    gold label 與 classes 的對齊由前處理負責，須完全一致（比對時大小寫不敏感、去空白）。
    """
    classes: List[str] = Field(
        default_factory=lambda: ["no", "yes"],
        description="分類類別清單；索引即整數 labelCode。二元 [no,yes] 或多分類 [negative,neutral,positive]"
    )

    # 預建的 label→labelCode 對照表（小寫為 key）。
    _labelCodeByLabel: Dict[str, int] = PrivateAttr(default_factory=dict)

    @model_validator(mode='after')
    def checkLabelSet(self):
        """去空白、檢查非空 / 不重複（大小寫不敏感）/ 至少 2 類，並預建 label→labelCode 對照表。"""
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
        # 一次建好對照表（小寫 key），labelToLabelCode 就能 O(1) 查，不必每次線性掃 classes。
        self._labelCodeByLabel = {c.lower(): i for i, c in enumerate(cleaned)}
        return self

    def labelToLabelCode(self, label: Any) -> int:
        """字串標籤 → 索引 labelCode；大小寫不敏感、去空白；未命中回 -1。"""
        # None 直接 -1，避免對 None 做字串操作。其餘一律去空白+小寫後查表，未命中（含拼錯、
        # 多餘空白外的差異）回 -1——「無法判定」的統一表示，下游據此排除。
        if label is None:
            return -1
        return self._labelCodeByLabel.get(str(label).strip().lower(), -1)

    def buildOllamaOutputFormat(self, taskType: str) -> Dict[str, Any]:
        """
        產生 Ollama `format` 用的 JSON schema。
        taskType="PPI"：單筆預測 {"label": <enum>}；其餘（BC5CDR）：{"answers": [{"id": int, "label": <enum>}]}。
        """
        # 用 enum 把 label 限定成 classes 之一：Ollama 端就會強制模型只輸出這些字串，
        # 大幅減少 ResponseParser 要處理的雜訊（少數不遵守的仍由 labelToLabelCode 兜底回 -1）。
        labelProp = {"type": "string", "enum": self.classes}
        if taskType == "PPI":
            return {
                "type": "object",
                "properties": {"label": labelProp},
                "required": ["label"],
            }
        # batch：要求每筆帶 id（1-based 序號），讓 ResponseParser 能把答案對回正確的 pair。
        return {
            "type": "object",
            "properties": {
                "answers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}, "label": labelProp},
                        "required": ["id", "label"],
                    },
                }
            },
            "required": ["answers"],
        }


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
