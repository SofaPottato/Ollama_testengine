"""Pydantic schema、自定義例外、資料模型的單一事實來源。"""
from pathlib import Path
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator
from typing import Dict, Any, List, Optional, ClassVar, FrozenSet, TypeAlias


# 任何處理 item 的模組都應引用此常數，避免 'sentID'/'label' 硬編碼散落多處
# sentID = sentenceID（樣本識別碼；PPI 為句子，BC5CDR 為 entity-pair）
RESERVED_ITEM_FIELDS: FrozenSet[str] = frozenset({'sentID', 'label'})


# ── 語意化型別別名：純為提升可讀性與 IDE 提示，型別檢查器仍視為 str ──────────
ModelName: TypeAlias = str   # Ollama 模型名稱，如 "llama3.2:1b"
PromptID:  TypeAlias = str   # Prompt 策略識別碼
TaskID:    TypeAlias = str   # Task 批次層級識別碼
RawOutput: TypeAlias = str   # LLM 原始文字回應（未解析）


# ── Config schema（依組合關係排序）───────────────────────────────
class PathsConfig(BaseModel):
    """
    路徑設定。輸出路徑未填時依 _DEFAULT_OUTPUT_NAMES 衍生到 outputRoot；
    相對路徑視為相對 outputRoot；絕對路徑原樣使用。
    """
    taskCsvPath:   Path = Field(..., description="前處理產出的標準 Task CSV 路徑")
    promptCmbPath: Path = Field(..., description="Prompt 組合 CSV 的路徑")
    outputRoot:    Path = Field(..., description="輸出檔案的根目錄")

    rawOutputPath:            Optional[Path] = Field(default=None, description="LLM 推論原始暫存 CSV 路徑")
    resultPath:               Optional[Path] = Field(default=None, description="OutputParser 解析後的結構化 CSV 路徑")
    singlePromptCmbOutputDir: Optional[Path] = Field(default=None, description="每個 promptID 獨立存檔的目錄")
    partialInfoPath:          Optional[Path] = Field(default=None, description="Pivot 後的寬格式 CSV 路徑")
    fullInfoPath:             Optional[Path] = Field(default=None, description="合併原始欄位的完整版 CSV 路徑")
    evalDir:                  Optional[Path] = Field(default=None, description="評估圖表與報表的輸出目錄")
    promptPreviewPath:        Optional[Path] = Field(default=None, description="渲染後的 userPrompt 預覽 CSV 路徑")

    # 各輸出路徑留 None 時的預設檔名
    _DEFAULT_OUTPUT_NAMES: ClassVar[Dict[str, str]] = {
        'rawOutputPath':            'raw.csv',
        'resultPath':               'result.csv',
        'singlePromptCmbOutputDir': 'singleOutput',
        'partialInfoPath':          'partialInfo.csv',
        'fullInfoPath':             'fullInfo.csv',
        'evalDir':                  'eval',
        'promptPreviewPath':        'promptPreview.csv',
    }

    @model_validator(mode='after')
    def checkAndMakeDir(self):
        """依 outputRoot 解析所有輸出路徑（None/相對/絕對三種情形），並統一 mkdir。"""
        # 三種情形統一解析：None → 套預設檔名；相對路徑 → 接到 outputRoot 下；絕對路徑 → 原樣。
        # 在 config 載入時就 resolve 完，下游拿到的路徑一律是絕對且可直接用。
        for name, default in self._DEFAULT_OUTPUT_NAMES.items():
            current: Optional[Path] = getattr(self, name)
            if current is None:
                resolved = self.outputRoot / default
            elif not current.is_absolute():
                resolved = self.outputRoot / current
            else:
                resolved = current
            setattr(self, name, resolved)

        # 統一在這裡把所有父目錄建好，下游各階段就「只管寫檔」，不必各自 mkdir。
        # 'Dir' 結尾或 outputRoot 本身當目錄建；其餘是檔案路徑，建它的 parent。
        for fieldName in self.model_fields:
            value = getattr(self, fieldName)
            if isinstance(value, Path):
                target = value if 'Dir' in fieldName or fieldName == 'outputRoot' else value.parent
                target.mkdir(parents=True, exist_ok=True)

        return self


class OllamaServerConfig(BaseModel):
    """Ollama 伺服器連線設定。預設指向本機 port 的 chat 端點。"""
    url: str = Field(default="http://localhost:11434/api/chat", description="Ollama API 端點")
    timeout: int = Field(default=1800, description="API 請求超時時間(秒)")


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
        # 大幅減少 OutputParser 要處理的雜訊（少數不遵守的仍由 labelToLabelCode 兜底回 -1）。
        labelProp = {"type": "string", "enum": self.classes}
        if taskType == "PPI":
            return {
                "type": "object",
                "properties": {"label": labelProp},
                "required": ["label"],
            }
        # batch：要求每筆帶 id（1-based 序號），讓 OutputParser 能把答案對回正確的 pair。
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


class PipelineConfig(BaseModel):
    """
    Pipeline 設定 Schema。
    taskType="PPI"：每個 task 一個預測標的，需設 labelColumn。
    taskType="BC5CDR"：每個 task 多個預測標的，需設 itemTemplate。
    """
    paths: PathsConfig
    taskType: str = Field(..., description="資料集類型，決定推論模式：'PPI' 或 'BC5CDR'")
    selectedModels: List[str] = Field(default_factory=list, description="要進行測試的 LLM 模型清單")
    contextColumns: List[str] = Field(default_factory=list, description="Task CSV 中作為 context 的欄位名稱")
    itemColumns: List[str] = Field(default_factory=list, description="items JSON 中對應 itemTemplate 佔位符的欄位名稱")
    labelColumn: Optional[str] = Field(default=None, description="PPI 模式下攜帶 true label 的欄位名")
    ollamaServer: OllamaServerConfig = Field(default_factory=OllamaServerConfig)
    llmOptions: Dict[str, Any] = Field(default_factory=lambda: {"temperature": 0}, description="LLM 推論參數")
    labelSet: LabelSet = Field(default_factory=LabelSet, description="分類類別清單（字串陣列，如 ['no','yes']）；索引即整數 labelCode")
    maxItemsPerBatch: int = Field(default=1, ge=1, description="每個 LLM task 包含的 item 數；>1 為批次模式")
    concurrencyPerModel: int = Field(default=8, description="每個模型的最大非同步併發數")
    maxConcurrentModels: int = Field(default=1, description="最大同時運行的模型數量")
    taskTemplate: str = Field(..., description="組裝 userPrompt 的文字模板；{key} 對應 context 欄位，{items} 對應批次展開")
    itemTemplate: Optional[str] = Field(default=None, description="單筆 item 的格式化模板；不設定代表單筆模式")

    @field_validator('labelSet', mode='before')
    @classmethod
    def _normalizeLabelSet(cls, v):
        """labelSet 寫成字串清單（如 ['no','yes']），於此包成 LabelSet。"""
        # ergonomic helper：讓 YAML 直接寫 labelSet: ["no","yes"]（而非 labelSet: {classes: [...]}）。
        # mode='before' 表示在 Pydantic 驗 LabelSet 之前先攔截、把 list 包成 {'classes': list}。
        if isinstance(v, LabelSet):
            return v
        if not isinstance(v, list):
            raise ValueError(
                f"labelSet 必須是字串清單（如 ['no','yes']），收到 {type(v).__name__}。"
            )
        return {'classes': v}

    @model_validator(mode='after')
    def validateTaskMode(self):
        """taskType 一致性檢查：必填欄位、禁用欄位、maxItemsPerBatch 限制。"""
        # 這裡是 taskType 的合法值把關點：通過後，下游（loadTaskData / _buildTaskBatches）就能信任
        # taskType ∈ {PPI, BC5CDR}，PPI 以外一律當 BC5CDR，不必再寫不可達的第三分支。
        if self.taskType not in ("PPI", "BC5CDR"):
            raise ValueError(
                f"taskType 必須為 'PPI' 或 'BC5CDR'，收到 '{self.taskType}'。"
            )
        if self.taskType == "PPI":
            # PPI 三條硬規則：
            #  - 必須有 labelColumn（true label 的來源欄）。
            #  - 不該有 itemTemplate（PPI 沒有 item 概念，設了代表用錯模式）。
            #  - maxItemsPerBatch 必為 1（強加 batch 概念只會讓下游邏輯複雜化卻無收益）。
            if not self.labelColumn:
                raise ValueError(
                    "PPI 模式必須設定 labelColumn，"
                    "用於指定 Task CSV 中攜帶 true label 的欄位名。"
                )
            if self.itemTemplate is not None:
                raise ValueError(
                    "PPI 模式不應設定 itemTemplate。"
                    "若資料集為一篇多 item，請將 taskType 改為 'BC5CDR' 並設定 itemColumns。"
                )
            if self.maxItemsPerBatch != 1:
                raise ValueError(
                    f"PPI 模式下 maxItemsPerBatch 必須為 1，目前為 {self.maxItemsPerBatch}。"
                )
        else:
            # BC5CDR：一定要有 itemTemplate，否則 item 無法渲染進 userPrompt。
            if not self.itemTemplate:
                raise ValueError(
                    "BC5CDR 模式必須提供 itemTemplate，"
                    "否則無法將 item 渲染進 userPrompt。"
                )
            # taskTemplate 必須含 {items} 佔位符：渲染好的 item 內容是當成 {items} 值塞進去的，
            # 少了它 format_map 會把整段 items 默默丟掉、模型完全看不到 item → fail-fast。
            if "{items}" not in self.taskTemplate:
                raise ValueError(
                    "BC5CDR 模式的 taskTemplate 必須包含 {items} 佔位符，"
                    "否則展開後的 item 內容無法填入 userPrompt。"
                )
        return self

    def buildResponseFormat(self) -> Dict[str, Any]:
        """回傳 Ollama `format` JSON schema（PPI → {label}；BC5CDR → {answers:[{id,label}]}）。"""
        # facade：把「模式判定」與「schema 產生」綁在一起，LLMEngine 直接拿結果丟給 Ollama。
        # 直接把 taskType 字串傳下去，由 LabelSet 端做字串比對，與下游各處分支一致。
        return self.labelSet.buildOllamaOutputFormat(taskType=self.taskType)


# ── Pipeline 例外體系 ─────────────────────────────────────────────────────
# 階層化例外：上層（Main_PromptCmb）可「捕 PipelineError 一網打盡」也可分別處理；
# 各階段內部明確 raise 對應子類，讓 traceback 一眼看出失敗在哪一步。
class PipelineError(Exception):
    """所有 Pipeline 錯誤的基底類別，Main_PromptCmb.py 統一捕捉。"""
    pass

class DataLoadError(PipelineError):
    """資料或 Prompt 載入失敗：檔案不存在、欄位缺漏、格式錯誤等。"""
    pass

class TaskBuildError(PipelineError):
    """任務建構失敗：模型/prompt 清單為空、JSON 欄位無法解析等。"""
    pass

class InferenceError(PipelineError):
    """LLM 推論階段失敗。"""
    pass

class ParsingError(PipelineError):
    """解析輸出失敗：找不到 raw.csv、解析後無有效資料等。"""
    pass


# ── 任務執行單位 ──────────────────────────────────────────────────────────
class LLMTask(BaseModel):
    """
    一次 LLM API 呼叫的輸入結構。
    唯一性由 composite key (model, promptID, taskID) 保證，不用字串拼接以避免特殊字元歧義。
    """
    # min_length=1：三個 ID 與 userPrompt 不可空（空字串會讓 checkpoint 比對與渲染出問題）。
    # sysPrompt 可空（允許「全部塞 user」的 prompt 設計）。
    taskID:    TaskID    = Field(..., min_length=1, description="批次層級識別碼")
    model:     ModelName = Field(..., min_length=1, description="Ollama 模型名稱")
    promptID:  PromptID  = Field(..., min_length=1, description="Prompt 策略識別碼")
    sysPrompt: str       = Field(default="", description="系統提示詞")
    userPrompt: str      = Field(..., min_length=1, description="使用者提示詞")
    items: List[Dict[str, Any]] = Field(default_factory=list, description="此任務包含的 item 清單（含 sentID/label）")
    context: Dict[str, Any] = Field(default_factory=dict, description="Task 層級 context 欄位")
