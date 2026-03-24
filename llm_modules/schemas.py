from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Dict, Any, List, Optional


class pathsConfig(BaseModel):
    dataPath: Path = Field(..., description="原始資料集 CSV 的路徑")
    promptsPath: Path = Field(..., description="Prompt 範本 CSV 的路徑")
    mainOutputDir: Path = Field(..., description="主要輸出的根目錄")
    singlePromptOutputDir: Path = Field(..., description="單一 Prompt 獨立輸出的目錄")
    evalDataDir: Path = Field(..., description="評估圖表與結果的輸出目錄")
    rawOutputPath: Path = Field(..., description="解析後的 CSV 輸出路徑")
    resultOutputPath: Path = Field(..., description="清理後結果的 CSV 輸出路徑")
    mergedLlmOutputPath: Path = Field(..., description="最終合併資料的 CSV 輸出路徑")
    rawTempOutputPath: Optional[Path] = Field(default=None, description="自訂暫存檔路徑 (選填)")

    @field_validator('*')
    @classmethod
    def ensure_directories(cls, v: Any, info: ValidationInfo) -> Any:
        if isinstance(v, Path) and ('Dir' in info.field_name):
            v.mkdir(parents=True, exist_ok=True)
        return v
class PairConfig(BaseModel):
    pairNumbers: int = Field(default=10, description="每個 Prompt 要放入的資料對數量")
    
class LLMAppConfig(BaseModel):
    """最外層的整體設定檔 Schema"""
    paths: pathsConfig
    selectedModels: List[str] = Field(default_factory=list, description="要進行測試的 LLM 模型清單")
    testLimits: Optional[int] = Field(default=None, description="開發測試用：限制讀取的資料筆數")
    apiUrl: str = Field(default="http://localhost:11434/api/chat", description="Ollama API 端點")
    concurrencyPerModel: int = Field(default=8, description="每個模型的最大非同步併發數")
    timeout: int = Field(default=1800, description="API 請求超時時間(秒)")
    llmOptions: Dict[str, Any] = Field(default_factory=lambda: {"temperature": 0}, description="LLM 推論參數")
    selectedModels: List[str] = Field(default_factory=lambda: ["llama3.2:1b"])
    pairSettings: PairConfig = Field(default_factory=PairConfig)
    taskTemplate: str = Field(
        default="{title}\n{abstract}\n{itemsContent}", 
        description="組裝 Prompt 的預設文字模板"
    )

# ==========================================
# 1. 自定義異常 (Custom Exceptions)
# ==========================================
class PipelineError(Exception):
    """所有 Pipeline 錯誤的基底類別"""
    pass

class DataLoadError(PipelineError):
    """資料或 Prompt 載入失敗"""
    pass

class TaskBuildError(PipelineError):
    """任務建構失敗 (例如缺少必要欄位)"""
    pass

class InferenceError(PipelineError):
    """LLM 推論階段失敗"""
    pass

class ParsingError(PipelineError):
    """解析輸出失敗"""
    pass

# ==========================================
# 2. 資料結構定義 (Data Models)
# ==========================================
@dataclass
class LLMTask:
    """
    LLM 推論任務的標準資料結構。
    告別 Dict[str, Any]，讓每個欄位都有明確的家。
    """
    model: str
    promptID: str
    sysPrompt: str
    userPrompt: str
    batchData: Dict[str, Any] 