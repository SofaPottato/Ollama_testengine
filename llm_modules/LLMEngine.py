import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
from .asyncOllamaEngine import MultiModelAsyncEngine
from dataclasses import asdict
from .schemas import LLMTask


class InferenceManager:
    def __init__(
        self, 
        rawOutputPath: Path,  
        apiUrl: str = "http://localhost:11434/api/chat",
        timeout: int = 1800,
        llmOptions: Dict[str, Any] = None,
        concurrencyPerModel: int = 8
    ):
        """
        初始化純粹的推論引擎。
        它不關心資料長怎樣，也不管 LLM 回傳什麼，只負責「排隊、打 API、存檔」。
        """
        logging.info("Initializing Pure InferenceManager()...")  
        
        self.rawCsvOutputPath = rawOutputPath
        self.rawCsvOutputPath.parent.mkdir(parents=True, exist_ok=True)

        self.engine = MultiModelAsyncEngine(
            apiUrl=apiUrl,
            timeout=timeout,
            llmOptions=llmOptions if llmOptions is not None else {"temperature": 0},
            concurrencyPerModel=concurrencyPerModel, 
            outputFile=str(self.rawCsvOutputPath)
        )

    def dispatchTasksToAsyncEngine(self, tasksList: List[LLMTask]) -> str:
        """
        主執行入口：接收已經組裝好的 tasksList，交給 Ollama 執行
        """
        logging.info("==== [LLMEngine] Starting Async Inference ====")
        
        if not tasksList:
            logging.error("❌ 接收到的任務清單為空，無法執行推論。")
            return ""
        if self.rawCsvOutputPath.exists():
            logging.info(f"📝 發現既存的暫存檔，引擎將自動接續寫入 (Append Mode): {self.rawCsvOutputPath}")
        else:
            logging.info(f"📄 建立全新暫存檔: {self.rawCsvOutputPath}")

        logging.info(f"🚀 交接給非同步引擎執行 (總任務批次數: {len(tasksList)})...")
        dict_tasks = [asdict(task) for task in tasksList]
        resultsList = asyncio.run(self.engine.executeAsyncInferenceBatches(dict_tasks))
        
        if not resultsList:
            logging.error("❌ 推論引擎回傳空結果。")
            return ""

        logging.info(f"✅ 推論完成！原始回應 (Raw Output) 已儲存至: {self.rawCsvOutputPath}")

        return str(self.rawCsvOutputPath)