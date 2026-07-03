import asyncio
import json
import logging
import csv
import os
import httpx
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Union
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.asyncio import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from .schemas import PipelineConfig, LLMTask, ModelName, RawOutput

# raw.csv 欄位順序的單一事實來源；下游讀檔時應據此驗證。
# 寫入端（_appendCsv）與讀取端（Pipeline.loadCompletedTaskRunIDs）都引用同一常數，
# 改 schema 只需改這裡一處——但既有 raw.csv 就得刪除，否則欄位對不上會 raise。
RAW_CSV_COLS: List[str] = [
    "model", "promptID", "taskID",
    "systemPrompt", "userPrompt", "rawOutput", "items", "context",
]

# raw.csv 用來判斷任務是否已完成的 composite key 欄位（對應 TaskRunID 三元組）。
TASK_RUN_ID_COLUMNS: Tuple[str, str, str] = ("model", "promptID", "taskID")

class OllamaClient:
    """非同步 Ollama API 客戶端，封裝連線池與 tenacity 重試（最多 3 次，指數退避）。"""
    def __init__(self, apiUrl: str, timeout: float, llmOptions: Dict[str, Any],
                 responseFormat: Dict[str, Any]):
        self.apiUrl = apiUrl
        self.timeout = timeout
        self.llmOptions = llmOptions
        self.responseFormat = responseFormat
        # 連線池上限設 50，遠高於實際併發（concurrencyPerModel * maxConcurrentModels）；
        # 設高一點是為了預留 buffer，避免併發調大時連線數成為瓶頸。
        limits = httpx.Limits(max_keepalive_connections=50, max_connections=50)
        # 總 timeout 用 config 值（預設 1800s，因長文本推論可能很慢），但連線建立只給 30s——連不上通常是 Ollama 沒開
        self.httpClient = httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout(self.timeout, connect=30.0)
        )

    # 只對網路/HTTP 類例外重試（RequestError 已涵蓋 ConnectError/ReadTimeout 等）；
    # 程式 bug 類例外不在清單內，會立刻往上拋而非重試 3 次。
    # reraise=True：重試耗盡後拋「原始例外」而非 tenacity 的 RetryError，traceback 更直接。
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        reraise=True
    )
    async def generate(self, modelName: ModelName, sysPrompt: str, userPrompt: str) -> RawOutput:
        """送出單次推論請求，回傳模型文字回應。網路錯誤自動重試，超限後往上拋。"""
        # stream=False：要完整回應而非逐 token 串流——批次解析不需要串流 
        # format=responseSchema 讓 Ollama 端強制輸出符合 schema 的 JSON。
        payload = {
            "model": modelName,
            "messages": [
                {"role": "system", "content": sysPrompt},
                {"role": "user", "content": userPrompt}
            ],
            "stream": False,
            "options": self.llmOptions,
            "format": self.responseFormat,
        }
        try:
            response = await self.httpClient.post(self.apiUrl, json=payload)
            response.raise_for_status()
            return response.json().get('message', {}).get('content', '')
        except httpx.HTTPStatusError as e: # HTTP 錯誤（4xx, 5xx）也算失敗，觸發重試；超限後記錄狀態碼與部分回應內容，然後往上拋
            # HTTP 錯誤 body 常含 Ollama 的具體訊息（model not found、context length…），
            # 截 500 字記下來方便排錯。
            body = e.response.text[:500] if e.response is not None else ""
            logging.warning(f"[Client] {modelName} HTTP {e.response.status_code}: {body}")
            raise
        except Exception as e:
            logging.warning(f"[Client] {modelName} 連線異常: {e}")
            raise

    async def close(self):
        """關閉 HTTP 客戶端，釋放底層 TCP 連線池。"""
        await self.httpClient.aclose()


class LLMEngine:
    """
    支援多模型動態路由的非同步推論引擎。
    雙層併發控制：modelSemaphoreDict（每模型）+ modelConcurrencySemaphore（跨模型）。
    每筆完成即 append 寫入 raw.csv 並 fsync 落盤，中斷可從 checkpoint 續跑。
    已完成任務的過濾由呼叫端（Pipeline.buildPendingTasks）負責，引擎只執行收到的清單。
    結果不透過回傳值傳遞，下游階段直接從 raw.csv 讀取。
    """

    # ── 建構與生命週期 ──────────────────────────────────────────────────────
    def __init__(self,
                 apiUrl: str,
                 timeout: float,
                 llmOptions: Dict[str, Any],
                 concurrencyPerModel: int,
                 maxConcurrentModels: int,
                 outputFile: Union[str, Path],
                 responseFormat: Dict[str, Any]):
        self.concurrencyPerModel = concurrencyPerModel
        self.maxConcurrentModels = maxConcurrentModels
        self.outputFile = str(outputFile)
        self.ollamaClient = OllamaClient(
            apiUrl=apiUrl, timeout=timeout, llmOptions=llmOptions,
            responseFormat=responseFormat,
        )
        # 雙層併發控制：
        #  - modelSemaphoreDict（內層）：限制「同一模型」的 in-flight 請求數。用 defaultdict 讓每個 model 第一次被存取時自動建立專屬 Semaphore。
        #  - modelConcurrencySemaphore（外層）：限制「同時載入幾個模型」，避免一次塞爆 GPU 記憶體。
        self.modelSemaphoreDict = defaultdict(lambda: asyncio.Semaphore(self.concurrencyPerModel))
        self.modelConcurrencySemaphore = asyncio.Semaphore(self.maxConcurrentModels)
        self.initializedModelSet = set()
        # 所有 append 寫檔共用一把鎖，序列化磁碟寫入，避免併發交錯寫出壞掉的 CSV 列。
        self.fileLock = asyncio.Lock()

    @classmethod
    def fromConfig(cls, config: PipelineConfig, outputFile: Union[str, Path]) -> "LLMEngine":
        """由 PipelineConfig 建立 engine，集中映射 config 欄位到引擎建構參數，避免 Pipeline 端硬寫。"""
        # 把「config → 引擎參數」的對應集中在這個 classmethod，Pipeline 端就不必知道引擎內部需要哪些欄位；之後引擎建構參數變動也只改這裡。
        return cls(
            apiUrl=config.ollamaServer.url,
            timeout=config.ollamaServer.timeout,
            llmOptions=config.llmOptions,
            concurrencyPerModel=config.concurrencyPerModel,
            maxConcurrentModels=config.maxConcurrentModels,
            outputFile=str(outputFile),
            responseFormat=config.buildResponseFormat(),
        )

    # ── Step 1: 全批次調度（公開入口）────────────────────────────────────
    async def runTasks(self, taskList: List[LLMTask]) -> None:
        """
        所有任務的調度入口。依 model 分組後以 gather 同時啟動各組，
        組內用 as_completed 交錯執行並即時更新 tqdm 進度條。
        結果直接 append 至 raw.csv，本方法不回傳；下游讀檔取得。
        """
        if not taskList:
            logging.warning("[Engine] 任務清單為空")
            return

        # 依 model 分桶，為每個模型起一條獨立 group coroutine。
        # 分桶的目的：讓「同模型」的任務歸在一起，才能用外層 semaphore 控制模型載入數。
        tasksByModelDict: Dict[ModelName, List[LLMTask]] = defaultdict(list)
        for task in taskList:
            tasksByModelDict[task.model].append(task)

        logging.info(
            f"[Engine] 推論啟動: {len(taskList)} 任務、{len(tasksByModelDict)} 模型 "
            f"({', '.join(tasksByModelDict)}) "
            f"(maxConcurrentModels={self.maxConcurrentModels}, "
            f"concurrencyPerModel={self.concurrencyPerModel})"
        )

        # tqdm 用 context manager 包住，gather 拋例外時也能保證 bar 正確關閉。
        # logging_redirect_tqdm：讓 logging 輸出不會把進度條沖掉。
        with tqdm(total=len(taskList), desc="總推論進度", unit="batch") as progressBar, \
            logging_redirect_tqdm():
            # gather 同時啟動所有「模型組」，每組內的任務交錯執行。
            await asyncio.gather(*[
                self._processTaskByModel(modelName, modelTaskList, progressBar)
                for modelName, modelTaskList in tasksByModelDict.items()
            ])

    async def _processTaskByModel(self, modelName: ModelName,
                                 modelTaskList: List[LLMTask],
                                 progressBar: tqdm) -> None:
        """處理單一模型的所有任務；外層 modelConcurrencySemaphore 控制同時載入幾個模型。"""
        # 外層 semaphore：先得到「模型載入名額」才開始跑這組，超過 maxConcurrentModels 的組會在此排隊。
        async with self.modelConcurrencySemaphore:
            logging.info(f"[Engine] 模型 {modelName} 取得許可: {len(modelTaskList)} 筆任務")

            # as_completed：誰先跑完就先更新進度條，進度回饋即時；組內真正的併發由內層 semaphore 控。
            taskCoroutineList = [self._processSingleTask(task) for task in modelTaskList]
            for completedCoroutine in asyncio.as_completed(taskCoroutineList):
                await completedCoroutine
                progressBar.update(1)

            logging.info(f"[Engine] 模型 {modelName} 完成，釋放許可")

    # ── Step 2: 單筆任務處理 ─────────────────────────────────────────────
    async def _processSingleTask(self, task: LLMTask) -> None:
        """
        單一任務的完整生命週期：Semaphore 排隊 → API 呼叫 → append 寫 CSV。
        API 失敗不 raise，改寫 "Error:..." 字串，下游 OutputParser 看到後標 -1 跳過。
        """
        # 內層 semaphore：限制同一模型的 in-flight 請求數，避免單一模型被打爆。
        async with self.modelSemaphoreDict[task.model]:
            # 每個 model 第一次取得 semaphore 時印一次啟動訊息。
            if task.model not in self.initializedModelSet:
                logging.info(f"[Engine] 模型 {task.model} 啟動排程: 併發上限 {self.concurrencyPerModel}")
                self.initializedModelSet.add(task.model)

            # 先拿結果（失敗也會回 Error 字串而非 raise），再寫檔。
            rawOutput = await self._tryGenerate(task)
            await self._appendCsv(task, rawOutput)

    async def _tryGenerate(self, task: LLMTask) -> RawOutput:
        """送出 LLM 請求；例外與空回應都統一回 Error 字串，讓批次能繼續且下游用同方式辨識。"""
        # 單筆失敗「不 raise」，否則 gather 會把整批中斷。改成回 "Error:..." 字串，
        # 讓這筆照常寫進 raw.csv（算「已嘗試/已完成」），下游 OutputParser 看到 "Error:" 標 -1。
        try:
            rawOutput = await self.ollamaClient.generate(
                task.model, task.sysPrompt, task.userPrompt
            )
        except Exception as e:
            logging.error(f"[Engine] 任務 {task.taskID} 失敗: {e}")
            rawOutput = ""
            
        return rawOutput or "Error: Max retries exceeded or connection failed"

    async def _appendCsv(self, task: LLMTask, rawOutput: RawOutput) -> None:
        """以 fileLock 序列化 append 寫 raw.csv；fsync 確保中斷時 checkpoint 不遺失最後一筆。"""
        rowDataDict = {
            "model":        task.model,
            "promptID":     task.promptID,
            "taskID":       task.taskID,
            "systemPrompt": task.sysPrompt,
            "userPrompt":   task.userPrompt,
            "rawOutput":    rawOutput,
            "items":        json.dumps(task.items, ensure_ascii=False),
            "context":      json.dumps(task.context, ensure_ascii=False),
        }

        # fileLock 序列化所有寫入：多個 coroutine 併發完成時，避免它們交錯寫出半行壞資料。
        async with self.fileLock:

            b_fileExists = os.path.isfile(self.outputFile)
            with open(self.outputFile, 'a', encoding='utf-8-sig', newline='') as f:
                # 固定 fieldnames = RAW_CSV_COLS，保證欄位順序與下游驗證一致
                csvWriter = csv.DictWriter(f, fieldnames=RAW_CSV_COLS)
                if not b_fileExists:
                    csvWriter.writeheader()
                csvWriter.writerow(rowDataDict)
                # flush + fsync 強制落盤：raw.csv 就是 checkpoint，process 被 kill 時也要保證已完成的最後一筆確實寫到磁碟，否則續跑會重做。
                f.flush()
                os.fsync(f.fileno())

    async def close(self):
        """釋放底層 httpx 連線池。Pipeline 結束時務必呼叫。"""
        await self.ollamaClient.close()
