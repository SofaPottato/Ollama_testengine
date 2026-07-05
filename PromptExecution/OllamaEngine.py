"""非同步 Ollama 推論引擎：對 promptInfoDf 每列送一次 /api/chat，結果逐筆 append 到 response.csv。"""
import asyncio
import json
import logging
import csv
import os
import httpx
import pandas as pd
from collections import defaultdict
from typing import Dict, Any, Union
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.asyncio import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
# response.csv 欄位順序（RESPONSE_CSV_COLS）的定義在 schemas：寫入端（本檔 appendCsv）與
# 讀取端（DataLoader / ResponseParser）共用同一常數。
from .schemas import ModelName, ResponseAns, RESPONSE_CSV_COLS


#============================================================================#
#   OllamaClient: 單次 API 呼叫層（連線池 + 重試）#
#============================================================================#
class OllamaClient:
    """非同步 Ollama API 客戶端，封裝連線池與 tenacity 重試（最多 3 次，指數退避）。"""
    def __init__(self, apiUrl: str, timeout: float, llmOptions: Dict[str, Any],
                 responseFormat: Dict[str, Any]):
        self.apiUrl = apiUrl
        self.timeout = timeout
        self.llmOptions = llmOptions
        self.responseFormat = responseFormat
        # 連線池上限設 50，遠高於實際併發（concurrencyPerModel * maxConcurrentModels）；
        # 設高一點是預留 buffer，避免併發調大時連線數成為瓶頸。
        limits = httpx.Limits(max_keepalive_connections=50, max_connections=50)
        # 總 timeout 用傳入值（長文本推論可能很慢），但連線建立只給 30s——連不上通常是 Ollama 沒開。
        self.httpClient = httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout(self.timeout, connect=30.0)
        )

    # 只對網路/HTTP 類例外重試（RequestError 已涵蓋 ConnectError/ReadTimeout 等）；程式 bug 類例外
    # 不在清單內，會立刻往上拋而非重試 3 次。reraise=True：重試耗盡後拋「原始例外」而非 tenacity 的
    # RetryError，traceback 更直接。
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        reraise=True
    )
    async def generate(self, modelName: ModelName, sysPrompt: str, userPrompt: str) -> ResponseAns:
        """送出單次推論請求，回傳模型文字回應。網路錯誤自動重試，超限後往上拋。"""
        # stream=False：要完整回應而非逐 token 串流（批次解析不需要串流）。
        # format=responseFormat 讓 Ollama 端強制輸出符合 schema 的 JSON。
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
        except httpx.HTTPStatusError as e:
            # HTTP 錯誤 body 常含 Ollama 的具體訊息（model not found、context length…），截 500 字記下來
            # 方便排錯，然後往上拋觸發重試。其餘網路類例外交給 tenacity，不再攔一層純轉發。
            body = e.response.text[:500] if e.response is not None else ""
            logging.warning(f"[Client] {modelName} HTTP {e.response.status_code}: {body}")
            raise

    async def close(self):
        """關閉 HTTP 客戶端，釋放底層 TCP 連線池。"""
        await self.httpClient.aclose()


#============================================================================#
#   LLMEngine: 任務調度層（Main 只呼叫 runAllTasks 這一個進入點）#
#============================================================================#
class LLMEngine:
    """支援多模型的非同步推論引擎。

    雙層併發控制：modelSemaphoreDict（每模型 in-flight 上限）+ modelConcurrencySemaphore（同時載入幾個模型）。
    每筆完成即 append 寫入 response.csv 並 fsync 落盤，中斷可從 checkpoint 續跑。
    已完成任務的過濾由呼叫端（TaskBuilder.buildPromptInfo）負責，引擎只執行收到的清單。
    結果不透過回傳值傳遞，下游階段直接從 response.csv 讀取。
    """

    # 不在 __init__ 收參數：所有連線/併發設定於 runAllTasks 呼叫時才傳入並建好，
    # 主程式端就能在呼叫點一眼看出這趟推論用到哪些設定。

    def runAllTasks(self, taskDf: pd.DataFrame,
                    apiUrl: str,
                    timeout: float,
                    llmOptions: Dict[str, Any],
                    concurrencyPerModel: int,
                    maxConcurrentModels: int,
                    outputFile: Union[str, Path],
                    responseFormat: Dict[str, Any]) -> None:
        """同步進入點：依傳入設定建好引擎狀態，asyncio.run 跑完所有任務，finally 確保釋放連線池。"""
        # 1) 把這一輪的設定收進 self（跨 async 方法共用），並建好 API 客戶端。
        self.concurrencyPerModel = concurrencyPerModel
        self.maxConcurrentModels = maxConcurrentModels
        self.outputFile = str(outputFile)
        self.ollamaClientObj = OllamaClient(
            apiUrl=apiUrl, timeout=timeout, llmOptions=llmOptions,
            responseFormat=responseFormat,
        )

        # 2) 建雙層併發控制與寫檔鎖：
        #  - modelSemaphoreDict（內層）：限制「同一模型」的 in-flight 請求數。用 defaultdict 讓每個 model
        #    第一次被存取時自動建立專屬 Semaphore。
        #  - modelConcurrencySemaphore（外層）：限制「同時載入幾個模型」，避免一次塞爆 GPU 記憶體。
        #  - fileLock：所有 append 寫檔共用一把鎖，序列化磁碟寫入，避免併發交錯寫出壞掉的 CSV 列。
        self.modelSemaphoreDict = defaultdict(lambda: asyncio.Semaphore(self.concurrencyPerModel))
        self.modelConcurrencySemaphore = asyncio.Semaphore(self.maxConcurrentModels)
        self.initializedModelSet = set()
        self.fileLock = asyncio.Lock()

        # 3) asyncio.run 跑完所有任務。整條 Pipeline 是同步流程，只有推論這步進入非同步；
        #    把 asyncio.run 收斂在引擎內，主程式端就只是一行同步呼叫，不必碰 event loop。
        logging.info(f"[Engine] 派送任務: {len(taskDf)} 筆")

        async def runAndClose():
            # 內層 coroutine 把 try/finally 包進同一個 async context：runTasks 即使中途拋例外，
            # close() 仍會在同一 event loop 內執行、釋放連線池。
            try:
                await self.runTasks(taskDf)
            finally:
                await self.close()

        asyncio.run(runAndClose())
        logging.info(f"[Engine] 推論完成 → {self.outputFile}")

    #============================================================================#
    #   以下為 asyncio 內部方法（依呼叫層級排列：runTasks → 每模型 → 單筆 → 寫檔）#
    #============================================================================#
    async def runTasks(self, taskDf: pd.DataFrame) -> None:
        """所有任務的調度入口：依 model 分組後以 gather 同時啟動各組，組內用 as_completed 交錯執行。

        結果直接 append 至 response.csv，本方法不回傳；下游讀檔取得。
        """
        if taskDf.empty:
            logging.warning("[Engine] 任務清單為空")
            return

        # 依 model 分桶（groupby），為每個模型起一條獨立 group coroutine，才能用外層 semaphore 控制模型載入數。
        modelGroupList = list(taskDf.groupby('model'))

        logging.info(
            f"[Engine] 推論啟動: {len(taskDf)} 任務、{len(modelGroupList)} 模型 "
            f"({', '.join(str(modelName) for modelName, _ in modelGroupList)}) "
            f"(maxConcurrentModels={self.maxConcurrentModels}, "
            f"concurrencyPerModel={self.concurrencyPerModel})"
        )

        # tqdm 用 context manager 包住，gather 拋例外時也能保證 bar 正確關閉。
        # logging_redirect_tqdm：讓 logging 輸出不會把進度條沖掉。
        with tqdm(total=len(taskDf), desc="總推論進度", unit="batch") as progressBar, \
            logging_redirect_tqdm():
            await asyncio.gather(*[
                self.processTaskByModel(modelName, modelDf, progressBar)
                for modelName, modelDf in modelGroupList
            ])

    async def processTaskByModel(self, modelName: ModelName,
                                 modelDf: pd.DataFrame,
                                 progressBar: tqdm) -> None:
        """處理單一模型的所有任務；外層 modelConcurrencySemaphore 控制同時載入幾個模型。"""
        # 外層 semaphore：先拿到「模型載入名額」才開始跑這組，超過 maxConcurrentModels 的組會在此排隊。
        async with self.modelConcurrencySemaphore:
            logging.info(f"[Engine] 模型 {modelName} 取得許可: {len(modelDf)} 筆任務")

            # itertuples 一列即一筆任務；as_completed：誰先跑完就先更新進度條，組內併發由內層 semaphore 控。
            taskCoroutineList = [self.processSingleTask(row) for row in modelDf.itertuples(index=False)]
            for completedCoroutine in asyncio.as_completed(taskCoroutineList):
                await completedCoroutine
                progressBar.update(1)

            logging.info(f"[Engine] 模型 {modelName} 完成，釋放許可")

    async def processSingleTask(self, row) -> None:
        """單一任務的完整生命週期：Semaphore 排隊 → API 呼叫 → append 寫 CSV。

        row 為 promptInfoDf 的一列（itertuples namedtuple，欄位見 TaskBuilder.PROMPT_INFO_COLS）。
        API 失敗不 raise，改寫 "Error:..." 字串，下游 ResponseParser 看到後標 -1 跳過。
        """
        # 內層 semaphore：限制同一模型的 in-flight 請求數，避免單一模型被打爆。
        async with self.modelSemaphoreDict[row.model]:
            # 每個 model 第一次取得 semaphore 時印一次啟動訊息。
            if row.model not in self.initializedModelSet:
                logging.info(f"[Engine] 模型 {row.model} 啟動排程: 併發上限 {self.concurrencyPerModel}")
                self.initializedModelSet.add(row.model)

            # 先拿結果（失敗也回 Error 字串而非 raise），再寫檔。
            responseAns = await self.tryGenerate(row)
            await self.appendCsv(row, responseAns)

    async def tryGenerate(self, row) -> ResponseAns:
        """送出 LLM 請求；例外與空回應都統一回 Error 字串，讓批次能繼續且下游用同方式辨識。"""
        # 單筆失敗「不 raise」，否則 gather 會把整批中斷。改回 "Error:..." 字串，讓這筆照常寫進 response.csv
        # （算「已嘗試/已完成」），下游 ResponseParser 看到 "Error:" 標 -1。
        try:
            responseAns = await self.ollamaClientObj.generate(
                row.model, row.sysPrompt, row.userPrompt
            )
        except Exception as e:
            logging.error(f"[Engine] 任務 {row.taskID} 失敗: {e}")
            responseAns = ""

        return responseAns or "Error: Max retries exceeded or connection failed"

    async def appendCsv(self, row, responseAns: ResponseAns) -> None:
        """以 fileLock 序列化 append 寫 response.csv；fsync 確保中斷時 checkpoint 不遺失最後一筆。"""
        rowDataDict = {
            "model":        row.model,
            "promptCmbID":  row.promptCmbID,
            "taskID":       row.taskID,
            "systemPrompt": row.sysPrompt,
            "userPrompt":   row.userPrompt,
            "responseAns":  responseAns,
            "items":        json.dumps(row.items, ensure_ascii=False),
            "sentence":     json.dumps(row.sentence, ensure_ascii=False),
        }

        # fileLock 序列化所有寫入：多個 coroutine 併發完成時，避免它們交錯寫出半行壞資料。
        async with self.fileLock:
            b_fileExists = os.path.isfile(self.outputFile)
            with open(self.outputFile, 'a', encoding='utf-8-sig', newline='') as f:
                # 固定 fieldnames = RESPONSE_CSV_COLS，保證欄位順序與下游驗證一致。
                csvWriter = csv.DictWriter(f, fieldnames=RESPONSE_CSV_COLS)
                if not b_fileExists:
                    csvWriter.writeheader()
                csvWriter.writerow(rowDataDict)
                # flush + fsync 強制落盤：response.csv 就是 checkpoint，process 被 kill 時也要保證已完成的
                # 最後一筆確實寫到磁碟，否則續跑會重做。
                f.flush()
                os.fsync(f.fileno())

    async def close(self):
        """釋放底層 httpx 連線池。Pipeline 結束時務必呼叫。"""
        await self.ollamaClientObj.close()
