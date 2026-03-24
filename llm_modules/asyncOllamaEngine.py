import asyncio
import json
import logging
import time
import csv  
import os  
import httpx
from collections import defaultdict
from typing import Dict, List, Any, Union
from pathlib import Path  
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.asyncio import tqdm

class AsyncOllamaClient:
    """
    非同步的 API 客戶端，負責與 Ollama 伺服器進行通訊與錯誤重試。
    """
    def __init__(self, apiUrl: str, timeout: float, llmOptions: Dict[str, Any]):
        self.apiUrl = apiUrl
        self.timeout = timeout
        self.llmOptions = llmOptions

        limits = httpx.Limits(max_keepalive_connections=100, max_connections=100)
        self.client = httpx.AsyncClient(limits=limits, timeout=self.timeout)

    # Tenacity 重試裝飾器
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError, httpx.ReadTimeout)),
        reraise=False # 重試 3 次都失敗後，回傳 None 交給外層處理
    )
    async def generate(self, modelName: str, sysPrompt: str, userPrompt: str):
        payload = {
            "model": modelName,
            "messages": [
                {"role": "system", "content": sysPrompt},
                {"role": "user", "content": userPrompt}
            ],
            "stream": False,
            "options": self.llmOptions
        }

        try:
            response = await self.client.post(self.apiUrl, json=payload)
            response.raise_for_status() 
            return response.json().get('message', {}).get('content', '')
        except Exception as e:
            logging.warning(f"⚠️ 模型 {modelName} 連線異常或逾時: {e}")
            raise 

    async def close(self):
        """關閉 HTTP 客戶端"""
        await self.client.aclose()


class MultiModelAsyncEngine:
    """
    支援多模型動態路由的非同步推論引擎。
    會自動為不同的模型建立專屬的併發閘門，確保不互相搶佔資源。
    """
    def __init__(self, 
                 apiUrl: str, 
                 timeout: float, 
                 llmOptions: Dict[str, Any], 
                 concurrencyPerModel: int, 
                 outputFile: Union[str, Path]):  
                 
        self.concurrencyPerModel = concurrencyPerModel
        self.outputFile = str(outputFile) 

        self.client = AsyncOllamaClient(apiUrl=apiUrl, timeout=timeout, llmOptions=llmOptions)
        self.semaphores = defaultdict(lambda: asyncio.Semaphore(self.concurrencyPerModel))
        
        self.debugPrintedModels = set()
        self.fileLock = asyncio.Lock()
        
    async def processSingleTask(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """處理單一任務批次，並根據模型名稱進入對應的排隊閘門"""
        model = task.get('model', 'unknown_model')
        
        async with self.semaphores[model]:
            systemPrompt = task.get('sysPrompt', '')
            userPrompt = task.get('userPrompt', '')


            if model not in self.debugPrintedModels:
                logging.info(f"\n📢 [Debug] Model '{model}' 已啟動專屬排程，最大併發限制: {self.concurrencyPerModel}")
                self.debugPrintedModels.add(model)


            rawOutput = await self.client.generate(model, systemPrompt, userPrompt)
            

            if rawOutput is None:
                rawOutput = "Error: Max retries exceeded or connection failed"

            completedTask = task.copy()
            completedTask['rawOutput'] = rawOutput
            completedTask['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")

            batchDataStr = json.dumps(completedTask.get('batchData', {}), ensure_ascii=False)
            
            # 定義 CSV 的欄位與要寫入的資料
            row_data = {
                "timestamp": completedTask['timestamp'],
                "model": completedTask.get('model', ''),
                "promptID": completedTask.get('promptID', ''),
                "systemPrompt": completedTask.get('sysPrompt', ''),
                "userPrompt": completedTask.get('userPrompt', ''),
                "rawOutput": rawOutput,
                "batchData": batchDataStr 
            }


            fileExists = os.path.isfile(self.outputFile)

            async with self.fileLock:
                fileExists = os.path.isfile(self.outputFile)
                with open(self.outputFile, 'a', encoding='utf-8-sig', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=row_data.keys())
                    if not fileExists:
                        writer.writeheader()
                    writer.writerow(row_data)

            return completedTask

    async def executeAsyncInferenceBatches(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """執行所有任務批次"""
        if not tasks:
            logging.warning("⚠️ 任務清單為空！")
            return []

        uniqueModels = set(t.get('model') for t in tasks)
        logging.info(f"🚀 開始非同步推論！共 {len(tasks)} 筆任務批次。")
        logging.info(f"⚙️ 偵測到 {len(uniqueModels)} 種模型: {', '.join(uniqueModels)}")
        logging.info(f"⚙️ 每個模型最大併發限制: {self.concurrencyPerModel}")


        coroutines = [self.processSingleTask(task) for task in tasks]
        
        results = await tqdm.gather(*coroutines, desc="總推論進度", unit="batch")
        
        await self.client.close()
        logging.info(f"✅ 所有任務執行完畢！原始 JSONL 備份已儲存至 {self.outputFile}")
        return results