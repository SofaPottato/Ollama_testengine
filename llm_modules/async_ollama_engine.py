import asyncio
import json
import logging
import time
from collections import defaultdict
from typing import Dict, List, Any
import csv  # 新增
import os   # 新增
# 請確保你的終端機有安裝這些套件: pip install httpx tenacity tqdm
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.asyncio import tqdm

class AsyncOllamaClient:
    """
    非同步的 API 客戶端，負責與 Ollama 伺服器進行通訊與錯誤重試。
    """
    def __init__(self, api_config: Dict[str, Any], llm_options: Dict[str, Any]):
        self.api_url = api_config.get('url', "http://localhost:11434/api/chat")
        self.timeout = api_config.get('timeout', 1800.0)
        self.llm_options = llm_options

        # 建立高效的非同步 HTTP 連線池
        limits = httpx.Limits(max_keepalive_connections=100, max_connections=100)
        self.client = httpx.AsyncClient(limits=limits, timeout=self.timeout)

    # Tenacity 重試裝飾器：最多試 3 次，等待時間為 1, 2, 4 秒
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError, httpx.ReadTimeout)),
        reraise=False # 重試 3 次都失敗後，回傳 None 交給外層處理
    )
    async def generate(self, model_name: str, sys_prompt: str, user_prompt: str):
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": self.llm_options
        }

        try:
            response = await self.client.post(self.api_url, json=payload)
            response.raise_for_status() 
            return response.json().get('message', {}).get('content', '')
        except Exception as e:
            logging.warning(f"⚠️ 模型 {model_name} 連線異常或逾時: {e}")
            raise # 拋給 tenacity 進行重試

    async def close(self):
        """關閉 HTTP 客戶端"""
        await self.client.aclose()


class MultiModelAsyncEngine:
    """
    支援多模型動態路由的非同步推論引擎。
    會自動為不同的模型建立專屬的併發閘門，確保不互相搶佔資源。
    """
    def __init__(self, api_config: Dict[str, Any], llm_options: Dict[str, Any], exec_settings: Dict[str, Any]):
        self.concurrency_per_model = exec_settings.get('concurrency_per_model', 8)
        self.output_file = exec_settings.get('output_file', 'results.jsonl')
        
        self.client = AsyncOllamaClient(api_config, llm_options)
        
        # 🌟 核心魔法：自動為出現的「每個新模型」配發一個獨立的 Semaphore (併發閘門)
        self.semaphores = defaultdict(lambda: asyncio.Semaphore(self.concurrency_per_model))
        
        self._debug_printed_models = set()

    async def _process_single_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """處理單一任務批次，並根據模型名稱進入對應的排隊閘門"""
        model = task.get('model', 'unknown_model')
        
        async with self.semaphores[model]:
            sys_p = task.get('sys_prompt', '')
            user_p = task.get('user_prompt', '')

            # 每個模型只印出一次啟動提示
            if model not in self._debug_printed_models:
                logging.info(f"\n📢 [Debug] Model '{model}' 已啟動專屬排程，最大併發限制: {self.concurrency_per_model}")
                self._debug_printed_models.add(model)

            # 呼叫 API
            raw_output = await self.client.generate(model, sys_p, user_p)
            
            # 如果重試 3 次都失敗
            if raw_output is None:
                raw_output = "Error: Max retries exceeded or connection failed"

            completed_task = task.copy()
            completed_task['raw_output'] = raw_output
            completed_task['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")

            # ==========================================
            # 🌟 改為即時寫入 CSV 的邏輯
            # ==========================================
            # 把複雜的 batch_data 轉成字串，以免破壞 CSV 格式
            batch_data_str = json.dumps(completed_task.get('batch_data', {}), ensure_ascii=False)
            
            # 定義 CSV 的欄位與要寫入的資料
            row_data = {
                "timestamp": completed_task['timestamp'],
                "model": completed_task.get('model', ''),
                "prompt_id": completed_task.get('prompt_id', ''),
                "sys_prompt": completed_task.get('sys_prompt', ''),
                "user_prompt": completed_task.get('user_prompt', ''),
                "raw_output": raw_output,
                "batch_data": batch_data_str # 複雜結構以字串形式存入單一儲存格
            }

            # 檢查檔案是否已存在 (用來決定要不要寫入標題列 Header)
            file_exists = os.path.isfile(self.output_file)

            # 使用 csv.DictWriter 安全地寫入 (自動處理內文的逗號與換行)
            # newline='' 是為了防止 Windows 底下 CSV 產生多餘的空行
            with open(self.output_file, 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row_data.keys())
                if not file_exists:
                    writer.writeheader() # 檔案剛建立時，先寫入第一行標題
                writer.writerow(row_data)

            return completed_task

    async def run_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """執行所有任務批次"""
        if not tasks:
            logging.warning("⚠️ 任務清單為空！")
            return []

        unique_models = set(t.get('model') for t in tasks)
        logging.info(f"🚀 開始非同步推論！共 {len(tasks)} 筆任務批次。")
        logging.info(f"⚙️ 偵測到 {len(unique_models)} 種模型: {', '.join(unique_models)}")
        logging.info(f"⚙️ 每個模型最大併發限制: {self.concurrency_per_model}")

        # 建立所有協程
        coroutines = [self._process_single_task(task) for task in tasks]
        
        # 併發執行並顯示進度條
        results = await tqdm.gather(*coroutines, desc="總推論進度", unit="batch")
        
        await self.client.close()
        logging.info(f"✅ 所有任務執行完畢！原始 JSONL 備份已儲存至 {self.output_file}")
        return results