import os#todo:斷點繼續功能
import time
import math
import re
import requests
import pandas as pd
import logging
import threading 
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
class InferenceManager:
     
    def __init__(self, config):
        """
        初始化推論管理器
        :param config: 完整的設定字典 (從 yaml 讀入)
        """

        self.cfg = config
        self.output_dir = config.get('output_dir', './output')
        self.models = config.get('selected_models', [])
        self.llm_options = config.get('llm_hyperparameters', {})
        self.exec_config = config.get('execution_settings', {})
        self.is_parallel = self.exec_config.get('parallel', False)
        self.max_workers = self.exec_config.get('max_workers', 3)
        self.model_concurrent_requests = self.exec_config.get('model_concurrent_requests', 1)
        self.api_config = config.get('ollama_server', {})
        self.api_url = self.api_config.get('url', "http://localhost:11434/api/chat")
        self.timeout = self.api_config.get('timeout', 600)
        self.max_retries = self.api_config.get('max_retries', 3)
        self.session = requests.Session() 
        adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50)#最多一次送出多少請求
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        self.csv_lock = threading.Lock()     # 防止 CSV 寫入衝突的鎖
        self._debug_lock = threading.Lock()  # 防止 Debug 訊息印出衝突的鎖
        self._debug_printed = False          
        self.batch_settings = config.get('pair_settings', {'pair_number': 10})
        self.pair_number = self.batch_settings.get('pair_number', 10)
        default_template = """Title: {title}\nAbstract: {abstract}\n\nTask: Determine if the Chemical induces the Disease for the following items based strictly on the text.\n\nItems to analyze:\n{items_content}\n\nIMPORTANT OUTPUT RULES:\n1. Output ONLY a numbered list.\n2. Format: "Item X: Yes" or "Item X: No".\n3. Do NOT provide explanations.\n4. Do NOT use Markdown formatting."""
        self.task_template = config.get('task_template', default_template)

    def run(self, data_df, prompt_configs):
        """
        [Public] 執行的主入口
        """
        os.makedirs(self.output_dir, exist_ok=True)
        raw_output_dir = os.path.join(self.output_dir, "raw_results")
        os.makedirs(raw_output_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M")
        final_save_path = os.path.join(raw_output_dir, f"raw_output_{timestamp}.csv")
        temp_save_path = os.path.join(raw_output_dir, f"temp_{timestamp}.csv")

        # 準備任務清單 (按模型分組，防止 VRAM 反覆載入)
        logging.info("============ 正在準備任務批次 (Task Preparation) ============")
        tasks_dict = self._prepare_tasks(data_df, prompt_configs)
        
        total_tasks = sum(len(q) for q in tasks_dict.values())
        logging.info(f"總任務數 (Batches): {total_tasks}")
        logging.info(f"執行模式: {'平行處理 (Parallel)' if self.is_parallel else '序列處理 (Sequential)'}")

        columns = ["Data_ID", "PMID", "Model", "Prompt_ID", "E1", "E2", "True_Label", "Pred_Label", "Raw_Output"]
        pd.DataFrame(columns=columns).to_csv(temp_save_path, index=False, encoding='utf-8-sig')

        try:
            if self.is_parallel:
                self._run_parallel(tasks_dict, temp_save_path)
            else:
                self._run_sequential(tasks_dict, temp_save_path)
        except KeyboardInterrupt:
            logging.warning("\n⚠️ 使用者中斷執行！目前進度已保留在暫存檔中。")
            return temp_save_path

        logging.info("🔄 正在整合與排序最終結果...")
        if os.path.exists(temp_save_path):
            final_df = pd.read_csv(temp_save_path)
            final_df = final_df.sort_values(['Model', 'Prompt_ID', 'Data_ID'])
            final_df.to_csv(final_save_path, index=False, encoding='utf-8-sig')
            os.remove(temp_save_path)
            logging.info(f"✅ 推論完成！檔案已儲存至: {final_save_path}")
            

            return final_save_path
        else:
            logging.error("❌ 錯誤：未產生任何結果檔案。")
            return None

    def _prepare_tasks(self, df, prompt_configs):
        """[Private] 將資料、模型、Prompt 展開為按模型分組的任務字典"""
        grouped = df.groupby('PMID')
        
        base_batches = []
        for pmid, group in grouped:
            title = group.iloc[0]['Title']
            abstract = str(group.iloc[0]['Abstract'])
            
            pairs_list = []#設置輸入配對
            for idx, row in group.iterrows():
                pairs_list.append({
                    'orig_idx': idx,
                    'E1_Name': row['E1_Name'],
                    'E2_Name': row['E2_Name'],
                    'True_Label': row.get('Relation_Type', row.get('Label', ''))
                })
            
            for i in range(0, len(pairs_list), self.pair_number):
                batch_pairs = pairs_list[i : i + self.pair_number]
                base_batches.append({
                    'pmid': pmid,
                    'title': title,
                    'abstract': abstract,
                    'batch_pairs': batch_pairs
                })
        
        model_task_queues = {model: [] for model in self.models}
        
        for model in self.models:
            for p_config in prompt_configs:
                for batch in base_batches:
                    model_task_queues[model].append({
                        'model': model,
                        'sys_prompt': p_config['text'],
                        'prompt_id': p_config['id'],
                        'batch_data': batch
                    })
                    
        return model_task_queues

    def _process_model_queue(self, model_name, task_list, temp_path, progress_bar=None):
        """[Private] 專屬執行緒函式：處理單一模型的所有任務 (支援模型內部的平行併發)"""
        results_buffer = []
        
        with ThreadPoolExecutor(max_workers=self.model_concurrent_requests) as inner_executor:
            futures = [inner_executor.submit(self._process_single_task, task) for task in task_list]
            
            for future in as_completed(futures):
                try:
                    batch_res = future.result()
                    results_buffer.extend(batch_res)
                    
                    if len(results_buffer) >= (100 * self.pair_number):
                        self._append_to_csv(results_buffer, temp_path)
                        results_buffer = [] 
                        
                except Exception as e:
                    logging.error(f"❌ Task Error ({model_name}): {e}")
                    
                if progress_bar:
                    progress_bar.update(1)

        if results_buffer:
            self._append_to_csv(results_buffer, temp_path)

    def _run_parallel(self, tasks_dict, temp_path):
        """[Private] 平行執行模式 (一模型一執行緒)"""
        num_models = len(tasks_dict)
        workers = min(self.max_workers, num_models)
        
        logging.info(f"⚠️ 平行模式開啟 (分配了 {workers} 個專屬執行緒處理 {num_models} 個模型)")
        total_tasks = sum(len(q) for q in tasks_dict.values())
        
        with tqdm(total=total_tasks, desc="平行推論總進度", unit="batch") as pbar:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = []
                for model_name, task_list in tasks_dict.items():
                    future = executor.submit(self._process_model_queue, model_name, task_list, temp_path, pbar)
                    futures.append(future)
                
                for future in as_completed(futures):
                    future.result() 

    def _run_sequential(self, tasks_dict, temp_path):
        """[Private] 序列執行模式"""
        total_tasks = sum(len(q) for q in tasks_dict.values())
        
        with tqdm(total=total_tasks, desc="序列推論總進度", unit="batch") as pbar:
            for model_name, task_list in tasks_dict.items():
                logging.info(f"🚀 開始處理模型: {model_name}")
                self._process_model_queue(model_name, task_list, temp_path, pbar)
                 
    def _process_single_task(self, task):
        """[Private] 處理單一原子任務 (包含 Prompt 生成、API 呼叫與解析)"""
        batch_data = task['batch_data']
        model = task['model']
        
        user_text = self._create_batch_prompt(
            batch_data['title'], 
            batch_data['abstract'], 
            batch_data['batch_pairs']
        )
        
        if not self._debug_printed:
            with self._debug_lock:
                if not self._debug_printed:
                    debug_msg = (
                        f"\n{'='*60}\n"
                        f"📢 正在檢視模型: {model} | Prompt ID: {task['prompt_id']}\n"
                        #f"{'-'*30}\n"
                        f"【System Prompt】:  {task['sys_prompt']}"
                        #f"{'-'*30}\n"
                        #f"【User Prompt】:\n{user_text}\n"
                        f"{'='*60}"
                    )
                    logging.info(debug_msg)
                    self._debug_printed = True
            
        raw_response = self._query_ollama(model, task['sys_prompt'], user_text)
        parsed_answers = self._parse_batch_response(raw_response, len(batch_data['batch_pairs']))
        
        results = []
        for j, pair_info in enumerate(batch_data['batch_pairs']):
            ans = parsed_answers[j] if j < len(parsed_answers) else "Index_Error"
            results.append({
                "Data_ID": pair_info['orig_idx'],
                "PMID": batch_data['pmid'],
                "Model": model,
                "Prompt_ID": task['prompt_id'],
                "E1": pair_info['E1_Name'],
                "E2": pair_info['E2_Name'],
                "True_Label": pair_info['True_Label'],
                "Pred_Label": ans,
                "Raw_Output": raw_response
            })
        return results

    def _create_batch_prompt(self, title, abstract, pairs):
        """[Private] 使用 Template 建立 Prompt"""
        items_content = ""
        for i, pair in enumerate(pairs, 1):
            items_content += f"Item {i}: Chemical: {pair['E1_Name']} | Disease: {pair['E2_Name']}\n"
            
        try:
            return self.task_template.format(
                title=title,
                abstract=abstract,
                items_content=items_content
            )
        except KeyError as e:
            logging.error(f"Error: Template format error. Missing key: {e}")
            return f"Error: Template format error. Missing key: {e}"

    def _query_ollama(self, model, sys_prompt, user_prompt):
        """[Private] 發送 API 請求 (含 Timeout 與 Exponential Backoff Retry 機制)"""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": self.llm_options
        }

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(self.api_url, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json().get('message', {}).get('content', '')
                else:
                    err = f"HTTP {response.status_code}: {response.text}"
                    logging.warning(f"⚠️ API Error (Attempt {attempt+1}): {err}")
            except Exception as e:
                logging.warning(f"⚠️ Connection Error (Attempt {attempt+1}): {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)

        logging.error("❌ Error: Max retries exceeded during API call.")
        return "Error: Max retries exceeded"

    def _parse_batch_response(self, text, batch_size):
        clean_results = ["Parse_Error"] * batch_size
        if not text or "Error:" in text:
            return clean_results
        pattern = re.compile(r"(?:Item|No\.?|^|\n)\s*\**(\d+)\**[^a-zA-Z0-9]*(Yes|No)", re.IGNORECASE)
        matches = pattern.findall(text)
        for num_str, answer in matches:
            idx = int(num_str) - 1  
            if 0 <= idx < batch_size:
                clean_results[idx] = answer.title()
                
        return clean_results
    def _append_to_csv(self, data, path):
        """[Private] 將資料 Append 到 CSV """
        if not data:
            return
            
        with self.csv_lock:
            pd.DataFrame(data).to_csv(path, mode='a', header=False, index=False, encoding='utf-8-sig')