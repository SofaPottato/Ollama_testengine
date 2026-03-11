import os
import time
import math
import re
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class InferenceManager:
    def __init__(self, config):
        """
        初始化推論管理器
        :param config: 完整的設定字典 (從 yaml 讀入)
        """
        self.cfg = config
        
        # 1. 基礎設定
        self.output_dir = config.get('output_dir', './output')
        self.models = config.get('selected_models', [])
        
        # 2. LLM 參數 (temperature, top_p 等)
        self.llm_options = config.get('llm_hyperparameters', {})
        
        # 3. 執行設定 (是否平行, Workers 數量)
        self.exec_config = config.get('execution_settings', {})
        self.is_parallel = self.exec_config.get('parallel', False)
        self.max_workers = self.exec_config.get('max_workers', 4)
        
        # 4. API 設定 (URL, Timeout, Retry)
        self.api_config = config.get('ollama_server', {})
        self.api_url = self.api_config.get('url', "http://localhost:11434/api/chat")
        self.timeout = self.api_config.get('timeout', 1800)
        self.max_retries = self.api_config.get('max_retries', 3)
        
        # 5. 批次與 Prompt 設定
        self.batch_settings = config.get('pair_settings', {'pair_number': 10})
        self.pair_number = self.batch_settings.get('pair_number', 10)
        
        # 預設 Template (防呆)
        default_template = """Title: {title}\nAbstract: {abstract}\nItems:\n{items_content}\nAnswer Yes or No."""
        self.task_template = config.get('task_template', default_template)

    def run(self, data_df, prompt_configs):
        """
        [Public] 執行的主入口
        :param data_df: 包含資料的 DataFrame
        :param prompt_configs: 由 PromptManager 生成的 list of dict
        :return: 最終結果 CSV 的路徑
        """
        # 1. 準備目錄與檔案路徑
        os.makedirs(self.output_dir, exist_ok=True)
        raw_output_dir = os.path.join(self.output_dir, "raw_results")
        os.makedirs(raw_output_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M")
        final_save_path = os.path.join(raw_output_dir, f"raw_output_{timestamp}.csv")
        temp_save_path = os.path.join(raw_output_dir, f"temp_{timestamp}.csv")

        # 2. 準備任務清單 (Flatten Tasks)
        print("============正在準備任務批次 (Task Preparation)============")
        tasks = self._prepare_tasks(data_df, prompt_configs)
        print(f"總任務數: {len(tasks)} (Batches * Models * Prompts)")
        print(f"執行模式: {'平行處理 (Parallel)' if self.is_parallel else '序列處理 (Sequential)'}")

        # 初始化暫存檔 (寫入 Header)
        columns = ["Data_ID", "PMID", "Model", "Prompt_ID", "E1", "E2", "True_Label", "Pred_Label", "Raw_Output"]
        pd.DataFrame(columns=columns).to_csv(temp_save_path, index=False, encoding='utf-8-sig')

        # 3. 執行推論
        results_buffer = []
        try:
            if self.is_parallel:
                self._run_parallel(tasks, temp_save_path)
            else:
                self._run_sequential(tasks, temp_save_path)
        except KeyboardInterrupt:
            print("\n⚠️ 使用者中斷執行！目前進度已保留在暫存檔中。")
            return temp_save_path

        # 4. 整合最終結果
        print("🔄 正在整合與排序最終結果...")
        if os.path.exists(temp_save_path):
            final_df = pd.read_csv(temp_save_path)
            # 依照邏輯排序
            final_df = final_df.sort_values(['Model', 'Prompt_ID', 'Data_ID'])
            final_df.to_csv(final_save_path, index=False, encoding='utf-8-sig')
            
            # 刪除暫存檔
            os.remove(temp_save_path)
            print(f"✅ 推論完成！檔案已儲存至: {final_save_path}")
            return final_save_path
        else:
            print("❌ 錯誤：未產生任何結果檔案。")
            return None

    def _prepare_tasks(self, df, prompt_configs):
        """[Private] 將資料、模型、Prompt 展開為單一任務列表"""
        tasks = []
        grouped = df.groupby('PMID')
        
        # 預先處理 Batch (減少重複計算)
        base_batches = []
        for pmid, group in grouped:
            title = group.iloc[0]['Title']
            abstract = str(group.iloc[0]['Abstract'])
            
            pairs_list = []
            for idx, row in group.iterrows():
                pairs_list.append({
                    'orig_idx': idx,
                    'E1_Name': row['E1_Name'],
                    'E2_Name': row['E2_Name'],
                    'True_Label': row.get('Relation_Type', row.get('Label', ''))
                })
            
            # 切分 Batch
            for i in range(0, len(pairs_list), self.pair_number):
                batch_pairs = pairs_list[i : i + self.pair_number]
                base_batches.append({
                    'pmid': pmid,
                    'title': title,
                    'abstract': abstract,
                    'batch_pairs': batch_pairs
                })
        
        # 展開所有組合 (Model x Prompt x Batch)
        for model in self.models:
            for p_config in prompt_configs:
                for batch in base_batches:
                    tasks.append({
                        'model': model,
                        'sys_prompt': p_config['text'],
                        'prompt_id': p_config['id'],
                        'batch_data': batch
                    })
        return tasks

    def _run_parallel(self, tasks, temp_path):
        """[Private] 平行執行模式"""
        results_buffer = []
        print(f"⚠️ 平行模式開啟 (Workers={self.max_workers})")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_single_task, task) for task in tasks]
            
            for future in tqdm(as_completed(futures), total=len(tasks), desc="平行推論進度", unit="batch"):
                try:
                    batch_res = future.result()
                    results_buffer.extend(batch_res)
                    
                    # 批量寫入 (減少 I/O)
                    if len(results_buffer) >= 50:
                        self._append_to_csv(results_buffer, temp_path)
                        results_buffer = [] # 清空
                except Exception as e:
                    print(f"❌ Task Error: {e}")

        # 寫入剩餘資料
        if results_buffer:
            self._append_to_csv(results_buffer, temp_path)

    def _run_sequential(self, tasks, temp_path):
        """[Private] 序列執行模式"""
        results_buffer = []
        for task in tqdm(tasks, total=len(tasks), desc="序列推論進度", unit="batch"):
            try:
                batch_res = self._process_single_task(task)
                results_buffer.extend(batch_res)
                
                if len(results_buffer) >= 10:
                    self._append_to_csv(results_buffer, temp_path)
                    results_buffer = []
            except Exception as e:
                print(f"❌ Task Error: {e}")

        if results_buffer:
            self._append_to_csv(results_buffer, temp_path)

    def _process_single_task(self, task):
        """[Private] 處理單一原子任務 (Atomic Task)"""
        batch_data = task['batch_data']
        model = task['model']
        
        # 1. 建立 User Prompt
        user_text = self._create_batch_prompt(
            batch_data['title'], 
            batch_data['abstract'], 
            batch_data['batch_pairs']
        )
        # 印出第一個prompt
        if not hasattr(self, '_debug_printed'):
            print("\n" + "="*60)
            print(f"正在檢視模型: {model} | Prompt ID: {task['prompt_id']}")
            print("-" * 30)
            print("【System Prompt】:")
            print(task['sys_prompt'])
            print("-" * 30)
            print("【User Prompt】")
            print(user_text)
            print("="*60 + "\n")
            self._debug_printed = True
        # ===============================================================
        # 2. 呼叫 API
        raw_response = self._query_ollama(model, task['sys_prompt'], user_text)
        
        # 3. 解析結果
        parsed_answers = self._parse_batch_response(raw_response, len(batch_data['batch_pairs']))
        
        # 4. 格式化輸出
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
            return f"Error: Template format error. Missing key: {e}"

    def _query_ollama(self, model, sys_prompt, user_prompt):
        """[Private] 發送 API 請求 (含 Retry 機制)"""
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
                response = requests.post(self.api_url, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json().get('message', {}).get('content', '')
                else:
                    err = f"HTTP {response.status_code}: {response.text}"
                    print(f"⚠️ API Error (Attempt {attempt+1}): {err}")
                    logging.info(f"⚠️ API Error (Attempt {attempt+1}): {err}")
            except Exception as e:
                print(f"⚠️ Connection Error (Attempt {attempt+1}): {e}")
            
            # Exponential backoff
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)

        return "Error: Max retries exceeded"

    def _parse_batch_response(self, text, batch_size):
        """[Private] 解析 LLM 回傳的 Item List"""
        clean_results = ["Parse_Error"] * batch_size
        if not text or "Error:" in text:
            return clean_results

        for i in range(1, batch_size + 1):
            pattern = re.compile(rf"(?:Item|No\.?|^|\n)\s*\**{i}\**[^a-zA-Z0-9]*(Yes|No)", re.IGNORECASE)
            match = pattern.search(text)
            if match:
                clean_results[i-1] = match.group(1).title() # Yes/No
        return clean_results

    def _append_to_csv(self, data, path):
        """[Private] 將資料 Append 到 CSV"""
        pd.DataFrame(data).to_csv(path, mode='a', header=False, index=False, encoding='utf-8-sig')