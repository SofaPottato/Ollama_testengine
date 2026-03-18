import asyncio
import logging
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import json
# 引入非同步引擎
from .async_ollama_engine import MultiModelAsyncEngine

class InferenceManager:
    def __init__(self, config: Dict[str, Any]):
        logging.info("Initializing InferenceManager() with Async Engine & Batching")
        self.cfg = config
        
        # 1. 模型與路徑設定
        self.models = config.get("selected_models", ["gemma3:270m"])
        paths_cfg = config.get("paths", {})
        self.output_dir = Path(paths_cfg.get('mainOutputDir', './data/llm_output/'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_output_path = self.output_dir / "raw_inference_results_temp.csv"
        self.csv_output_path = self.output_dir / "raw_inference_results.csv"
        
        # 2. 批次處理設定 (從你原本的 config 讀取)
        pair_settings = config.get('pair_settings', {})
        self.pair_number = pair_settings.get('pair_number', 10)
        self.task_template = config.get("task_template", "{title}\n{abstract}\n{items_content}")
        
        # 3. 準備傳給 Async Engine 的設定
        api_config = config.get("ollama_server", {"url": "http://localhost:11434/api/chat", "timeout": 1800})
        llm_options = config.get("llm_hyperparameters", {"temperature": 0})
        exec_cfg = config.get("execution_settings", {})
        exec_settings = {
            'concurrency_per_model': exec_cfg.get("model_concurrent_requests", 12),
            'output_file': str(self.jsonl_output_path)
        }
        
        self.engine = MultiModelAsyncEngine(api_config, llm_options, exec_settings)

    def run(self, df: pd.DataFrame, prompts: List[Dict[str, str]]) -> str:
        """主執行入口"""
        logging.info("==== [Step 3.1] Preparing Batched Tasks ====")
        tasks = self._prepare_tasks(df, prompts)
        
        if not tasks:
            logging.error("❌ 無法生成任何任務。")
            return ""

        if self.jsonl_output_path.exists():
            self.jsonl_output_path.unlink() # 清理舊的暫存檔

        logging.info(f"🚀 交接給非同步引擎執行 (總任務批次數: {len(tasks)})...")
        results = asyncio.run(self.engine.run_tasks(tasks))
        
        if not results:
            logging.error("❌ 推論引擎回傳空結果。")
            return ""

        # 將完成的 JSONL 透過 Regex 解析並轉回 Pipeline 所需的 CSV 格式
        return self._parse_and_convert_to_csv()

    def _prepare_tasks(self, df: pd.DataFrame, prompt_configs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """將資料依 PMID 分組、切分 Batch，並生成 User Prompt"""
        tasks = []
        grouped = df.groupby('PMID')
        base_batches = []
        
        # 1. 預先處理 Batch
        for pmid, group in grouped:
            title = group.iloc[0].get('Title', '')
            abstract = str(group.iloc[0].get('Abstract', ''))
            
            pairs_list = []
            for idx, row in group.iterrows():
                pairs_list.append({
                    'orig_idx': idx,
                    'E1_Name': row.get('E1_Name', ''),
                    'E2_Name': row.get('E2_Name', ''),
                    'True_Label': row.get('Relation_Type', row.get('Label', ''))
                })
            
            # 切分 Batch (依據 pair_number，例如每 10 個一組)
            for i in range(0, len(pairs_list), self.pair_number):
                batch_pairs = pairs_list[i : i + self.pair_number]
                base_batches.append({
                    'pmid': pmid,
                    'title': title,
                    'abstract': abstract,
                    'batch_pairs': batch_pairs
                })
        
        # 2. 組合 Model x Prompt x Batch
        for model in self.models:
            for p_config in prompt_configs:
                prompt_id = p_config.get('Prompt_ID', p_config.get('id', 'Unknown'))
                sys_prompt = p_config.get('Prompt_Text', p_config.get('text', ''))
                
                for batch in base_batches:
                    # 建立 items_content
                    items_content = ""
                    for i, pair in enumerate(batch['batch_pairs'], 1):
                        items_content += f"Item {i}: Chemical: {pair['E1_Name']} | Disease: {pair['E2_Name']}\n"
                    
                    # 套用 YAML 中的 Template
                    try:
                        user_text = self.task_template.format(
                            title=batch['title'],
                            abstract=batch['abstract'],
                            items_content=items_content
                        )
                    except KeyError as e:
                        logging.error(f"Template 格式錯誤，缺少對應的 Key: {e}")
                        user_text = f"Title: {batch['title']}\nAbstract: {batch['abstract']}\nItems:\n{items_content}"

                    # 將資料打包進 task 字典，交給 Async Engine
                    tasks.append({
                        'model': model,
                        'prompt_id': prompt_id,
                        'sys_prompt': sys_prompt,
                        'user_prompt': user_text,
                        'batch_data': batch # 保留原始 Batch 資訊，方便解析時對應
                    })
                    
        return tasks

    def _parse_batch_response(self, text: str, batch_size: int) -> List[str]:
        """[與你原版相同] 解析 LLM 回傳的 Item List"""
        clean_results = ["Parse_Error"] * batch_size
        if not text or "Error:" in text:
            return clean_results

        for i in range(1, batch_size + 1):
            # 你的原始 Regex 邏輯
            pattern = re.compile(rf"(?:Item|No\.?|^|\n)\s*\**{i}\**[^a-zA-Z0-9]*(Yes|No)", re.IGNORECASE)
            match = pattern.search(text)
            if match:
                clean_results[i-1] = match.group(1).title() # Yes/No
        return clean_results

    def _parse_and_convert_to_csv(self) -> str:
        """讀取暫存 CSV，套用 Regex 解析，展開成最終的 DataFrame 並存成 CSV"""
        logging.info("==== [Step 3.2] Parsing LLM Outputs & Building CSV ====")
        try:
            if not self.jsonl_output_path.exists():
                logging.error(f"❌ 找不到暫存結果檔案: {self.jsonl_output_path}")
                return ""

            # 🌟 修正 1：改用 read_csv 讀取我們剛剛存的暫存檔
            df_temp = pd.read_csv(str(self.jsonl_output_path), encoding='utf-8-sig')
            results = []

            # 遍歷每一筆完成的任務
            for _, task_result in df_temp.iterrows():
                model = task_result.get('model')
                prompt_id = task_result.get('prompt_id')
                # 確保 raw_output 是字串，避免 NaN 報錯
                raw_output = str(task_result.get('raw_output', '')) 
                
                # 🌟 修正 2：因為寫入 CSV 時 batch_data 變成了字串，讀出來要轉回字典
                batch_data_str = task_result.get('batch_data', '{}')
                import json # 確保有載入 json 模組
                try:
                    # 如果讀出來是字串就轉換，如果是字典就直接用
                    batch_data = json.loads(batch_data_str) if isinstance(batch_data_str, str) else batch_data_str
                except Exception as e:
                    logging.warning(f"解析 batch_data 失敗: {e}")
                    batch_data = {}

                batch_pairs = batch_data.get('batch_pairs', [])
                pmid = batch_data.get('pmid', '')

                # 呼叫 Regex 解析
                parsed_answers = self._parse_batch_response(raw_output, len(batch_pairs))

                # 將 Batch 展開為原本的一對一資料列 (Row)
                for j, pair_info in enumerate(batch_pairs):
                    ans = parsed_answers[j] if j < len(parsed_answers) else "Index_Error"
                    results.append({
                        "Data_ID": pair_info.get('orig_idx', ''),
                        "PMID": pmid,
                        "Model": model,
                        "Prompt_ID": prompt_id,
                        "E1": pair_info.get('E1_Name', ''),
                        "E2": pair_info.get('E2_Name', ''),
                        "True_Label": pair_info.get('True_Label', ''),
                        "Pred_Label": ans,
                        "Raw_Output": raw_output # 保留原始輸出方便 Debug
                    })

            # 轉為 DataFrame 並依照邏輯排序
            final_df = pd.DataFrame(results)
            final_df = final_df.sort_values(['Model', 'Prompt_ID', 'Data_ID'])
            
            csv_path_str = str(self.csv_output_path)
            final_df.to_csv(csv_path_str, index=False, encoding='utf-8-sig')
            
            logging.info(f"✅ 解析完成！格式化資料已儲存至: {csv_path_str}")
            return csv_path_str
            
        except Exception as e:
            logging.error(f"❌ 解析暫存檔轉最終 CSV 時發生錯誤: {e}")
            return ""