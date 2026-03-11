import yaml
from itertools import combinations, product
import os
import logging

class PromptManager:
    def __init__(self, prompts_yaml_path):
        self.yaml_path = prompts_yaml_path
        self.method_pool = self._load_yaml()
        self.generated_prompts = []
        print(f"📚 Prompt Library loaded from {self.yaml_path} ({len(self.method_pool)} items)")

    def _load_yaml(self):
        if not os.path.exists(self.yaml_path):
            raise FileNotFoundError(f"❌ File not found: {self.yaml_path}")
            
        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        # 結構為 { 'EMO': {1: 'text', 2: 'text'}, 'Role': {4: 'text'} }
        return data.get('prompts', {})

    def generate_combinations(self, config):
        """主入口：生成並排序"""
        mode = config.get('prompt_mode', 'auto').lower()
        self.generated_prompts = [] 

        print(f"🔧 Prompt Generation Mode: {mode.upper()}")

        if mode == 'auto':
            self._generate_auto_mode(config)
        elif mode == 'manual':
            self._generate_manual_mode(config)
        else:
            print(f"❌ Error: Unknown prompt mode '{mode}'")
            return []

        self._sort_results()

        print(f"✅ Generated {len(self.generated_prompts)} prompt combinations.")
        logging.info(f"✅ Generated {len(self.generated_prompts)} prompt combinations.")
        return self.generated_prompts

    def _sort_results(self):
        """
        [Private] 對生成的 Prompts 進行排序
        規則 1: 組合長度 (短的在前) -> 1, 2, 1+2
        規則 2: 數字大小 (小的在前) -> 1+2, 1+3
        """
        def sort_key(item):
            parts_str = item['id'].split(' + ')
            parts_int = [int(p) for p in parts_str if p.isdigit()]
            return (len(parts_int), parts_int)

        self.generated_prompts.sort(key=sort_key)
        
    def _generate_auto_mode(self, config):
        settings = config.get('auto_settings', {})

        target_methods = settings.get('methods', list(self.method_pool.keys()))
        
        # 2. 取得 max_size，若未指定，預設為類別的總數 (即全排列組合)
        max_size = settings.get('max_size', len(target_methods))
        
        # 確保 max_size 不會大於實際的類別數量 (避免出現要挑 4 個但只有 3 個類別的情況)
        limit = min(max_size, len(target_methods))
        
        logging.info(f"⚙️ Auto Mode: Combinations from 1 to {limit} methods.")

        # 3. 外層迴圈：依據 limit 決定堆疊幾層 (1層, 2層... 到 limit層)
        for r in range(1, limit + 1):
            
            # 從目標類別中，抽出 r 個類別的組合 (如 A+B, B+C)
            for method_combination in combinations(target_methods, r):
                
                # 準備這個類別組合下的所有具體 Prompt 選項
                prompt_list = []
                for cat in method_combination:
                    if cat in self.method_pool:
                        # 轉為 [(id, text), (id, text)...] 格式
                        prompt_items = [(str(k), v) for k, v in self.method_pool[cat].items()]
                        prompt_list.append(prompt_items)
                
                # 4. 內層迴圈：將選定類別內的選項進行笛卡兒積 (窮舉)
                for item_combo in product(*prompt_list):
                    ids = [item[0] for item in item_combo]
                    texts = [item[1] for item in item_combo]
                    
                    self._add_combination_from_parts(ids, texts)

    def _add_combination_from_parts(self, ids, texts):
        # 將組合的 ID 與文本合併加入清單
        self.generated_prompts.append({
            "id": " + ".join(ids),
            "text": "\n".join(texts) 
        })

    def _generate_manual_mode(self, config):
        manual_list = config.get('manual_keys', [])
        for i, combo_keys in enumerate(manual_list, 1):
            try:
                combo_keys = sorted([int(k) for k in combo_keys])
            except ValueError:
                continue

            valid_combo = [k for k in combo_keys if k in self.method_pool]
            if valid_combo:
                self._add_combination(valid_combo)
                
        
    def _add_combination(self, combo_keys):
        combo_id = " + ".join([str(k) for k in combo_keys])
        combined_text = "".join([self.method_pool[k] for k in combo_keys])
        
        self.generated_prompts.append({
            "id": combo_id,
            "text": combined_text
        })