import yaml
from itertools import combinations, product
import os
import logging

class PromptManager:

    def __init__(self, prompts_yaml_path):
        self.yaml_path = prompts_yaml_path
        self.method_pool = self._load_yaml()
        self.generated_prompts = []
        
        logging.info(f"PromptManager(prompts_yaml_path='{self.yaml_path}')")
        logging.info(f"📚 Prompt Library loaded from {self.yaml_path} ({len(self.method_pool)} items)")

    def _load_yaml(self):
        if not os.path.exists(self.yaml_path):
            logging.error(f"❌ File not found: {self.yaml_path}")
            raise FileNotFoundError(f"❌ File not found: {self.yaml_path}")
            
        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data.get('prompts', {})

    def generate_combinations(self, config):
        """生成並排序"""       
        mode = config.get('prompt_mode', 'auto').lower()
        self.generated_prompts = [] 

        logging.info(f"🔧 Prompt Generation Mode: {mode.upper()}")

        if mode == 'auto':
            self._generate_auto_mode(config)
        elif mode == 'manual':
            self._generate_manual_mode(config)
        else:
            logging.error(f"❌ Error: Unknown prompt mode '{mode}'")
            return []
        
        self._sort_results()
        
        logging.info(f"✅ Generated {len(self.generated_prompts)} prompt combinations.")
        return self.generated_prompts

    def _sort_results(self):
        """
        對生成的 Prompts 進行排序
        """
        def sort_key(item):
            parts_str = item['id'].split(' + ')
            parts_int = [int(p) for p in parts_str if p.isdigit()]#1,2,1+2,1+3
            return (len(parts_int), parts_int)

        self.generated_prompts.sort(key=sort_key)

    def _generate_auto_mode(self, config):
        """
        auto 設置下的prompt排序
        根據選擇類別與max_size對類別內prompt做窮舉
        """
        settings = config.get('auto_settings', {})

        target_methods = settings.get('methods', list(self.method_pool.keys()))
        max_size = settings.get('max_size', len(target_methods))
        limit = min(max_size, len(target_methods))
        
        logging.info(f"⚙️ Auto Mode: Combinations from 1 to {limit} methods.")
        logging.info(f"Target methods: {target_methods}")
        
        for r in range(1, limit + 1):
            for method_combination in combinations(target_methods, r):
                prompt_list = []
                for cat in method_combination:
                    if cat in self.method_pool:
                        prompt_items = [(str(k), v) for k, v in self.method_pool[cat].items()]
                        prompt_list.append(prompt_items)
            
                for item_combo in product(*prompt_list):
                    ids = [item[0] for item in item_combo]
                    texts = [item[1] for item in item_combo]
                    
                    self._add_combination_from_parts(ids, texts)
            
    def _generate_manual_mode(self, config):
        """
        鋪平所有類別，依照選擇數字加入prompt list
        """
        logging.info("Initializing _generate_manual_mode()")
        manual_list = config.get('manual_keys', [])
        
        logging.info(f"Manual keys provided: {manual_list}")
        flat_pool = {}
        for cat, items in self.method_pool.items():
            if not isinstance(items, dict):
                continue
            for k, v in items.items():
                flat_pool[str(k)] = v
        for i, combo_keys in enumerate(manual_list, 1):
            try:
                combo_keys = sorted([int(k) for k in combo_keys])
            except ValueError:
                continue
            ids = []
            texts = []
            for k in combo_keys:
                k_str = str(k)
                if k_str in flat_pool:
                    ids.append(k_str)
                    texts.append(flat_pool[k_str])

            if ids:
                self._add_combination_from_parts(ids, texts)
    def _add_combination_from_parts(self, ids, texts):
        self.generated_prompts.append({
            "id": " + ".join(ids),
            "text": "\n".join(texts) 
        })              
