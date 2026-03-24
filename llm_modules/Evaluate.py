import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import logging 
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

class LLMEvaluationSystem:
    def __init__(self, inputCsvPath: str, outputBaseDir: str = "./output", indexCols: list = None):
        """
        初始化評估系統
        :param inputCsvPath: 包含預測結果的 CSV 路徑
        :param outputBaseDir: 輸出結果的根目錄
        :param indexCols: 識別資料的固定欄位清單 (解耦用)
        """
        self.input_csv_path = inputCsvPath
        self.df = pd.read_csv(self.input_csv_path)
        
        default_index_cols = ['Data_ID', 'PMID', 'E1', 'E2']
        self.index_cols = indexCols if indexCols else default_index_cols
        
        self.fixed_cols = self.index_cols + ['True_Label']
        
        self.pred_cols = [c for c in self.df.columns if c not in self.fixed_cols]
        self.y_true = self.df['True_Label']
        
        self.results_list = []
        self.report_df = None
        self.correctness_matrix = pd.DataFrame(index=self.df.index)
        self.hard_samples = None
        self.upper_bound = 0.0
        
        self.output_dir = outputBaseDir
        self.plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        logging.info(f"LLMEvaluationSystem(inputCsvPath='{self.input_csv_path}', df_shape={self.df.shape}, pred_cols_count={len(self.pred_cols)})")
        logging.info(f"🚀 System Initialized. Output directory: {self.output_dir}")

    def _calculate_single_metric(self, y_true_subset, y_pred_subset):
        """
        [Private Method] 計算單一模型的指標
        """
        if len(y_true_subset) == 0:
            return None
            
        metrics = {
            "Accuracy": accuracy_score(y_true_subset, y_pred_subset),
            "Precision": precision_score(y_true_subset, y_pred_subset, zero_division=0),
            "Recall": recall_score(y_true_subset, y_pred_subset, zero_division=0),
            "F1_Score": f1_score(y_true_subset, y_pred_subset, zero_division=0),
            "MCC": matthews_corrcoef(y_true_subset, y_pred_subset)
        }
        return {k: round(v, 2) for k, v in metrics.items()}
    
    def runEvaluation(self):
        """
        執行主要評估迴圈：遍歷所有模型欄位，計算指標並記錄對錯
        """
        for col in self.pred_cols:
            y_pred = self.df[col]
            
            valid_mask = y_pred.isin([0, 1])
            if valid_mask.sum() == 0:
                logging.warning(f"⚠️ Warning: Model '{col}' has no valid predictions (0 or 1). Skipping.")
                continue
                
            y_t_v = self.y_true[valid_mask]
            y_p_v = y_pred[valid_mask]

            metrics = self._calculate_single_metric(y_t_v, y_p_v)
            if metrics:
                res_dict = {"Model_Prompt_ID": col}
                res_dict.update(metrics)
                res_dict["Valid_Count"] = len(y_t_v)
                self.results_list.append(res_dict)
                
            is_correct = (y_pred == self.y_true).astype(int)
            self.correctness_matrix[col] = is_correct

        if self.results_list:
            self.report_df = pd.DataFrame(self.results_list)
            self.report_df = self.report_df.sort_values('F1_Score', ascending=False)
        else:
            logging.error("❌ No valid results generated.")
        
    def analyzeDifficulty(self):
        """
        計算難題 (Hard Samples) 與 理論上限 (Upper Bound)
        """
        if self.correctness_matrix.empty:
            logging.warning("Correctness matrix is empty. Skipping difficulty analysis.")
            return

        correct_counts = self.correctness_matrix.sum(axis=1)
        hard_indices = correct_counts[correct_counts == 0].index
        
        review_cols = self.index_cols + ['True_Label']
        available_review_cols = [c for c in review_cols if c in self.df.columns]
        self.hard_samples = self.df.loc[hard_indices, available_review_cols]

        total_samples = len(self.df)
        solvable_samples = total_samples - len(self.hard_samples)
        self.upper_bound = solvable_samples / total_samples if total_samples > 0 else 0
        
        logging.info(f"Difficulty Analysis Complete. Upper Bound: {self.upper_bound:.2%} (Found {len(self.hard_samples)} hard samples)")

    def plotConfusionMatrices(self):
        """
        為每個模型繪製混淆矩陣並存檔
        """
        logging.info("============Generating Confusion Matrices============")
        
        for col in self.pred_cols:
            y_pred = self.df[col]
            valid_mask = y_pred.isin([0, 1])
            if valid_mask.sum() == 0: continue
            
            y_t_v = self.y_true[valid_mask]
            y_p_v = y_pred[valid_mask]
            
            cm = confusion_matrix(y_t_v, y_p_v, labels=[0, 1])
            
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Pred: 0', 'Pred: 1'],
                        yticklabels=['True: 0', 'True: 1'])
            plt.title(f"Confusion Matrix: {col}")
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            
            safe_name = str(col).replace(":", "_").replace("+", "_").replace(" ", "_").replace("/", "_")
            save_path = os.path.join(self.plots_dir, f"CM_{safe_name}.png")

            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
    def plotHeatmap(self):
        """
        繪製模型對錯分佈熱圖 (Heatmap)
        """        
        if self.correctness_matrix.empty: 
            logging.warning("Correctness matrix is empty. Skipping heatmap plotting.")
            return
        
        logging.info("============Generating Correctness Heatmap============")
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.correctness_matrix.T, cmap="RdYlGn", cbar=True, cbar_kws={'label': 'Correct (1) / Incorrect (0)'})
        plt.title("Model Correctness Heatmap (Green=Correct)")
        plt.xlabel("Sample Index")
        plt.ylabel("Models")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "correctness_heatmap.png")

        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def saveResults(self):
        """
        輸出所有 CSV 報表
        """
        if self.report_df is not None:
            self.report_df.to_csv(os.path.join(self.output_dir, "eval_summary.csv"), index=False, encoding='utf-8-sig')
            
        if self.hard_samples is not None:
            self.hard_samples.to_csv(os.path.join(self.output_dir, "samples_to_review.csv"), index=False, encoding='utf-8-sig')
            
        logging.info(f"✅ All results saved to: {self.output_dir}")