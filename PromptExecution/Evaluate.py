import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import logging
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from pathvalidate import sanitize_filename
from .schemas import LabelSet


class PromptCmbEval:
    """
    各 (model, promptID) 組合的分類效能評估器（支援二元與多分類）。
    partialInfo.csv → evalSummary.csv + cm 圖 + 熱圖 + samplesToReview.csv。
    有效標籤為 classes 索引 0..N-1，-1（無法解析）一律排除於指標、計為答錯於難題分析。
    """

    _PRED_SUFFIX = '__pred'

    def __init__(self, partialInfoCsvPath: Path, outputDirPath: Path = Path("./output"),
                 labelSet: LabelSet = None):

        self.partialInfoCsvPath = Path(partialInfoCsvPath)
        self.outputDirPath = Path(outputDirPath)
        self.plotsDirPath = self.outputDirPath / "plots"

        # _validLabels = [0..N-1]，即「有效預測」的值域；用來把 -1（無法解析）從指標中濾掉。
        cfg = labelSet or LabelSet()
        self.classes = cfg.classes
        self._validLabels = list(range(len(self.classes)))

        self.inputDf = None
        self.predCols = []
        self.idCols = []
        self.yTrueLabelSeries = None

        self.metricsResultsList = []
        self.metricsSummaryDf = None
        self.correctnessMatrixDf = None
        self.hardSamplesDf = None
        self.upperBound = 0.0

    def run(self) -> Path:
        """評估入口：讀檔 → 計算指標 → 難題分析 → 繪圖 → 存檔。"""
        self._loadData()
        self._evalAllPredCols()
        self._analyzeUpperBound()
        self._plotConfusionMatrices()
        self._plotHeatmap()
        self._saveResults()
        return self.outputDirPath

    # ── 私有流程方法 ─────────────────────────────────────────────────────────

    def _loadData(self):
        if not self.partialInfoCsvPath.exists():
            raise FileNotFoundError(f"Eval input CSV not found: {self.partialInfoCsvPath}")

        self.inputDf = pd.read_csv(str(self.partialInfoCsvPath), encoding='utf-8-sig')

        self.idCols = ['sentID']
        self.predCols = [col for col in self.inputDf.columns if col not in ('trueLabel', 'sentID')]

        self.yTrueLabelSeries = self.inputDf['trueLabel']
        self.correctnessMatrixDf = pd.DataFrame(index=self.inputDf.index)

        self.plotsDirPath.mkdir(parents=True, exist_ok=True)

        logging.info(
            f"[Eval] 載入完成: shape={self.inputDf.shape}, "
            f"預測欄={len(self.predCols)}, index欄={self.idCols} → {self.outputDirPath}"
        )

    def _evalAllPredCols(self):
        """
        遍歷所有預測欄，計算指標並記錄每個樣本的對錯矩陣。
        指標計算只用有效預測（predLabel ∈ classes 索引 0..N-1）；對錯矩陣用全體樣本（含 -1，-1 一律判錯）。
        """
        # 二元任務的 P/R/F1 固定以 labelCode 1（classes[1]）為正類（sklearn average='binary'）。
        # 明示正類，labelSet 順序若被手誤翻轉（如 ['yes','no']），這行 log 會顯示正類變成 'no'，肉眼可擋。
        if len(self.classes) == 2:
            logging.info(f"[Eval] 正類 = '{self.classes[1]}'；負類 = '{self.classes[0]}' ")

        correctnessByCol = {}
        for predColName in self.predCols:
            # 指標只用有效預測：整欄都 -1（該 runKey 全解析失敗）→ getValidLabelSeries 回 None，跳過指標。
            validLabelsTuple = self._getValidLabelSeries(predColName)
            if validLabelsTuple is None:
                logging.warning(f"[Eval] 跳過 {predColName}: 無有效預測 (predLabel ∉ classes 索引 0..N-1)")
                continue
            yTrueValidSeries, yPredValidSeries = validLabelsTuple

            evalMetricsDict = self._calcMetrics(yTrueValidSeries, yPredValidSeries)
            if evalMetricsDict:
                resultRowDict = {"modelPromptID": predColName.removesuffix(self._PRED_SUFFIX)}
                resultRowDict.update(evalMetricsDict)
                resultRowDict["validCount"] = len(yTrueValidSeries)
                self.metricsResultsList.append(resultRowDict)

            correctnessByCol[predColName] = (self.inputDf[predColName] == self.yTrueLabelSeries).astype(int)

        if correctnessByCol:
            self.correctnessMatrixDf = pd.DataFrame(correctnessByCol, index=self.inputDf.index)

        if self.metricsResultsList:
            self.metricsSummaryDf = pd.DataFrame(self.metricsResultsList).sort_values('f1Score', ascending=False)
        else:
            logging.warning("[Eval] 無有效結果，未產生 eval_summary.csv")

    def _calcMetrics(self, yTrueSeries, yPredSeries) -> dict:
        """
        計算單一 runKey 的分類指標（Accuracy / Precision / Recall / F1 / MCC）。
        二元 → 正類為 classes 索引 1；多分類 → macro 平均（各類別等權）。
        MCC 原生支援多分類；zero_division=0 讓無正類預測時回傳 0 而非報錯。
        yTrue 為空時回傳 None。
        """
        if len(yTrueSeries) == 0:
            return None
        avg = 'binary' if len(self.classes) == 2 else 'macro'
        metricsDict = {
            "accuracy":  accuracy_score(yTrueSeries, yPredSeries),
            "precision": precision_score(yTrueSeries, yPredSeries, average=avg, zero_division=0),
            "recall":    recall_score(yTrueSeries, yPredSeries, average=avg, zero_division=0),
            "f1Score":   f1_score(yTrueSeries, yPredSeries, average=avg, zero_division=0),
            "mcc":       matthews_corrcoef(yTrueSeries, yPredSeries)
        }
        return {k: round(v, 2) for k, v in metricsDict.items()}

    def _analyzeUpperBound(self):
        """
        計算難題（所有 runKey 都答錯的樣本）與理論上限。
        Upper Bound = (總樣本 - 難題) / 總樣本，反映「完美解非難題」時的天花板準確率。
        Upper Bound 遠低於目標時，加 prompt 試誤無效，需從資料/模型本身改進。
        """
        if self.correctnessMatrixDf.empty:
            logging.warning("[Eval] correctness matrix 為空，跳過難題分析")
            return

        # 難題 = 對錯矩陣某列「橫向加總 == 0」的樣本，即沒有任何 runKey 答對它。
        correctCountsSeries = self.correctnessMatrixDf.sum(axis=1)
        hardSampleIndexList = correctCountsSeries[correctCountsSeries == 0].index

        # 難題清單只留 sentID + trueLabel，給人工覆檢用（available 過濾避免欄位不存在時 KeyError）。
        reviewCols = self.idCols + ['trueLabel']
        availableReviewCols = [c for c in reviewCols if c in self.inputDf.columns]
        self.hardSamplesDf = self.inputDf.loc[hardSampleIndexList, availableReviewCols]

        # Upper Bound = 非難題比例：代表準確率天花板。
        totalSampleCount = len(self.inputDf)
        solvableSampleCount = totalSampleCount - len(self.hardSamplesDf)
        self.upperBound = solvableSampleCount / totalSampleCount if totalSampleCount > 0 else 0

        logging.info(f"[Eval] 難題分析完成: Upper Bound={self.upperBound:.2%}, 難題 {len(self.hardSamplesDf)} 筆")

    def _plotConfusionMatrices(self):
        """為每個 runKey 繪製混淆矩陣 PNG（排除 -1），存至 plots/ 目錄。"""
        logging.info("[Eval] 繪製混淆矩陣中")

        for predColName in self.predCols:
            validLabelsTuple = self._getValidLabelSeries(predColName)
            if validLabelsTuple is None:
                continue
            yTrueValidSeries, yPredValidSeries = validLabelsTuple
            displayName = predColName.removesuffix(self._PRED_SUFFIX)

            # labels=_validLabels 確保即使某類別無預測，矩陣仍為 N×N
            confusionMatrixArr = confusion_matrix(yTrueValidSeries, yPredValidSeries, labels=self._validLabels)

            # 圖大小隨類別數縮放，避免多分類時格子擠成一團
            plt.figure(figsize=(max(6, len(self.classes) * 2), max(5, len(self.classes) * 1.6)))
            sns.heatmap(confusionMatrixArr, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=[f'Pred: {c}' for c in self.classes],
                        yticklabels=[f'True: {c}' for c in self.classes])
            plt.title(f"Confusion Matrix: {displayName}")
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()

            savePath = self.plotsDirPath / f"CM{sanitize_filename(str(displayName), replacement_text='_').replace('+', '_').replace(' ', '_')}.png"
            plt.savefig(str(savePath), bbox_inches='tight')
            plt.close()

    def _plotHeatmap(self):
        """繪製所有 runKey 對每個樣本的對錯熱圖（綠=對、紅=錯），存至 outputDir。"""
        if self.correctnessMatrixDf.empty:
            logging.warning("[Eval] correctness matrix 為空，跳過熱圖")
            return

        logging.info("[Eval] 繪製對錯熱圖中")
        plt.figure(figsize=(12, 8))

        displayDf = self.correctnessMatrixDf.rename(columns=lambda c: c.removesuffix(self._PRED_SUFFIX))
        sns.heatmap(displayDf.T, cmap=ListedColormap(["#d73027", "#1a9850"]),
                    vmin=0, vmax=1, cbar=False)
        plt.title("Model Correctness Heatmap (Green=Correct)")
        plt.xlabel("Sample Index")
        plt.ylabel("Models")
        plt.tight_layout()
        savePath = self.outputDirPath / "correctnessHeatmap.png"
        plt.savefig(str(savePath), bbox_inches='tight')
        plt.close()

    def _saveResults(self):
        """輸出 evalSummary.csv（按 F1 排序）與 samplesToReview.csv（難題清單）。"""
        if self.metricsSummaryDf is not None:
            # 在 summary 末尾附一列 upperBound：把「天花板」與各組合指標放同一張表方便對照。
            summaryDf = self.metricsSummaryDf.copy()
            upperBoundRow = {col: "" for col in summaryDf.columns}
            upperBoundRow["modelPromptID"] = f"upperBound: {self.upperBound:.2%}"
            summaryDf = pd.concat([summaryDf, pd.DataFrame([upperBoundRow])], ignore_index=True)
            summaryDf.to_csv(str(self.outputDirPath / "evalSummary.csv"), index=False, encoding='utf-8-sig')

        if self.hardSamplesDf is not None:
            self.hardSamplesDf.to_csv(str(self.outputDirPath / "samplesToReview.csv"), index=False, encoding='utf-8-sig')

        logging.info(f"[Eval] 所有結果已儲存 → {self.outputDirPath}")

    # ── 工具方法 ──────────────────────────────────────────────────────────────

    def _getValidLabelSeries(self, col: str):
        """回傳 (yTrueValidSeries, yPredValidSeries)，若無有效預測（值不在 classes 索引）則 None。"""
        # 用 isin(_validLabels) 做遮罩濾掉 -1：同時套用到 true 與 pred，確保兩邊對齊同一批樣本。
        # 整欄都無效（全 -1）→ 回 None，呼叫端據此跳過該 runKey 的指標/繪圖。
        yPredSeries = self.inputDf[col]
        validMaskSeries = yPredSeries.isin(self._validLabels)
        if validMaskSeries.sum() == 0:
            return None
        return self.yTrueLabelSeries[validMaskSeries], yPredSeries[validMaskSeries]
