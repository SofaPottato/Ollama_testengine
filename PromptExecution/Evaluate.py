"""分類效能評估：對每個 runKey 算指標、找難題、畫混淆矩陣與對錯熱圖。"""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score)

from .schemas import LabelSet, PRED_SUFFIX, safeFileStem


class PromptCmbEval:
    """各 (model, promptCmbID) 組合（runKey）的分類效能評估器（支援二元與多分類）。

    輸入 mlTableDf（sentID + trueLabel + 各 {runKey}__pred），產出四種檔到 evalDir：
      - evalSummary.csv：每個 runKey 一列（accuracy/precision/recall/f1Score/mcc/validCount），
        按 f1Score 由高到低排序，末尾附三列 upperBound（全部 / F1 前 10 / 前 20）。
      - FalsePredictionByAllPromptCmb.csv：所有 runKey 都答錯的難題（sentID, trueLabel），供人工覆檢。
      - correctnessHeatmap.png：全 runKey × 樣本的對錯熱圖（綠=對、紅=錯）。
      - plots/CM<runKey>.png ：每個 runKey 的混淆矩陣。
    有效標籤為 classes 索引 0..N-1；-1（無法解析）一律排除於指標、並於難題分析計為答錯。

    evalDir（產物輸出目錄）建構時定下；資料輸入（mlTableDf / labelSet）與各產物路徑於用到的方法呼叫時傳入。
    跨步驟共用的「計算結果」於 loadMLTableDf 一次初始化並存於 self，plot/save 方法直接取用。

    對外有六個步驟，由 Main 照順序接起來：
      1. loadMLTableDf        ：吃 Step 6 的寬表，備妥評估狀態。
      2. evalPromptCmb        ：每個 runKey 算指標 + 對錯矩陣。
      3. analyzeUpperBound    ：難題清單與準確率天花板（全部 / F1 前 10 / 前 20）。
      4. plotConfusionMatrices：每個 runKey 一張混淆矩陣 PNG。
      5. plotHeatmap          ：全 runKey × 樣本的對錯熱圖。
      6. saveSummary          ：evalSummary.csv + 難題清單 CSV。
    """

    def __init__(self, evalDir: Path):
        # evalDir 是「這個評估器往哪寫」的固定屬性（root）；底下各產物的檔名於各 plot/save 方法呼叫時再給。
        self.evalDir = Path(evalDir)

    #============================================================================#
    #   Step 1: mlTableDf → 評估狀態（predCols / trueLabel / 快取容器）#
    #============================================================================#
    def loadMLTableDf(self, mlTableDf: pd.DataFrame):
        """吃 Step 6 的寬表，備妥後續評估要用的欄位與狀態。"""
        self.inputDf = mlTableDf.copy()

        self.idCols = ['sentID']
        # 預測欄一律以 __pred 後綴辨識：寬表帶 originalAns / passage 等 index 欄時也不會被誤當預測欄。
        self.predCols = [col for col in self.inputDf.columns if col.endswith(PRED_SUFFIX)]

        self.yTrueLabelSeries = self.inputDf['trueLabel']
        self.correctnessMatrixDf = pd.DataFrame(index=self.inputDf.index)

        # 評估過程累積、跨步驟共用的結果狀態，於載入時一次初始化。
        # validLabelSeriesByCol：每個 predCol 的有效標籤 (yTrue, yPred)；evalPromptCmb 算好後快取，
        #   plotConfusionMatrices 重用，避免同一份過濾算兩遍。
        self.validLabelSeriesByCol = {}
        self.metricsResultsList = []
        self.metricsSummaryDf = None
        self.hardSamplesDf = None
        self.upperBound = 0.0        # 用上全部 runKey 的天花板
        self.upperBoundTop10 = 0.0   # 只取 F1 前 10 名 runKey 的天花板
        self.upperBoundTop20 = 0.0   # 只取 F1 前 20 名 runKey 的天花板

        logging.info(
            f"[Eval] 載入完成: shape={self.inputDf.shape}, "
            f"預測欄={len(self.predCols)}, index欄={self.idCols} → {self.evalDir}"
        )

    #============================================================================#
    #   Step 2: 每個 runKey 算指標（Acc/P/R/F1/MCC）+ 對錯矩陣#
    #============================================================================#
    def evalPromptCmb(self, labelSet: LabelSet):
        """遍歷所有預測欄算指標並記錄每個樣本的對錯矩陣。

        指標只用有效預測（predLabel ∈ 0..N-1）；對錯矩陣用全體樣本（含 -1，-1 一律判錯）。
        """
        # validLabels = [0..N-1]，即「有效預測」的值域；用來把 -1（無法解析）從指標中濾掉。
        classes = labelSet.classes
        validLabels = list(range(len(classes)))

        # 二元任務的 P/R/F1 固定以 labelCode 1（classes[1]）為正類。明示正類：若 labelSet 順序被手誤
        # 翻轉（如 ['yes','no']），這行 log 會顯示正類變成 'no'，肉眼可擋。
        if len(classes) == 2:
            logging.info(f"[Eval] 正類 = '{classes[1]}'；負類 = '{classes[0]}' ")

        # 二元 → 正類為 classes 索引 1（average='binary'）；多分類 → macro 平均（各類別等權）。
        avg = 'binary' if len(classes) == 2 else 'macro'

        correctnessByCol = {}
        for predColName in self.predCols:
            # 1) 有效預測遮罩：isin(validLabels) 濾掉 -1，同時套用到 true 與 pred，確保兩邊對齊同一批樣本。
            yPredSeries = self.inputDf[predColName]
            validMaskSeries = yPredSeries.isin(validLabels)
            # 整欄都無效（全 -1，該 runKey 全解析失敗）→ 跳過該 runKey 的指標。
            if validMaskSeries.sum() == 0:
                logging.warning(f"[Eval] 跳過 {predColName}: 無有效預測 (predLabel ∉ classes 索引 0..N-1)")
                continue
            yTrueValidSeries = self.yTrueLabelSeries[validMaskSeries]
            yPredValidSeries = yPredSeries[validMaskSeries]
            self.validLabelSeriesByCol[predColName] = (yTrueValidSeries, yPredValidSeries)  # 快取供混淆矩陣重用

            # 2) 算五項指標；zero_division=0 讓無正類預測時回 0 而非報錯，MCC 原生支援多分類。
            #    欄序固定：modelPromptCmbID → 五項指標 → validCount，與 evalSummary.csv 一致。
            resultRowDict = {
                "modelPromptCmbID": predColName.removesuffix(PRED_SUFFIX),
                "accuracy":  round(accuracy_score(yTrueValidSeries, yPredValidSeries), 2),
                "precision": round(precision_score(yTrueValidSeries, yPredValidSeries, average=avg, zero_division=0), 2),
                "recall":    round(recall_score(yTrueValidSeries, yPredValidSeries, average=avg, zero_division=0), 2),
                "f1Score":   round(f1_score(yTrueValidSeries, yPredValidSeries, average=avg, zero_division=0), 2),
                "mcc":       round(matthews_corrcoef(yTrueValidSeries, yPredValidSeries), 2),
                "validCount": len(yTrueValidSeries),
            }
            self.metricsResultsList.append(resultRowDict)

            # 3) 記對錯欄：用全體樣本（含 -1；-1 必不等於 trueLabel，一律判錯）。
            correctnessByCol[predColName] = (self.inputDf[predColName] == self.yTrueLabelSeries).astype(int)

        if correctnessByCol:
            self.correctnessMatrixDf = pd.DataFrame(correctnessByCol, index=self.inputDf.index)

        if self.metricsResultsList:
            self.metricsSummaryDf = pd.DataFrame(self.metricsResultsList).sort_values('f1Score', ascending=False)
        else:
            logging.warning("[Eval] 無有效結果，未產生 evalSummary.csv")

    #============================================================================#
    #   Step 3: 難題清單 + 準確率天花板（全部 / F1 前 10 / 前 20）#
    #============================================================================#
    def analyzeUpperBound(self):
        """計算難題（所有 runKey 都答錯的樣本）與理論上限。

        Upper Bound = (總樣本 - 難題) / 總樣本，反映「完美挑選組合」時的準確率天花板。
        另計只取 F1 前 10 / 前 20 名 runKey 的 Upper Bound：實務不會用上全部組合，看「最好的少數組合」
        的天花板更貼近實際 ensemble 規模。Upper Bound 遠低於目標時，加 prompt 試誤無效，需從資料/模型本身改進。
        """
        if self.correctnessMatrixDf.empty:
            logging.warning("[Eval] correctness matrix 為空，跳過難題分析")
            return

        # 難題 = 對錯矩陣某列「橫向加總 == 0」的樣本，即沒有任何 runKey 答對它。
        correctCountsSeries = self.correctnessMatrixDf.sum(axis=1)
        hardSampleIndexList = correctCountsSeries[correctCountsSeries == 0].index

        # 難題清單只留 sentID + trueLabel 給人工覆檢（available 過濾避免欄位不存在時 KeyError）。
        reviewCols = self.idCols + ['trueLabel']
        availableReviewCols = [c for c in reviewCols if c in self.inputDf.columns]
        self.hardSamplesDf = self.inputDf.loc[hardSampleIndexList, availableReviewCols]

        # 全體 Upper Bound = 非難題比例（用上所有 runKey 時的準確率天花板）。
        totalSampleCount = len(self.inputDf)
        solvableSampleCount = totalSampleCount - len(self.hardSamplesDf)
        self.upperBound = solvableSampleCount / totalSampleCount

        # 只取 F1 前 N 名 runKey 的子矩陣各自再算一次 Upper Bound（N=10, 20）。metricsSummaryDf 已按 f1Score
        # 由高到低排序；runKey 不足 N 個時 head(n) 自動取全部。
        # （能走到這裡代表已過 correctnessMatrixDf.empty 檢查，totalSampleCount>0、metricsSummaryDf 非 None 皆成立。）
        topNUpperBoundDict = {}
        for n in (10, 20):
            # modelPromptCmbID 是去掉 __pred 後綴的名稱；補回後綴才對得上 correctnessMatrixDf 的欄。
            topNameList = self.metricsSummaryDf['modelPromptCmbID'].head(n).tolist()
            topColList = [f"{name}{PRED_SUFFIX}" for name in topNameList]
            topColList = [c for c in topColList if c in self.correctnessMatrixDf.columns]
            if not topColList:
                topNUpperBoundDict[n] = (0.0, totalSampleCount)
                continue
            topCorrectCountsSeries = self.correctnessMatrixDf[topColList].sum(axis=1)
            topHardCount = int((topCorrectCountsSeries == 0).sum())
            topNUpperBoundDict[n] = ((totalSampleCount - topHardCount) / totalSampleCount, topHardCount)
        self.upperBoundTop10, hardTop10 = topNUpperBoundDict[10]
        self.upperBoundTop20, hardTop20 = topNUpperBoundDict[20]

        logging.info(
            f"[Eval] 難題分析完成: Upper Bound(全部)={self.upperBound:.2%}（難題 {len(self.hardSamplesDf)} 筆）, "
            f"Top10={self.upperBoundTop10:.2%}（難題 {hardTop10} 筆）, "
            f"Top20={self.upperBoundTop20:.2%}（難題 {hardTop20} 筆）"
        )

    #============================================================================#
    #   Step 4: 每個 runKey 一張混淆矩陣 PNG#
    #============================================================================#
    def plotConfusionMatrices(self, labelSet: LabelSet, plotsDir: Optional[Path] = None):
        """為每個 runKey 繪製混淆矩陣 PNG（排除 -1），存至 plotsDir（預設 evalDir/plots）。"""
        logging.info("[Eval] 繪製混淆矩陣中")

        classes = labelSet.classes
        validLabels = list(range(len(classes)))
        plotsDirPath = Path(plotsDir) if plotsDir is not None else self.evalDir / "plots"
        plotsDirPath.mkdir(parents=True, exist_ok=True)

        # 直接用 evalPromptCmb 快取好的有效標籤；沒入快取（該欄全無效被跳過）的就略過。
        for predColName in self.predCols:
            validLabelsTuple = self.validLabelSeriesByCol.get(predColName)
            if validLabelsTuple is None:
                continue
            yTrueValidSeries, yPredValidSeries = validLabelsTuple
            displayName = predColName.removesuffix(PRED_SUFFIX)

            # labels=validLabels 確保即使某類別無預測，矩陣仍為 N×N。
            confusionMatrixArr = confusion_matrix(yTrueValidSeries, yPredValidSeries, labels=validLabels)

            # 圖大小隨類別數縮放，避免多分類時格子擠成一團。
            plt.figure(figsize=(max(6, len(classes) * 2), max(5, len(classes) * 1.6)))
            sns.heatmap(confusionMatrixArr, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=[f'Pred: {c}' for c in classes],
                        yticklabels=[f'True: {c}' for c in classes])
            plt.title(f"Confusion Matrix: {displayName}")
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()

            savePath = plotsDirPath / f"CM{safeFileStem(displayName)}.png"
            plt.savefig(str(savePath), bbox_inches='tight')
            plt.close()

    #============================================================================#
    #   Step 5: 全 runKey × 樣本的對錯熱圖#
    #============================================================================#
    def plotHeatmap(self, heatmapPath: Optional[Path] = None):
        """繪製所有 runKey 對每個樣本的對錯熱圖（綠=對、紅=錯），存至 heatmapPath（預設 evalDir/correctnessHeatmap.png）。"""
        if self.correctnessMatrixDf.empty:
            logging.warning("[Eval] correctness matrix 為空，跳過熱圖")
            return

        logging.info("[Eval] 繪製對錯熱圖中")
        plt.figure(figsize=(12, 8))

        displayDf = self.correctnessMatrixDf.rename(columns=lambda c: c.removesuffix(PRED_SUFFIX))
        sns.heatmap(displayDf.T, cmap=ListedColormap(["#d73027", "#1a9850"]),
                    vmin=0, vmax=1, cbar=False)
        plt.title("Model Correctness Heatmap (Green=Correct)")
        plt.xlabel("Sample Index")
        plt.ylabel("Models")
        plt.tight_layout()
        savePath = Path(heatmapPath) if heatmapPath is not None else self.evalDir / "correctnessHeatmap.png"
        plt.savefig(str(savePath), bbox_inches='tight')
        plt.close()

    #============================================================================#
    #   Step 6: evalSummary.csv + 難題清單 CSV#
    #============================================================================#
    def saveSummary(self, summaryPath: Optional[Path] = None,
                    reviewPath: Optional[Path] = None):
        """輸出指標總表（按 F1 排序）與難題清單；預設 evalDir/evalSummary.csv、evalDir/FalsePredictionByAllPromptCmb.csv。"""
        summaryPath = Path(summaryPath) if summaryPath is not None else self.evalDir / "evalSummary.csv"
        reviewPath = Path(reviewPath) if reviewPath is not None else self.evalDir / "FalsePredictionByAllPromptCmb.csv"
        if self.metricsSummaryDf is not None:
            # summary 末尾附三列 upperBound（全部 / F1 前 10 / F1 前 20）：把「天花板」與各組合指標放同一張表方便對照。
            summaryDf = self.metricsSummaryDf.copy()
            upperBoundRowList = [
                ("accupperBound (all)",       self.upperBound),
                ("accupperBound (top10 f1)",  self.upperBoundTop10),
                ("accupperBound (top20 f1)",  self.upperBoundTop20),
            ]
            # 先把三列組好、最後一次 concat（不在迴圈內反覆 concat 重建 df）。
            appendRowList = []
            for label, value in upperBoundRowList:
                row = {col: "" for col in summaryDf.columns}
                row["modelPromptCmbID"] = f"{label}: {value:.2%}"
                appendRowList.append(row)
            summaryDf = pd.concat([summaryDf, pd.DataFrame(appendRowList)], ignore_index=True)
            summaryDf.to_csv(str(summaryPath), index=False, encoding='utf-8-sig')

        if self.hardSamplesDf is not None:
            self.hardSamplesDf.to_csv(str(reviewPath), index=False, encoding='utf-8-sig')
        logging.info(f"[Eval] 所有結果已儲存 → {self.evalDir}")
