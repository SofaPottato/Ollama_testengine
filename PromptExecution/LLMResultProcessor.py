import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict
from .schemas import PipelineError, LabelSet


class LLMResultProcessor:
    """
    將 OutputParser 的長表格清理後，由長表 pivot 成寬表，供下游 Evaluate 與人工檢視使用。
    result.csv (long) → partialInfo.csv (wide) + fullInfo.csv（含 rawOutput / sysPrompt）。
    """
    #後綴命名規則：所有衍生 runKey 欄位都帶後綴，方便後續過濾。
    _PRED_SUFFIX = '__pred'
    _RAW_SUFFIX = '__raw'
    _SYS_PROMPT_SUFFIX = '__sysPrompt'
    _RUN_KEY_SEPARATOR = '|'   # model 與 promptID 之間的分隔字元；'_' 會與模型名沖突，使用'|'分隔
    _REQUIRED_COLS = ('sentID', 'model', 'promptID', 'predLabel', 'trueLabel')
    # pivot 時「不能」當 index 的欄：model/promptID/runKey 是 column 維度，predLabel/rawOutput 是 value。
    _NON_INDEX_COLS = {'model', 'promptID', 'runKey', 'predLabel', 'rawOutput'}

    def __init__(self, parsedOutputCsvPath: Path, partialInfoCsvPath: Path, fullInfoCsvPath: Path,
                 promptList: List[Dict] = None, labelSet: LabelSet = None):
        self.parsedOutputCsvPath = Path(parsedOutputCsvPath) 
        self.partialInfoCsvPath = Path(partialInfoCsvPath)
        self.fullInfoCsvPath = Path(fullInfoCsvPath)
        self.promptList = promptList or []
        self.labelSet = labelSet or LabelSet()

        self.inputDf = None
        self.partialDf = None
        self.fullDf = None

    def run(self) -> Path:
        """讀取 → 標準化 → Pivot → 存檔，回傳精簡版寬表格路徑。"""
        logging.info(f"[Processor] 啟動: {self.parsedOutputCsvPath}")
        self._loadData()
        self._prepareDf()
        self._pivotLongToWide()
        return self._saveData()

    # ── 私有流程方法 ─────────────────────────────────────────────────────────

    def _loadData(self):
        """讀取並驗證輸入 CSV，缺必要欄位即拋 PipelineError。"""
        if not self.parsedOutputCsvPath.exists():
            raise PipelineError(f"File not found: {self.parsedOutputCsvPath}")
        try:
            self.inputDf = pd.read_csv(self.parsedOutputCsvPath, encoding='utf-8-sig')
        except Exception as e:
            raise PipelineError(f"Failed to read CSV: {e}") from e

        missingList = [c for c in self._REQUIRED_COLS if c not in self.inputDf.columns]
        if missingList:
            raise PipelineError(f"Missing required columns: {missingList}")

    def _prepareDf(self):
        """備份 originalLabel、標準化 trueLabel、組 runKey。"""
        # originalLabel 留原始字串（如 "positive"），方便人工檢視 fullInfo 時不必反查 classes。
        self.inputDf['originalLabel'] = self.inputDf['trueLabel']
        self.inputDf['trueLabel'] = self.inputDf['trueLabel'].apply(self.labelSet.labelToLabelCode)

        # runKey = model|promptID，作為 pivot 的 column 維度。
        # astype(str) 兼容 model/promptID被 pandas 推斷成 int 的情況；用 '|' 而非 '_' 是因為模型名常含 '_' 會混淆。
        self.inputDf['runKey'] = (self.inputDf['model'].astype(str) + self._RUN_KEY_SEPARATOR +
                                  self.inputDf['promptID'].astype(str))

    def _pivotLongToWide(self):
        """
        long → wide pivot：以樣本為列、runKey 為欄、predLabel 為值。
        index 欄動態偵測（排除 _NON_INDEX_COLS），讓上游新增資料欄不需要改本模組。
        同時產出 partialDf（predLabel）與 fullDf（predLabel + __raw 後綴欄）。
        所有 runKey 衍生欄位皆帶後綴：__pred / __raw / __sysPrompt，便於後續過濾。
        """
        indexCols = [c for c in self.inputDf.columns if c not in self._NON_INDEX_COLS]

        try:
            predWideDf = self.inputDf.pivot_table(
                index=indexCols,
                columns='runKey',
                values='predLabel',
                aggfunc='first'
            ).fillna(-1)  # 某樣本在某 runKey 沒資料 → NaN，補 -1（與「無法解析」共用 sentinel）。
            # 手動加 __pred 後綴：之後 fullDf 才能靠後綴把 pred / raw 兩種欄區分開。 
            predWideDf.columns = [f"{c}{self._PRED_SUFFIX}" for c in predWideDf.columns]
            self.partialDf = predWideDf.reset_index()

            # 第二次 pivot 取 rawOutput（給人工 review）：缺值補空字串。
            rawWideDf = self.inputDf.pivot_table(
                index=indexCols,
                columns='runKey',
                values='rawOutput',
                aggfunc='first'
            ).fillna('')
            rawWideDf.columns = [f"{c}{self._RAW_SUFFIX}" for c in rawWideDf.columns]


            self.fullDf = pd.concat([predWideDf, rawWideDf], axis=1).reset_index()
        except Exception as e:
            raise PipelineError(f"Pivot failed: {e}") from e

    def _saveData(self) -> Path:
        """寫精簡版與完整版，記錄統計後回傳精簡版路徑。"""
        try:
            self._savePartialInfoCsv()
            self._saveFullInfoCsv()
            self._logSummary()
            return self.partialInfoCsvPath
        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError(f"Failed to save results: {e}") from e

    def _savePartialInfoCsv(self) -> None:
        """精簡版：sentID + trueLabel + 各 runKey 的 __pred 欄。"""
        # partialInfo.csv 只保留 sentID、trueLabel、以及預測欄（即 runKey + __pred 後綴）。 
        predCols = [c for c in self.partialDf.columns if c.endswith(self._PRED_SUFFIX)]
        leanCols = [c for c in ('sentID', 'trueLabel') if c in self.partialDf.columns] + predCols
        self.partialDf[leanCols].to_csv(self.partialInfoCsvPath, index=False, encoding='utf-8-sig')

    def _saveFullInfoCsv(self) -> None:
        """完整版：補 {runKey}__sysPrompt 欄 → 欄位重排 → 寫檔。"""
        # 補 {runKey}__sysPrompt 欄（promptList 為空則跳過）。
        # 把每個 runKey 當次用的 system prompt 補進去，方便事後追溯。
        sysPromptCols: List[str] = []
        if self.promptList:
            promptIDToText = {p['promptID']: p['promptText'] for p in self.promptList}
            runKeyIter = (self.inputDf[['model', 'promptID', 'runKey']]
                          .drop_duplicates()
                          .itertuples(index=False))
            newColsDict: Dict[str, pd.Series] = {}
            for row in runKeyIter:
                colName = f"{row.runKey}{self._SYS_PROMPT_SUFFIX}"
                newColsDict[colName] = pd.Series(promptIDToText.get(row.promptID, ''), index=self.fullDf.index)
                if colName not in sysPromptCols:
                    sysPromptCols.append(colName)
            if newColsDict:
                self.fullDf = pd.concat([self.fullDf, pd.DataFrame(newColsDict)], axis=1)

        # 欄位重排：sentID → labels → __raw → __pred → __sysPrompt → 其他 index 欄。
        # 用後綴/白名單分類各欄再串成固定順序，讓人工開 fullInfo 檢視時版面一致好讀。
        predCols = [c for c in self.fullDf.columns if c.endswith(self._PRED_SUFFIX)]
        rawCols = [c for c in self.fullDf.columns if c.endswith(self._RAW_SUFFIX)]
        indexCols = [c for c in self.fullDf.columns if c not in predCols and c not in rawCols and c not in sysPromptCols]
        labelCols = [c for c in ('originalLabel', 'trueLabel') if c in indexCols]
        idCols = [c for c in ('sentID',) if c in indexCols]
        otherIndexCols = [c for c in indexCols if c not in labelCols and c not in idCols]
        orderedCols = idCols + labelCols + rawCols + predCols + sysPromptCols + otherIndexCols
        self.fullDf[orderedCols].to_csv(self.fullInfoCsvPath, index=False, encoding='utf-8-sig')

    def _logSummary(self) -> None:
        """輸出 parse rate 與 partial / full 兩張寬表的 shape 資訊。"""
        validCount = (self.inputDf['predLabel'] != -1).sum()
        totalCount = len(self.inputDf)
        logging.info(
            f"[Processor] 完成: partial={self.partialDf.shape}, full={self.fullDf.shape}, "
            f"parse rate={validCount}/{totalCount} ({validCount/totalCount:.1%}) "
            f"→ {self.partialInfoCsvPath}"
        )
