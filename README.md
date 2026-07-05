# EnsemblePrompt

以**本地 LLM（Ollama）**在生物醫學關係抽取任務上，系統性地**窮舉組合多種 Prompt 技巧（prompt technique）並評估其效能**的實驗流水線。

核心問題：對同一個分類任務，把不同的 Prompt 技巧（情緒激勵、角色扮演、Few-shot…）**排列組合**成不同的 system prompt，哪一種組合表現最好？把它們當成一個 ensemble，準確率的天花板（upper bound）又在哪裡？EnsemblePrompt 把「生成組合 → 推論 → 解析 → 評估」整條流程自動化，一次跑完 train 與 test。

目前內建兩種任務型別：

- **PPI**（Protein-Protein Interaction）：二元分類，判斷句子中 `PROTEIN1` 與 `PROTEIN2` 是否有交互作用（`yes` / `no`）。
- **BC5CDR**（Chemical-Disease）：一篇文章含多個 entity pair，批次判斷（batch inference）。

---

## 運作原理

一個 **Prompt 組合（prompt combination）** 是從「方法池」中各分類挑選項目後串接而成的 system prompt。方法池是一份 YAML，把技巧分成幾個**分類（method）**，每個分類底下有數個**編號項目**：

```yaml
prompts:
  EMO:              # 情緒激勵
    01: |
      This is very important to my career
    02: |
      If you misclassify an interaction here, it could corrupt the interactome knowledge base...
  Role:             # 角色扮演
    01: |
      You are an expert Biocurator for a major protein-protein interaction (PPI) knowledge graph
  Few_shot:         # 少樣本示範
    01: |
      Example 1: ...
```

- 每個項目有唯一 ID：`分類名 + 兩位數編號`，例如 `EMO01`、`Role02`、`Few_shot01`。
- **窮舉模式（Auto）**：對選定的分類做 `1..maxCmbNum` 層組合，再對各分類的項目做笛卡兒積，枚舉出所有可能組合。
- **手動模式（Manual）**：只生成你明確列出的組合，例如 `[["EMO01", "Role01"]]`。

每一組組合有一個識別碼 **`promptCmbID`**（如 `EMO01 + Role01`），其 `promptText` 就是實際送給模型的 system prompt。一次「模型 × Prompt 組合」的完整識別碼稱為 **`runKey`**（如 `llama3.2:1b|EMO01 + Role01`），它是下游所有寬表欄名與評估報告的維度識別碼。

---

## Pipeline（7 個步驟）

整條流程由 [Main_EnsemblePrompt.py](Main_EnsemblePrompt.py) 串接。Step 1 產生的 Prompt 組合由 train / test **共用同一份**，Step 2~7 對每個 split 各跑一次。

| 步驟 | 模組 | 輸入 → 輸出 |
|------|------|-------------|
| 1. 生成 Prompt 組合 | [`PromptCmbGen`](PromptExecution/PromptCmbGen.py) | 方法池 YAML → `ppiPromptCmb.csv`（`promptCmbID`, `promptText`） |
| 2. 載入資料與 checkpoint | [`DataLoader`](PromptExecution/DataLoader.py) | Dataset CSV → `(datasetDf, labelSet)`；`response.csv` → 已完成任務集合 |
| 3. 建構待執行任務 | [`TaskBuilder`](PromptExecution/TaskBuilder.py) | `datasetDf × models × prompts` → `promptInfoDf`（跳過已完成的）|
| 4. LLM 推論 | [`OllamaEngine` (`LLMEngine`)](PromptExecution/OllamaEngine.py) | 非同步呼叫 Ollama，逐筆 append 到 `response.csv` |
| 5. 解析輸出 | [`ResponseParser`](PromptExecution/ResponseParser.py) | `response.csv` → 長表 `result.csv`（一 item 一列）|
| 6. 後處理 | [`LLMResultProcessor`](PromptExecution/LLMResultProcessor.py) | 長表 → 寬表 `mlTable.csv` / `fullResultInfo.csv`（pivot）|
| 7. 評估 | [`Evaluate` (`PromptCmbEval`)](PromptExecution/Evaluate.py) | `mlTable` → 指標總表、混淆矩陣、對錯熱圖、難題清單 |

型別、schema、跨模組共用常數與自訂例外集中在 [`schemas.py`](PromptExecution/schemas.py)；logger 與隨機種子的初始化在 [`ExperimentInitializer`](PromptExecution/ExperimentInitializer.py)。

### 幾個設計重點

- **斷點續跑（checkpoint / resume）**：`response.csv` 同時是輸出檔與 checkpoint。每筆推論完成即 `append` 並 `fsync` 落盤；重跑時 Step 2 會讀回已完成的 `(model, promptCmbID, taskID)` 三元組，Step 3 據此跳過，中斷後可無痛續跑。**若要重新跑一輪，刪除或備份 `response.csv` 即可。**
- **結構化輸出（structured output）**：透過 Ollama 的 `format` JSON schema（由 `LabelSet.buildOllamaOutputFormat` 產生）強制模型只能輸出 `classes` 之一，大幅減少解析雜訊。少數不遵守的仍由 `labelToLabelCode` 兜底回 `-1`。
- **雙層併發控制**：`concurrencyPerModel`（每個模型的 in-flight 上限）+ `maxConcurrentModels`（同時載入幾個模型，避免塞爆 GPU 記憶體）。網路錯誤自動重試（tenacity，最多 3 次指數退避）。
- **失敗不中斷**：單筆推論失敗會寫入 `"Error:..."` marker 而非拋例外，下游 `ResponseParser` 看到後標 `-1`（無法判定），整批照常完成。指標計算一律排除 `-1`。
- **Upper Bound 分析**：評估報告除了各組合的 Accuracy / Precision / Recall / F1 / MCC，還算出「完美挑選組合」時的準確率天花板——全部組合、F1 前 10、F1 前 20 各一份。若天花板遠低於目標，代表再怎麼試 prompt 也無效，需從資料或模型本身改進。

---

## 安裝

需要 **Python 3.11**（開發環境為 3.11.14）。

```bash
pip install -r requirements.txt
```

另需在本機安裝並啟動 [Ollama](https://ollama.com/)，並先拉好要用的模型：

```bash
ollama pull llama3.2:1b
ollama serve            # 預設監聽 http://localhost:11434
```

---

## 使用方式

所有實驗參數都寫在 [Main_EnsemblePrompt.py](Main_EnsemblePrompt.py) 的 `main()` 與 `runExperiment()` 裡（無 CLI，改參數即改實驗）。直接執行：

```bash
python Main_EnsemblePrompt.py
```

### 常改的參數

在 `main()`（Step 1）：

```python
b_exhaustiveCmb        = True                         # True=窮舉(Auto)；False=手動(Manual)
selectedPromptTechList = ["EMO", "Role", "Few_shot"] # 參與組合的分類；["ALLMethod"]=全部
maxCmbNum              = 3                            # 一組最多含幾個分類
manualPromptCmbList    = [["EMO01", "RAR02"], ...]   # 手動模式下明確列出的組合
promptTechPath         = "data/PromptGeneration/PromptTechnique_PPISimpified.yaml"
```

Dataset 與輸出路徑：

```python
trainDatasetPath = "data/PPIDataset/HPRD50/HPRD50_train.csv"  # 必要欄位：taskID, passage, label
testDatasetPath  = "data/PPIDataset/HPRD50/HPRD50_test.csv"
trainOutputRoot  = "data/output/PPI/HPRD50/train"
testOutputRoot   = "data/output/PPI/HPRD50/test"
```

在 `runExperiment()`（Step 3 / Step 4）：

```python
selectedModels      = ["llama3.2:1b"]  # 要測試的 Ollama 模型清單，可放多個
ollamaUrl           = "http://localhost:11434/api/chat"
concurrencyPerModel = 2                # 每個模型的並發請求數
maxConcurrentModels = 1                # 同時載入的模型數
llmOptions = {"temperature": 0, "num_predict": 60, "num_ctx": 8192, "num_gpu": 99}
```

### 切換到 BC5CDR

在 `runExperiment()` 把 `taskType` 改為 `"BC5CDR"`，並對應調整：`sentenceColumns=["title", "abstract"]`、`taskTemplate` 需含 `{items}` 佔位符、設定 `itemTemplate` / `itemColumns`。Dataset CSV 需含 `taskID` + `items`（JSON array，每筆含 `sentID` / `label` / `e1` / `e2`）欄位。

---

## 資料格式

**PPI Dataset CSV**（必要欄位 `taskID`, `passage`, `label`）：

```csv
taskID,passage,label
HPRD50.d0.s0_0,"...two evolutionarily conserved subunits ( PROTEIN1 and PROTEIN2 ...",no
HPRD50.d1.s0_3,"Identification of residues in the PROTEIN1 that contact ... PROTEIN2",yes
```

`label` 欄推導出 `labelSet`（類別清單），其在清單中的索引即整數標籤 `labelCode`（`0..N-1`），比對時大小寫不敏感、去空白。

---

## 輸出檔案

每個 split 的輸出目錄（如 `data/output/PPI/HPRD50/train/`）底下：

```
ppiPromptCmb.csv               # Step 1：生成的所有 Prompt 組合（train/test 共用，放在上一層）
datasetPromptInfo.csv          # Step 3：待執行任務清單（含 system/user prompt），供檢視
response.csv                   # Step 4：模型原始回應 + checkpoint（斷點續跑來源）
result.csv                     # Step 5：解析後的長表（一 item 一列）
Result_PromptCmb/              # Step 5：按 promptCmbID 分檔，方便檢視單一組合
mlTable.csv                    # Step 6：精簡寬表（每樣本一列、每 runKey 一欄 __pred），給評估吃
fullResultInfo.csv             # Step 6：完整寬表，另含 __raw（原文）與 __sysPrompt，供人工 review
eval/
  evalSummary.csv                       # 各 runKey 指標（按 F1 排序）+ 末尾三列 Upper Bound
  FalsePredictionByAllPromptCmb.csv     # 所有組合都答錯的難題清單
  correctnessHeatmap.png                # 全組合 × 樣本的對錯熱圖（綠=對、紅=錯）
  plots/CM<runKey>.png                  # 每個組合一張混淆矩陣
```

---

## 專案結構

```
EnsemblePrompt/
├── Main_EnsemblePrompt.py          # 進入點：串接 7 步，對 train/test 各跑一次
├── requirements.txt
├── PromptExecution/                # 各階段模組（每檔一個階段，無狀態、資料流在 Main 一眼可見）
│   ├── PromptCmbGen.py             # Step 1
│   ├── DataLoader.py               # Step 2
│   ├── TaskBuilder.py              # Step 3
│   ├── OllamaEngine.py             # Step 4（非同步推論引擎）
│   ├── ResponseParser.py           # Step 5
│   ├── LLMResultProcessor.py       # Step 6
│   ├── Evaluate.py                 # Step 7
│   ├── schemas.py                  # 型別 / schema / 例外 / 共用常數的單一事實來源
│   └── ExperimentInitializer.py    # logger + 隨機種子
└── data/
    ├── PromptGeneration/           # Prompt 方法池 YAML
    ├── PPIDataset/                 # PPI / BC5CDR 資料集
    └── output/                     # 實驗輸出
```
