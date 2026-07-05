import logging
import os
from pathlib import Path

from PromptExecution.ExperimentInitializer import ExperimentInitializer
from PromptExecution.PromptCmbGen import PromptCmbGen
from PromptExecution.DataLoader import DataLoader
from PromptExecution.TaskBuilder import TaskBuilder
from PromptExecution.OllamaEngine import LLMEngine
from PromptExecution.ResponseParser import ResponseParser
from PromptExecution.LLMResultProcessor import LLMResultProcessor
from PromptExecution.Evaluate import PromptCmbEval

_ROOT = Path(__file__).parent
os.chdir(_ROOT)


def main():
    #============================================================================#
    #   環境初始化#
    #============================================================================#
    initObj = ExperimentInitializer()
    initObj.initializeGlobalLogger(logDir="data/logs", logName="llmLog.log")
    initObj.setupSeed(42)
    logging.info("========================================")
    logging.info("        Experiment Start                ")
    logging.info("========================================")

    #============================================================================#
    #   Step 1: 生成 Prompt 組合（PromptTech YAML →　窮舉組合 → 寫 promptCmbPath）#
    #============================================================================#
    logging.info("[Step 1/7] Generating Prompt Combinations (shared by train/test)")
    b_exhaustiveCmb    = True                                                   # True=窮舉組合(Auto)；False=手動指定(Manual)
    selectedPromptTechList = ["EMO", "Role", "Few_shot"]   # Auto：參與組合的方法分類；['ALLMethod']=全部
    maxCmbNum          = 3                                                          # Auto：一組最多包含幾個方法分類
    manualPromptCmbList    = [["EMO01", "RAR02"],            # Manual：明確列出的組合（b_exhaustiveCmb=False 時生效）
                          ["S2A01", "RE201"]]

    promptTechPath     = "data/PromptGeneration/PromptTechnique_PPISimpified.yaml"     # Prompt 方法池(Technique)來源 YAML
    promptCmbPath      = "data/output/PPI/HPRD50/ppiPromptCmb.csv"                            # 生成的 Prompt 組合 CSV（共用）

    pgObj = PromptCmbGen()
    promptTechPoolDict = pgObj.loadMethodPool(promptTechPath)
    promptCmbDf = pgObj.genPromptCmb(promptTechPoolDict, b_exhaustiveCmb, selectedPromptTechList, maxCmbNum, manualPromptCmbList)
    pgObj.savePromptCmb(promptCmbDf, promptCmbPath)

    #============================================================================#
    #   Step 2~7: 對 train、test 各跑一次（中段流程共用 runExperiment）#
    #============================================================================#
    trainDatasetPath = "data/PPIDataset/HPRD50/HPRD50_train.csv" # 所需欄位:　（taskID,passage,label）
    testDatasetPath  = "data/PPIDataset/HPRD50/HPRD50_test.csv"
    trainOutputRoot  = "data/output/PPI/HPRD50/train"
    testOutputRoot   = "data/output/PPI/HPRD50/test"



    runExperiment("train", trainDatasetPath, trainOutputRoot, promptCmbDf)
    runExperiment("test", testDatasetPath, testOutputRoot, promptCmbDf)

    logging.info("[Pipeline] 流程結束（train + test 皆完成）")


def runExperiment(splitName, datasetPath, outputRoot, promptCmbDf):
    """對單一 split（train 或 test）跑 Step 2~7。promptCmbDf 由 train/test 共用，外部傳入。"""
    logging.info("==================================================")
    logging.info(f"   資料集: {splitName}  →  {outputRoot}")
    logging.info("==================================================")
    outputRoot = Path(outputRoot)
    outputRoot.mkdir(parents=True, exist_ok=True)

#============================================================================#
#   Step 2: 載入 DatasetCsv 與 checkpoint#
#============================================================================#
    logging.info("[Step 2/7] Loading DatasetCsv & checkpoint")
    taskType       = "PPI"                                                  # 推論模式：'PPI' 或 'BC5CDR'
    labelColumn    = "label"                                            # DatasetCsv 中攜帶 true label 的欄位（PPI 必填）
    sentenceColumns = ["passage"]                                       #DatasetCsv 中要判斷的句子欄位，會塞在TaskTemplate裡面的Sentence
    responsePath   = outputRoot / "response.csv"        # 模型原始回應（未解析）；也是 checkpoint 來源


    dlObj = DataLoader()
    if taskType == "PPI":
        datasetDf, labelSet = dlObj.loadDatasetPPI(datasetPath, labelColumn, sentenceColumns)
    else:
        datasetDf, labelSet = dlObj.loadDatasetBC5CDR(datasetPath, sentenceColumns)
    finishedTaskRunIDSet = dlObj.loadCompletedTaskRunIDs(responsePath)

#============================================================================#
#   Step 3: 建構待執行 LLM 任務#
#============================================================================#
    logging.info("[Step 3/7] Building LLM Tasks")
    selectedModels    = ["llama3.2:1b"]              # 要測試的 Ollama 模型清單， ["llama3.2:1b","deepseek-r1:8b","gemma3:latest",......]
    datasetPromptInfoPath = outputRoot / "datasetPromptInfo.csv"  # 寫入promptInfo 檔案，內含Dataset+UserPrompt+systemPrompt
    taskTemplate = (
        "Sentence: {passage}\n"
        "\n"
        "Task: Based on the sentence, do PROTEIN1 and PROTEIN2 have a protein-protein interaction (PPI)?\n"
        "Answer with exactly one of the allowed labels: no, yes.\n"
    )
    # BC5CDR 模式專用（僅 taskType=='BC5CDR' 時使用；PPI 時忽略）。
    # 註：真的要跑 BC5CDR，上面的 taskType/sentenceColumns/taskTemplate 也需改成 BC5CDR 版
    #     （sentenceColumns=['title','abstract']、taskTemplate 需含 {items} 佔位符）。
    maxItemsPerBatch = 20                        # 一篇多個 entity pair 時，每批最多幾個 pair
    itemTemplate     = "{i}. {e1} - {e2}\n"      # 每個 pair 的組裝樣板，拼接後塞進 taskTemplate 的 {items}
    itemColumns      = ["e1", "e2"]              # 要組裝進 itemTemplate 的 item 欄（None=全部非保留欄）

    tbObj = TaskBuilder()
    if taskType == "PPI":
        userPromptDf = tbObj.buildUserPromptPPI(datasetDf, sentenceColumns, taskTemplate, labelColumn)
    else:
        userPromptDf = tbObj.buildUserPromptBC5CDR(datasetDf, sentenceColumns, taskTemplate,
                                                   maxItemsPerBatch, itemTemplate, itemColumns)
    # userPromptDf → × models × prompts、跳過已完成 → promptInfoDf
    promptInfoDf = tbObj.buildPromptInfo(userPromptDf, promptCmbDf, selectedModels, finishedTaskRunIDSet)

    # 檢視檔保留所有欄位（sentence、items 皆原名原樣）。
    tbObj.savePromptInfo(promptInfoDf, datasetPromptInfoPath)  # 存待執行任務供檢視
#============================================================================#
#   Step 4: LLM 推論（結果 append 到 response.csv）#
#============================================================================#
    if not promptInfoDf.empty: 
        #在buildPromptInfo 中透過finishedTaskRunIDSet 過濾掉已完成的任務，若 promptInfoDf 為空，表示所有任務都已完成，直接跳過推論步驟。
        logging.info(f"[Step 4/7] Running Inference ({len(promptInfoDf)} tasks)")
        ollamaUrl           = "http://localhost:11434/api/chat"  # Ollama API 端點
        ollamaTimeout       = 600                                         # 單次請求的最大等待秒數
        concurrencyPerModel = 2                                       # 每個模型同時發送的最大並發請求數
        maxConcurrentModels = 1                                      # 同時運行推論的最大模型數量
        llmOptions = {
            "temperature": 0,        # 取樣隨機性；0 表示完全確定性輸出
            "num_predict": 60,      # 最多生成的 token 數量
            "num_ctx":     8192,    # 模型的 context window 大小（token）
            "num_gpu":     99,      # 使用的 GPU 層數；99 表示盡量全部放 GPU
        }

        llmEngObj = LLMEngine()
        llmEngObj.runAllTasks(promptInfoDf, ollamaUrl, ollamaTimeout, llmOptions, concurrencyPerModel, maxConcurrentModels, responsePath, labelSet.buildOllamaOutputFormat(taskType))
    else:
        logging.info("[Step 4/7] All tasks completed. Skipping inference.")

#============================================================================#
#   Step 5: 解析 LLM 輸出（response.csv → result.csv）#
#============================================================================#
    logging.info("[Step 5/7] Parsing LLM Outputs")
    resultPath      = outputRoot / "result.csv"                     # ResponseParser 解析後的預測結果
    singleOutputDir = outputRoot / "Result_PromptCmb"           # 單一 prompt 組合的輸出子目錄
    singleOutputDir.mkdir(parents=True, exist_ok=True)


    rpObj = ResponseParser()
    responseDf = rpObj.loadResponse(responsePath)
    resultDf = rpObj.parseToResultDf(responseDf, labelSet)
    rpObj.saveResults(resultDf, resultPath, singleOutputDir)
# (model, promptCmbID) 組合須唯一，否則下游 pivot 成 runKey 欄時會撞名
#============================================================================#
#   Step 6: 後處理（result.csv → mlTable / fullResultInfo）#
#============================================================================#
    logging.info("[Step 6/7] Processing Results")
    mlTablePath = outputRoot / "mlTable.csv"      # pivot 後的精簡寬表
    fullResultInfoPath    = outputRoot / "fullResultInfo.csv"               # 含原始欄位的完整寬表


    lrpObj = LLMResultProcessor()
    mlTableDf = lrpObj.writeMLTable(resultDf)
    fullResultInfoDf = lrpObj.writeFullResultInfo(resultDf, mlTableDf, promptCmbDf)   # 直接吃長表，內部自行 pivot 出 __pred
    mlTableDf.to_csv(mlTablePath, index=False, encoding='utf-8-sig')
    fullResultInfoDf.to_csv(fullResultInfoPath, index=False, encoding='utf-8-sig')

#============================================================================#
#   Step 7: 評估（mlTable → eval/）#
#============================================================================#
    logging.info("[Step 7/7] Evaluating")
    evalDir = outputRoot / "eval"   # 評估產物的 root 資料夾（名稱由 runExperiment 參數傳入，可自訂）
    evalDir.mkdir(parents=True, exist_ok=True)
    # evalDir 底下各產物的完整路徑（與其他步驟一致，皆為 <dir> / 檔名），可在此自訂
    summaryPath = evalDir / "evalSummary.csv"                     # 指標總表
    falsePredictionByAllPromptPath  = evalDir / "FalsePredictionByAllPromptCmb.csv"   # 全組合皆答錯的難題清單
    heatmapPath = evalDir / "correctnessHeatmap.png"             # 對錯熱圖
    plotsDir    = evalDir / "plots"                               # 各組合混淆矩陣 PNG 的子資料夾

    evalObj = PromptCmbEval(evalDir)
    evalObj.loadMLTableDf(mlTableDf)                                     # 傳入 Step 6 的寬表
    evalObj.evalPromptCmb(labelSet)                             #計算每個PromptCmb 的分數（Acc/P/R/F1/MCC）
    evalObj.analyzeUpperBound()                                     # 算難題與準確率天花板（全部 / F1 前 10 / 前 20）
    evalObj.plotConfusionMatrices(labelSet, plotsDir)        # 每組合畫混淆矩陣 → evalDir/plots/
    evalObj.plotHeatmap(heatmapPath)                          # 畫全組合對錯熱圖 → evalDir/correctnessHeatmap.png
    evalObj.saveSummary(summaryPath, falsePredictionByAllPromptPath)             # 輸出 evalSummary.csv + FalsePredictionByAllPromptCmb.csv
    logging.info(f"[資料集:{splitName}] 完成 → {outputRoot}")


if __name__ == "__main__":
    main()
