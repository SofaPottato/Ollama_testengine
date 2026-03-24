import logging
import os
import sys
from pathlib import Path
import pandas as pd
from .TaskBuilder import TaskBuilder
from .LLMEngine import InferenceManager
from .OutputParser import RegexOutputParser
from .LLMResultProcessor import LLMResultProcessor
from .Evaluate import LLMEvaluationSystem
from .schemas import DataLoadError, TaskBuildError, InferenceError, ParsingError, LLMAppConfig 


class ExperimentPipeline:
    def __init__(self, config: LLMAppConfig):
        """
        初始化實驗流程 (大腦統籌中心)
        """
        logging.info("Initializing ExperimentPipeline()")
        self.config = config
        self.pathsConfig = self.config.paths
        # =================放路徑的地方=================
        self.dataPath = self.pathsConfig.dataPath
        self.promptsPath = self.pathsConfig.promptsPath
        self.mainOutputDir = self.pathsConfig.mainOutputDir
        self.singlePromptOutputDir = self.pathsConfig.singlePromptOutputDir
        self.evalDataDir = self.pathsConfig.evalDataDir
        self.rawTempPath = self.pathsConfig.rawTempOutputPath    
        self.parsedCsvPath = self.pathsConfig.rawOutputPath             
        self.resultOutputPath = self.pathsConfig.resultOutputPath       
        self.mergedOutputPath = self.pathsConfig.mergedLlmOutputPath    
        
        # =================建立所有必要的輸出資料夾=================
        self.rawTempPath.parent.mkdir(parents=True, exist_ok=True)  
        self.mainOutputDir.mkdir(parents=True, exist_ok=True)
        self.singlePromptOutputDir.mkdir(parents=True, exist_ok=True)
        self.evalDataDir.mkdir(parents=True, exist_ok=True)
        self.mergedOutputPath.parent.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"ExperimentPipeline initialized.")
        
    def getCompletedTasks(self) -> set:
        """
        讀取暫存檔，取得已完成的任務指紋 (Model, PromptID, UserPrompt)
        """
        completed = set()
        if not self.rawTempPath.exists():
            return completed
            
        try:
            tempDf = pd.read_csv(str(self.rawTempPath), encoding='utf-8-sig')
            
            # 確保檔案沒有壞掉且具備需要的欄位
            if all(col in tempDf.columns for col in ['model', 'promptID', 'userPrompt']):
                for _, row in tempDf.iterrows():
                    onlyID = (str(row['model']), str(row['promptID']), str(row['userPrompt']))
                    completed.add(onlyID)
                    
            logging.info(f"♻️ 發現斷點紀錄！已載入 {len(completed)} 筆完成的任務。")
        except Exception as e:
            logging.warning(f"⚠️ 讀取斷點紀錄失敗，將忽略舊紀錄重新開始: {e}")
            
        return completed
    
    def run(self):
        """執行實驗流程：嚴格把關，發生錯誤即 Crash"""

        logging.info(f"==== [Step 1] Loading Data & Prompts ====")
        df = self.loadDataSet()
        prompts = self.loadPromptTemplate(self.promptsPath)
        logging.info("==== [Step 2] Building Tasks ====")
        completedTasks = self.getCompletedTasks()
        builder = TaskBuilder(
            models=self.config.selectedModels,
            pairNumber=self.config.pairSettings.pairNumbers,
            taskTemplate=self.config.taskTemplate
        )
        
        tasksList = builder.buildLLMInferenceTasks(df, prompts, completedTasks=completedTasks)

        if not tasksList and not completedTasks: 
            raise TaskBuildError("無效的任務建構：無法生成任何新任務，且沒有找到歷史斷點紀錄。")

        if tasksList:
            logging.info(f"==== [Step 3] Running LLM Inference ({len(tasksList)} tasks remaining) ====")
            engine = InferenceManager(
                rawOutputPath=self.pathsConfig.rawCsvOutputPath, 
                apiUrl=self.config.apiUrl,
                timeout=self.config.timeout,
                llmOptions=self.config.llmOptions,
                concurrencyPerModel=self.config.concurrencyPerModel
            )
            rawOutputStrPath = engine.dispatchTasksToAsyncEngine(tasksList)
            
            if not rawOutputStrPath or not os.path.exists(rawOutputStrPath):
                raise InferenceError("推論失敗：沒有產生暫存檔案。")
        else:
            logging.info("==== [Step 3] All tasks already completed! Skipping Inference. ====")

        logging.info("==== [Step 4] Parsing LLM Outputs ====")
        parser = RegexOutputParser(
            rawCsvPath=self.rawTempPath,
            csvOutputPath=self.parsedCsvPath,
            singlePromptDir=self.singlePromptOutputDir
        )
        parsedOutputStrPath = parser.parse()
        
        if not parsedOutputStrPath or not os.path.exists(parsedOutputStrPath):
            raise ParsingError("解析失敗：沒有產生解析後的 CSV 檔案。")

        logging.info("==== [Step 5] Processing Results ====")
        processor = LLMResultProcessor(
            inputCsvPath=parsedOutputStrPath,
            outputCsvPath=str(self.resultOutputPath),
            mergedPath=str(self.mergedOutputPath),
            originalDf=df 
        )
        processedCsvPath = processor.cleanAndMergeOriginalData()
        if not processedCsvPath:
            raise PipelineError("資料後處理失敗，無法產出最終 CSV。")

        logging.info("==== [Step 6] Evaluating ====")
        evaluator = LLMEvaluationSystem(
            inputCsvPath=processedCsvPath, 
            outputBaseDir=str(self.evalDataDir)
        )

        evaluator.runEvaluation()
        evaluator.analyzeDifficulty()
        evaluator.plotConfusionMatrices()
        evaluator.plotHeatmap()
        evaluator.saveResults()
        
        logging.info("實驗全部執行完畢！(Pipeline Completed Successfully!)")

    def loadDataSet(self):
        if not self.dataPath or not os.path.exists(self.dataPath):
            raise DataLoadError(f"找不到指定的資料集檔案: {self.dataPath}")
        
        df = pd.read_csv(self.dataPath)
        if self.config.testLimits is not None:
            limit = self.config.testLimits
            df = df.head(limit)
            logging.warning(f"⚠️test: Using only first {limit} pairs.")
        return df

    def loadPromptTemplate(self, path: Path) -> list:
        if not path.exists():
            raise DataLoadError(f"找不到 Prompt CSV 檔案: {path}")
            
        promptsDf = pd.read_csv(path)
        if 'promptID' not in promptsDf.columns or 'promptText' not in promptsDf.columns:
            raise DataLoadError("Prompt CSV 遺失必要的 'promptID' 或 'promptText' 欄位。")
            
        return promptsDf.to_dict(orient='records')