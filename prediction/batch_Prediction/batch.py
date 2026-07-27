import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.constant import *
import pickle
from src.utils import load_model
from sklearn.pipeline import Pipeline



PREDICTION_FOLDER = "batch prediction"
PREDICTION_CSV = "prediction_csv"
PREDICTION_FILE = "prediction.csv"
FEATURE_ENG_FOLDER = "feature_eng"

ROOT_DIR = os.getcwd()
BATCH_PREDICTION = os.path.join(ROOT_DIR,PREDICTION_FOLDER,PREDICTION_CSV)
FEATURE_ENG = os.path.join(ROOT_DIR,PREDICTION_FOLDER,FEATURE_ENG_FOLDER)

class batch_prediction:
    def __init__(self,input_file_path,model_file_path,
                 transformer_file_path,feature_engineering_file_path):
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.transformer_file_path = transformer_file_path
        self.feature_engineering_file_path = feature_engineering_file_path
    
    def start_batch_prediction(self):
        try:
            with open(self.feature_engineering_file_path,'rb') as f:
                feature_pipeline = pickle.load(f)
                
            with open (self.transformer_file_path,'rb')as f:
                processor = pickle.load(f)    
            
            model = load_model(file_path=self.model_file_path)
            
            feature_engineering_pipeline = Pipeline([
                ('feature_eng',feature_pipeline)
            ])
            
            df = pd.read_csv(self.input_file_path)
            
            df = feature_engineering_pipeline.transform(df)
            
            df.to_csv('Feature_engineering.csv')
            
            file_path=os.path.join(FEATURE_ENG,'batch_feature_eng.csv')
            df.to_csv(file_path,index = False)
            
            if 'Time_taken (min)' in df.columns:
                df = df.drop('Time_taken (min)', axis=1)
            
            transformed_data= processor.transform(df)
            
            file_path =  os.path.join(FEATURE_ENG,'processor.csv')            
            
            prediction = model.predict(transformed_data)
            
            df_prediction = pd.DataFrame(prediction,columns=['PREDICTION'])
            
            os.makedirs(BATCH_PREDICTION,exist_ok=True)
            csv_path = os.path.join(BATCH_PREDICTION,'output.csv')
            df_prediction.to_csv(csv_path,index= False)
            logging.info(f"batch prediction done")
            
        except Exception as e:
            raise CustomException(e,sys)              
