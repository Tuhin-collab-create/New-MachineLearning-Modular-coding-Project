import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.config.configuration import TRAIN_FILE_PATH, TEST_FILE_PATH, RAW_FILE_PATH,DATASET_PATH 
from src.logger import logging
from src.exception import CustomException
from src.constant import *
from src.components.data_transformation import DataTransformationConfig,DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion

class Train:
    def __init__(self):
        self.c = 0
        print(f"**************{self.c}****************")
        
    def main(self):
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_training(train_arr, test_arr))
         