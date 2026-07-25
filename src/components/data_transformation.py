from src.logger import logging
from src.exception import CustomException
import os, sys
from src.config.configuration import *
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from src.utils import save_obj
from src.config.configuration import PREPROCESSING_OBJ_FILE,TRANSFORM_TRAIN_FILE_PATH,TRANSFORM_TEST_FILE_PATH,FEATURE_ENGG_OBJ_FILE_PATH

class Feature_Engineering(BaseEstimator,TransformerMixin):
    def __init__(self):
        logging.info("********feature Engineering Started*************")
        
    def distance_numpy(self, df, lat1, lon1,lat2, lon2 ):
        p = np.pi/180
        a = 0.5 - np.cos((df[lat2]-df[lat1])*p)/2 + np.cos(df[lat1]*p) * np.cos(df[lat2]*p) * (1-np.cos((df[lon2]-df[lon1])*p))/2
        df['distance'] = 12734 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    
    def transform_data(self,df):
        try:
            self.distance_numpy(df,"Restaurant_latitude","Restaurant_longitude",
                                "Delivery_location_latitude","Delivery_location_longitude")
            df.drop([
                "Restaurant_latitude","Restaurant_longitude",'ID',
                "Delivery_location_latitude","Delivery_location_longitude",
                "Delivery_person_ID","Order_Date","Time_Orderd","Time_Order_picked"                
            ],axis=1,inplace=True)
            
            logging.info("Drropping colummns from our dataqset")
            return df
        except Exception as e:
            raise CustomException(e,sys)    
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X:pd.DataFrame,y=None):
        try:    
            transformed_df=self.transform_data(X)   
            return transformed_df
        except Exception as e:
            raise CustomException(e,sys) from e
    
@dataclass    
class DataTransformationConfig():
    preprocessed_object_file = PREPROCESSING_OBJ_FILE
    transform_train_file_path = TRANSFORM_TRAIN_FILE_PATH
    transform_test_file_path= TRANSFORM_TEST_FILE_PATH
    feature_engg_obj_path = FEATURE_ENGG_OBJ_FILE_PATH
 
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation_obj(self):
        try:
            Road_traffic_density= ['Jam', 'High', 'Medium', 'Low']
            Weather_conditions =['Fog', 'Stormy', 'Sandstorms', 'Windy', 'Cloudy', 'Sunny']
            
            ordinal_columns = ["Road_traffic_density","Weather_conditions"]
            categorical_columns = ['Type_of_order', 'Type_of_vehicle', 'Festival', 'City']
            numerical_columns= ['Delivery_person_Age', 'Delivery_person_Ratings', 
                                'Vehicle_condition', 'multiple_deliveries','distance']
            
            numerical_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='constant',fill_value=0)),
                ('scaler',StandardScaler(with_mean=False))
            ])
            
            categorical_pipeline = Pipeline(steps=[
                ('impute',SimpleImputer(strategy='most_frequent')),
                ('onehot',OneHotEncoder(handle_unknown= 'ignore')),
                ('scaler',StandardScaler(with_mean=False))
            ])
            
            ordinal_pipeline = Pipeline(steps=[
                ('impute',SimpleImputer(strategy='most_frequent')),
                ('ordinal',OrdinalEncoder(categories=[Road_traffic_density, Weather_conditions]))
            ])
            
            preprocessor = ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_columns),
                ('categorical_pipelie',categorical_pipeline,categorical_columns),
                ('ordinal_pipeline',ordinal_pipeline,ordinal_columns)
            ])
            
            logging.info('Pipeline steps complete')
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    
    def get_feature_engineering_object(self):
        try:
            feature_engineering = Pipeline(steps=[
                ('fe', Feature_Engineering())
            ])
            return feature_engineering
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path) 
            
            logging.info("train and test data loaded")
            fe_obj= self.get_feature_engineering_object()
            
            train_df = fe_obj.fit_transform(train_df)
            test_df =fe_obj.transform(test_df)
            
            train_df.to_csv("train_df.csv")
            test_df.to_csv("test_df.csv")
            logging.info('train and test data is saved after data_transformation successfully.')
            
            processing_obj = self.get_data_transformation_obj()
            target_column = "Time_taken (min)"
            
            X_train = train_df.drop(columns=target_column,axis=1)
            y_train = train_df[target_column]
            
            X_test = test_df.drop(columns=target_column,axis=1)
            y_test = test_df[target_column]
            
            X_train = processing_obj.fit_transform(X_train)
            X_test = processing_obj.transform(X_test)
            
            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]
            
            df_train = pd.DataFrame(train_arr)
            df_test = pd.DataFrame(test_arr)
            
            os.makedirs(os.path.dirname(self.data_transformation_config.transform_train_file_path),exist_ok=True)
            df_train.to_csv(self.data_transformation_config.transform_train_file_path,index=False,header=True)
            
            os.makedirs(os.path.dirname(self.data_transformation_config.transform_test_file_path),exist_ok=True)
            df_test.to_csv(self.data_transformation_config.transform_test_file_path,index=False,header=True)

            
            save_obj(file_path=self.data_transformation_config.preprocessed_object_file,obj = processing_obj)
            save_obj(file_path=self.data_transformation_config.feature_engg_obj_path,obj = fe_obj)
            
            return (train_arr,test_arr,
                    self.data_transformation_config.preprocessed_object_file)
            
        except Exception as e:
            raise CustomException(e,sys)
