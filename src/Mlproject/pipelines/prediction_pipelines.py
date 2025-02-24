import sys
import pandas as pd
from src.Mlproject.exception import CustomException
from src.Mlproject.utils import load_object

class PredictPipelines:
    def __init__(self):
        pass
    
    def predict(self,features):
        
        try:
        
        
        
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
            raise CustomException(e,sys)

    
class CustomData:
    def __init__(self,
                age:int,
                sex:bool,
                bmi:float,
                children:int,
                smoker:bool,
                region:bool,
                expenses:float):
        
        self.age =age
        self.sex =sex
        self.bmi =bmi
        self.children=children
        self.smoker=smoker
        self.region=region
        self.expenses=expenses
        
        
    def get_data_as_data_frame(self):
        
        try:
            custom_data_input_dict={
                "age":[self.age],
                "sex":[self.sex],
                "bmi":[self.bmi],
                "smoker":[self.smoker],
                "region":[self.region],
                "expenses":[self.expenses],
                "children":[self.children]   
            }
                
                
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)