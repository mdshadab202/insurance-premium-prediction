# import os
# import sys
# import numpy as np
# import pandas as pd
# from src.Mlproject.logger import logging
# from src.Mlproject.exception import CustomException
# from src.Mlproject.utils import save_object, evaluate_models
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from dataclasses import dataclass

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path = os.path.join("artifacts", "model.pkl")

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()

#     def train_model(self, train_array, test_array):
#         try:
#             logging.info("Splitting dependent and independent variables.")
            
#             X_train, y_train = train_array[:, :-1], train_array[:, -1]
#             X_test, y_test = test_array[:, :-1], test_array[:, -1]

#             models = {
#                 "Linear Regression": LinearRegression(),
#                 "Decision Tree": DecisionTreeRegressor(),
#                 "Random Forest": RandomForestRegressor()
#             }

#             model_report = evaluate_models(models, X_train, y_train, X_test, y_test)
#             logging.info(f'model_report:\n {model_report}')

#             best_model_name = max(model_report, key=model_report.get('r2_score'))
#             best_model = models[best_model_name]
            
            
#             print("This is the model ")
#             print ("best model name ")
        

#             logging.info(f"Best Model Found: {best_model_name}")

#             save_object(self.model_trainer_config.trained_model_file_path, best_model)

#             return best_model_name, model_report[best_model_name]

#         except Exception as e:
#             raise CustomException(e, sys)


import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from src.Mlproject.logger import logging
from src.Mlproject.exception import CustomException
from src.Mlproject.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def train_model(self, train_array, test_array):
        try:
            logging.info("Splitting dependent and independent variables.")
            
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor()
            }

            # Evaluate models
            model_report = evaluate_models(models, X_train, y_train, X_test, y_test)
            logging.info(f'Model Report:\n {model_report}')

            # Select the best model based on R2 Score
            best_model_name = max(model_report, key=lambda k: model_report[k]['r2_score'])
            best_model = models[best_model_name]

            logging.info(f"Best Model Found: {best_model_name}")

            # Hyperparameter tuning (only for RandomForest)
            if isinstance(best_model, RandomForestRegressor):
                param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
                grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='r2')
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                logging.info("Hyperparameter tuning completed.")

            # Save the best model
            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            return best_model_name, model_report[best_model_name]

        except Exception as e:
            raise CustomException(e, sys)
