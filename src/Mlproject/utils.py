import os
import sys
import pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from src.Mlproject.logger import logging
from src.Mlproject.exception import CustomException

# Load environment variables from .env file
load_dotenv()

# Retrieve database credentials from .env
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

def save_object(file_path, obj):
    """
    Saves a Python object using pickle.

    Parameters:
        file_path (str): Path where the object should be saved.
        obj (object): The object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)  # Ensure the directory exists

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)  # Save the object

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Loads a Python object using pickle.

    Parameters:
        file_path (str): Path to the saved object.

    Returns:
        object: The loaded Python object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(models, X_train, y_train, X_test, y_test):
    """
    Evaluates multiple regression models on the dataset.

    Parameters:
        models (dict): Dictionary of models { 'model_name': model_object }
        X_train (array): Training features
        y_train (array): Training target variable
        X_test (array): Testing features
        y_test (array): Testing target variable

    Returns:
        dict: A dictionary with model names and their performance metrics.
    """
    try:
        results = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)  # Train the model
            y_pred = model.predict(X_test)  # Make predictions

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results[model_name] = {"MSE": mse, "R2 Score": r2}

        return results

    except Exception as e:
        raise CustomException(e, sys)

def perform_grid_search(model, param_grid, X_train, y_train):
    """
    Performs Grid Search to find the best hyperparameters for a model.

    Parameters:
        model (object): The ML model
        param_grid (dict): Dictionary of hyperparameters
        X_train (array): Training features
        y_train (array): Training target variable

    Returns:
        object: Best model with tuned hyperparameters
    """
    try:
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='r2', cv=5, verbose=1)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

    except Exception as e:
        raise CustomException(e, sys)
