from flask import Flask ,render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.Mlproject.pipelines.prediction_pipelines import CustomData,PredictPipelines


application= Flask(__name__)

app= application

##Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data =CustomData(
            age=int(request.form.get('age')),
            sex=request.form.get('sex'),
            bmi=float(request.form.get('bmi')),
            children=request.form.get('children'),
            smoker=request.form.get('smoker'),
            region=request.form.get('region'),
            expenses=request.form.get('expenses')
            
        )
        
        predict_df = data.get_data_as_data_frame()
        print(predict_df)
        
        predict_pipelines=PredictPipelines()
        results = predict_pipelines.predict(predict_df)
        return render_template('index.html',results=results[0])
    
if __name__ =="__main__":
    app.run(host="0.0.0.0",debug=True)
    