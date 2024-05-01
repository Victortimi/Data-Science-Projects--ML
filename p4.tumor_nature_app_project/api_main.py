from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from starlette.responses import Response
from pydantic import BaseModel
import joblib
import pandas as pd
from io import StringIO
from typing import Optional
from Test_Connection import insert_json_data, insert_csv_data, past_prediction
from api_preprocesser import preprocessing
from typing import List
import json
import warnings
import datetime

# Filter out the scikit-learn warning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
app = FastAPI()

# Load the machine learning model
model = joblib.load('model_lrib.joblib')


class Features(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    
class PredictionRequest(BaseModel):
    features: Optional[Features]
    df_in: Optional[str]

class PredictionResponse(BaseModel):
    prediction: str


class PastPrediction(BaseModel):
    features: Features
    prediction: str



@app.post('/predict')
def predict(data : PredictionRequest):
    if data.features is not None:
        
        json_str = json.dumps(data.features, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)
        json_dict = json.loads(json_str)
        
        feature_json =  {"mean radius":json_dict['mean_radius'], "mean area":json_dict['mean_radius'], "mean perimeter":json_dict['mean_perimeter'], "mean texture":json_dict['mean_texture']}
        
        df = pd.json_normalize(feature_json)
        print(df)
        
        prepocessed_df = preprocessing(df)
        
        scaled = core_scale(prepocessed_df)
        
        predictions = model.predict(scaled)
        
        predictions_list = predictions.tolist()
      
         # Map 0 to "benign" and 1 to "malignant"
        predictions_mapped = ['benign' if pred == 0 else 'malignant' for pred in predictions_list]

        prepocessed_df['diagnosis'] = predictions_mapped

        prepocessed_df['timestamp'] = datetime.datetime.now()

        insert_csv_data("breast_cancer", "prediction_table", prepocessed_df)
        
        return prepocessed_df
            
    if data.df_in is not None:
        
        df = pd.read_json(data.df_in, orient='records')
        
        prepocessed_df = preprocessing(df)
        
        scaled = core_scale(prepocessed_df)
        
        predictions = model.predict(scaled)
        
        predictions_list = predictions.tolist()

        # Map 0 to "benign" and 1 to "malignant"
        predictions_mapped = ['benign' if pred == 0 else 'malignant' for pred in predictions_list]

        prepocessed_df['diagnosis'] = predictions_mapped

        prepocessed_df['timestamp'] = datetime.datetime.now()

        insert_csv_data("breast_cancer", "prediction_table", prepocessed_df)
        
        return prepocessed_df
        
# This function is created tp scale the input csv
def core_scale(df):
    scaler = joblib.load('scaler_lri.joblib')
    scaled  = scaler.transform(df)
    return scaled


@app.get('/past_predictions')
def get_past_predictions():
    past_prediction_data = past_prediction("breast_cancer", "prediction_table")
    return past_prediction_data
