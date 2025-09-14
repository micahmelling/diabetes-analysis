# Flask app X
# what the data contract looks like X
# Dockerfile X
# requirement.txt X
# readme (curl)


import json
import pandas as pd
import joblib
from flask import Flask, request


app = Flask(__name__)
MODEL = joblib.load('random_forest.pkl')


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def convert_json_to_dataframe(json_object):
    return pd.DataFrame.from_dict([json_object], orient='columns')


@app.route("/", methods=["POST", "GET"])
def home():
    """
    Home route that will confirm if the app is healthy
    """
    return "app is healthy"


@app.route("/health", methods=["POST", "GET"])
def health():
    """
    Health check endpoint that wil confirm if the app is healthy
    """
    return "app is healthy"


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to make predictions
    {
    "age_group": "under_40",
    "admission_type_id": 6,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 3,
    "num_lab_procedures": 25,
    "num_procedures": 0,
    "num_medications": 1
    }
    """
    input_data = request.json
    input_df = convert_json_to_dataframe(input_data)

    input_df['admission_type_id'] = input_df['admission_type_id'].astype(str)
    input_df['discharge_disposition_id'] = input_df['discharge_disposition_id'].astype(str)
    input_df['admission_source_id'] = input_df['admission_source_id'].astype(str)

    prediction = round(MODEL.predict_proba(input_df)[0][1], 2)

    return {
        'prediction': prediction
    }

