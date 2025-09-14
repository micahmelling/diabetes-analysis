# Predict Readmission Rates

# EDA
Data is intended to be stored in the data directory. The data is fairly clean, but we need to ensure
that variables are cast to the correct type. We also need to improve the categorical coding of certain variables. From
EDA, we see a moderate connection to the chosen target of "readmitted". We opted to turn this into a binary variable for modeling. 
We mostly relied on statistical correlations (mutual information) to get a quick feel for important variables.
For computational purposes, only certain variables were chosen to be able to train models in the allotted amount of time.
Overall, we were aware of feature leakage and chose to stick with seemingly "safe" features when subsetting data.

# Modeling
Our modeling problem was to predict the likelihood of readmission. 

We used hyperopt to predict the probability of readmission. We trained a single random forest classification model,
though our set up allows us to easily add models. Consequently, this is the model that was put into the API. We optimized
the model for calibrated predictions, and it appeared to do solidly given the time constraints on training. In the future,
we would add more models and evaluation / explanation techniques.

# Other Business Questions
1) Predict length of hospital stay given variables that would be known at the time of admission.
2) Use unsupervised ML to find similar cases and use that to inform treatment plans.
3) Treat this as more of a research project to tease out the effect that certain variables have on readmission. In other words, move beyond prediction.

In sum, these goals will help with proactive treatment and planning to improve care and isolate important factors. We also need
to retrain the model if we use predictions to influence the environment. We need to be aware of the feedback loop we might create.

## API Section
Use the API (app.py) to pass in a json payload to get a predicted probability or readmission.

Send POST request to /predict endpoint. Locally, you can run this the Flask dev server and hit 
localhost over port 5000. 

Here is a sample input payload:

```javascript
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
```

Here is a sample output payload

```javascript
{
    "prediction": prediction
}
```

You can call the service with curl, as such:

```bash
$ curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age_group": "under_40",
    "admission_type_id": 6,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 3,
    "num_lab_procedures": 25,
    "num_procedures": 0,
    "num_medications": 1
  }'

```
You can turn this app into a containerized app via Docker using the Dockerfile. It will boot gunicorn running over port 8000. 
The gunicorn calls also reference key and cert files that can be used for end-to-end encryption. These can be created
with OpenSSL. 

```bash
$ docker build -t diabetes . <br>
$ docker run --rm -it diabetes
```

