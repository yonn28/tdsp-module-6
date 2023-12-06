# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf
import os, mlflow
import mlflow.tensorflow as mlflow_tf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
from pydantic import BaseModel
from typing import List
import requests

command = """
mlflow server \
        --backend-store-uri sqlite:///tracking.db \
        --default-artifact-root file:mlruns \
        -p 5000 &
"""
get_ipython().system_raw(command)

token = "2Xs6g3C5Rk0uJFoY7LurDyXvspO_5LFmydpknG4XmCkJE37A" # Agregue el token dentro de las comillas
os.environ["NGROK_TOKEN"] = token

# !ngrok authtoken $NGROK_TOKEN

from pyngrok import ngrok
ngrok.connect(5000, "http")

mlflow.set_tracking_uri("http://localhost:5000")

exp = mlflow.create_experiment(name="models", artifact_location="models_runs")

df = pd.read_csv('../../src/nombre_paquete/database/database.csv')

df.head()

df.shape

df.dtypes

df.describe()

df['Source'].value_counts()

df['Root Mean Square'].max()

df['Type'].value_counts()

df['Status'].value_counts()

df.isna().sum()

columns_to_drop = ['ID', 'Source', 'Location Source','Magnitude Source','Magnitude Error','Magnitude Seismic Stations','Azimuthal Gap']

df = df.drop(columns=columns_to_drop)

def process_dataframe(df):
    null_counts = df.isnull().sum()
    for column in df.columns:
        if null_counts[column] > df.shape[0]*0.1:
            df = df.drop(column, axis=1)
        else:
            if pd.api.types.is_numeric_dtype(df[column]):    
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)
            else:
                most_common = df[column].mode().iloc[0]
                df[column] = df[column].fillna(most_common)
    return df

df=process_dataframe(df)

numerical_columns = df.select_dtypes(include=["float64"]).columns
numerical_data = df[numerical_columns]

sns.heatmap(numerical_data.corr(), cmap="YlGnBu", annot=True);

sns.boxplot(x=df['Depth'])

sns.distplot(df['Depth'])

sns.distplot(df['Magnitude'])

df.plot.scatter(x='Depth',
                      y='Magnitude')

sns.pairplot(df,hue='Status')

fig = px.scatter_geo(df, lat='Latitude', lon='Longitude', hover_name='Magnitude')
fig.update_layout(title='Terremotos en los periodos, 1965-2016', title_x=0.5)
fig.show()

# codificacion de las variables categoricas

df.head()

df=df.drop(["Date", "Time"], axis=1)

df['Type']=pd.factorize(df['Type'])[0]

df['Magnitude Type']=pd.factorize(df['Magnitude Type'])[0]

df['Status']=pd.factorize(df['Status'])[0]

# partiendo en test y prueba

X, y = df.drop('Magnitude', axis=1), df['Magnitude']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

# Probando un modelo de regresion lineal

with mlflow.start_run(experiment_id=exp, run_name='linear_regression'):
    reg = LinearRegression()
    model_reg = reg.fit(X_train, y_train)
    mlflow.sklearn.log_model(model_reg, "model")
    y_pred_reg = model_reg.predict(X_test)
    
    
    mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred_reg))

# Probando una red neuronal profunda

with mlflow.start_run(experiment_id=exp, run_name="deep_learning"):
    input_layer = tf.keras.layers.Input(shape=X_train.shape[1])
    x=tf.keras.layers.Dense(units=20, activation="relu")(input_layer)
    output_layer = tf.keras.layers.Dense(units=1, activation="linear")(x)
    model_deep = tf.keras.models.Model(inputs=input_layer,outputs=output_layer)
    model_deep.compile(optimizer='adam', loss="mse",metrics=['mae'])
    history_deep=model_deep.fit(x=X_train, y=y_train, epochs=15, batch_size=32)
                                
    mlflow.keras.log_model(model_deep, artifact_path="model")
    y_pred_deep = model_deep.predict(X_test)
    mlflow.log_metric("mae",mean_absolute_error(y_test, y_pred_deep))

plt.plot(history_deep.history['loss'], label='Training Loss')

# haciendo dump para el modelo de regresion lineal y probando en notebook para hacer el deployment

joblib.dump(model_reg, "model.joblib")


# +
# Reemplace esto con su implementación:
class ApiInput(BaseModel):
    features: List[float]

# Reemplace esto con su implementación:
class ApiOutput(BaseModel):
    magnitude: float


# -

def predict(data: ApiInput):
    model=joblib.load("model.joblib")
    features_2d = [data.features]
    labels_predict = model.predict(features_2d).flatten().tolist()
    print(labels_predict[0])
    prediction = ApiOutput(magnitude=labels_predict[0])
    return prediction


# +
inp = ApiInput(features=[-46.2066,165.9628,0,20.0,8,1])
print(type(inp), inp.features)

pred = predict(inp)
display(pred)
# -

y_test.iloc[0]

# codigo de despliegue

# +
# %%writefile main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib

# Reemplace esto con su implementación:
class ApiInput(BaseModel):
    features: List[float]

# Reemplace esto con su implementación:
class ApiOutput(BaseModel):
    magnitude: float

app = FastAPI()
model = joblib.load("model.joblib")

# Reemplace esto con su implementación:
@app.post("/predict")
async def predict(data: ApiInput) -> ApiOutput:
    features_2d = [data.features]
    labels_predict = model.predict(features_2d).flatten().tolist()
    prediction = ApiOutput(magnitude=labels_predict[0])
    return prediction
# -

# !mkdir mlapi
# !mv main.py model.joblib mlapi/
# %cd mlapi/

# %%writefile requirements.txt
scikit-learn
fastapi
uvicorn

# %%writefile railway.json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}

# probando el despliegue

model_url = "https://mlds-6-deployment-production.up.railway.app"

inp = ApiInput(features=[-46.2066,165.9628,0,20.0,8,1])
r = requests.post(
    os.path.join(model_url, "predict"),
    json=inp.dict(),
    )
print(r.json())
