# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../../src/nombre_paquete/database/database.csv')

df.head()

# Todos los valores de source corresponden a agencias que monitorean los terremotos ISCGEM(inernational seismological centre)

df['Source'].value_counts()

df['Location Source'].value_counts()

# Tipo de magnitud puede ser ML (local Richter magnitude), MS(Surface wave mangnitude)

df['Type'].value_counts()

df['Status'].value_counts()

# Validación de los valores nulos

df.isna().sum()

# Algunas columnas como source, Location Source, Magnitude Source, ID, Magnitude Error, Magnitude Seismic Stations, Azimuthal Gap  no aportan para predecir la profundidad o la magnitud de un terremoto ya que son solo relacionadas a como se realizaron las mediciones.

columns_to_drop = ['ID', 'Source', 'Location Source','Magnitude Source','Magnitude Error','Magnitude Seismic Stations','Azimuthal Gap']

df = df.drop(columns=columns_to_drop)

# Tomando solo las variables numericas para hacer un grafico de corelacion

numerical_columns = df.select_dtypes(include=["float64"]).columns
numerical_data = df[numerical_columns]

sns.heatmap(numerical_data.corr(), cmap="YlGnBu", annot=True);








