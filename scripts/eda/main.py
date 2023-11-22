# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

df = pd.read_csv('../../src/nombre_paquete/database/database.csv')

df.head()

df.shape

df.dtypes

df.describe()

# Todos los valores de source corresponden a agencias que monitorean los terremotos ISCGEM(inernational seismological centre)

df['Source'].value_counts()

df['Root Mean Square'].max()

# Tipo de magnitud puede ser ML (local Richter magnitude), MS(Surface wave mangnitude)

df['Type'].value_counts()

df['Status'].value_counts()

# Validación de los valores nulos

df.isna().sum()

# Algunas columnas como source, Location Source, Magnitude Source, ID, Magnitude Error, Magnitude Seismic Stations, Azimuthal Gap  no aportan para predecir la profundidad o la magnitud de un terremoto ya que son solo relacionadas a como se realizaron las mediciones.

columns_to_drop = ['ID', 'Source', 'Location Source','Magnitude Source','Magnitude Error','Magnitude Seismic Stations','Azimuthal Gap']

df = df.drop(columns=columns_to_drop)


# para los valores faltantes tomando la media los valores numericos, y la moda para los valores categoricos.

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

# Tomando solo las variables numericas para hacer un grafico de corelacion

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




