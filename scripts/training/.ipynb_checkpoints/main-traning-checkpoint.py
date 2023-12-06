import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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

# Probando un modelo de regresion lineal

X, y = df.drop('Magnitude', axis=1), df['Magnitude']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

reg = LinearRegression()

model = reg.fit(X_train, y_train)

reg.score(X_test, y_test)

# Probando una red neuronal profunda

X_train.shape[1]

input_layer = tf.keras.layers.Input(shape=X_train.shape[1])

x=tf.keras.layers.Dense(units=20, activation="relu")(input_layer)

output_layer = tf.keras.layers.Dense(units=1, activation="linear")(x)

model = tf.keras.models.Model(inputs=input_layer,outputs=output_layer)

model.compile(optimizer='adam', loss="mse",metrics=['mae'])

history=model.fit(x=X_train, y=y_train, epochs=20, batch_size=32)

plt.plot(history.history['loss'], label='Training Loss')

from sklearn.metrics import mean_absolute_error

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

mae

predictions

y_test
