<!-- #region -->
# Reporte de Datos

Este documento contiene los resultados del análisis exploratorio de datos.

## Resumen general de los datos

El dataset tiene (23412, 9) con las columnas (Date,Time,Latitude,Longitude,Type,Depth,Depth Error, Depth Seismic Stations, Magnitude,Magnitude Type,Magnitude Error, Magnitude Seismic Stations, Azimuthal Gap, Horizontal Distance, Horizontal Error, Root Mean Square, ID, Source, Location Source, Magnitude Source, Status ).


## Resumen de calidad de los datos

Las siguiente tabla muestra la distribución de los valores nulos.
 

| Columnas                   | valores nulos |
|----------------------------|---------------|
| Date                       | 0             |
| Time                       | 0             |
| Longitude                  | 0             |
| Type                       | 0             |
| Depth                      | 0             |
| Depth Error                | 18951         |
| Depth Seismic Stations     | 16315         |
| Magnitude                  | 0             |
| Magnitude Type             | 0             |
| Magnitude Error            | 23085         |
| Magnitude Seismic Stations | 20848         |
| Azimuthal Gap              | 16113         |
| Horizontal Distance        | 21808         |
| Horizontal Error           | 22256         |
| Root Mean Square           | 6060          |
| ID                         | 0             |
| Source                     | 0             |
| Location Source            | 0             |
| Magnitude Source           | 0             |
| Status                     | 0             |

Para completar los valores faltantes utilizaré la media para los valores numericos, y para los valores categoricas la moda.


## Variable objetivo

la variable objetivo es la magnitud, y la siguiente grafica muestra la distribución. se ve que se tiene una distribución no uniforme se tienen muchos datos para algunos valores y pocos para otros.

![imgMag.PNG](attachment:imgMag.PNG)

## Variables individuales

Algunas columnas como source, Location Source, Magnitude Source, ID, Magnitude Error, Magnitude Seismic Stations, Azimuthal Gap no aportan para predecir la profundidad o la magnitud de un terremoto ya que son solo relacionadas a como se realizaron las mediciones. Por lo tanto se eliminaron.

![variablesPlot.PNG](attachment:variablesPlot.PNG)

## Ranking de variables

Las varialbes que se identificaron con mayor relacion a la magnitud son la profundidad, la logitud y la latitud, las demas variables son relaciondas a la medición de los diferentes valores, y no ayudan a realizar una proyección de la posible magnitud.

## Relación entre variables explicativas y variable objetivo

las variables numericas que más muestran relación con la magnitud son la latitud y la longitud

![correlation.PNG](attachment:correlation.PNG)
<!-- #endregion -->
