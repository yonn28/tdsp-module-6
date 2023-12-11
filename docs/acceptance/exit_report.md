<!-- #region -->
# Informe de salida

## Resumen Ejecutivo

Durante el proceso de ejecucion del proyecto se noto que se para hacer un sistema de predicción de terremotos 



## Resultados del proyecto

- Resumen de los entregables:

| Etapa                                        | Logros                                                                                    |
|----------------------------------------------|-------------------------------------------------------------------------------------------|
| Entendimiento del negocio y carga de datos   | Se obtuvo el dataset                                                                      |
| Preprocesamiento, análisis exploratorio      | Se limpiaron los datos quitando valores nulos y algunas escalas que no estaban en Richter |
| Modelamiento y extracción de características | Identificación del modelo; en este caso, se utilizó una regresión lineal                  |
| Despliegue                                   | Despliegue en Railway                                                                     |
| Evaluación y entrega final                   | Del 1 de diciembre al 8 de diciembre                                                      |



- El modelo base se obtuvo usando la libreria optuna, en la cual se trato de optimizar dos modelos una regresion linear y una red neuronal con una capa intermedia despues de 10 iteraciones se llego a que la regresion lineal con los hiperparametros fit_intercept: True, y normalize: True. Eran la mejor optimización con un MAE de 0.31361
![comparacion_de_modelos.PNG](attachment:comparacion_de_modelos.PNG)
- Con un MAE de 0.31361

## Lecciones aprendidas

- En este dataset en especial durante el preprocesamiento se noto que algo como la fecha no tiene mucha relacion con la magnitud del terremoto.
- Algunos casos la eleccion del modelo complejo no refleja un mejor desempeño en performance.
- Tratar primero con los modelos más sencillos antes de probar algúnos de los más complejos esto puede ayudar con el gasto en recursos.



## Impacto del proyecto

- Mejora en los sistemas de alerta en las ciudades en las cuales se tienen registros de alta actividad sismica.
- Realizar algún tipo de data augmentation para tener suficiente información con la cual entrenar un modelo complejo.

## Conclusiones

- Se obtuvo un resultado desplegado y listo para ser consumido usando REST.
- Aunque no es un modelo que tenga aplicacion en la vida real, ya que la predicción de terrmotos es más cuestion preventiva que no predictiva ya que puede suceder en cualquier momento solo que con más probabilidad en algunas zonas.

## Agradecimientos

- Gracias a los monitores, y profesor por la retroalimentación durante la realización del proyecto.
<!-- #endregion -->
