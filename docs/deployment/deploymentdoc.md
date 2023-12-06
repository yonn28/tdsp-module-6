# Despliegue de modelos

## Infraestructura

- **Nombre del modelo:** modelo
- **Plataforma de despliegue:** runway
- **Requisitos técnicos:** scikit-learn,fastapi,uvicorn
- **Requisitos de seguridad:** por ahora el endpoint es totalmente publico en el siguiente link https://mlds-6-deployment-production.up.railway.app
- **Diagrama de arquitectura:** ![architecture.PNG](attachment:architecture.PNG)

## Código de despliegue

- **Archivo principal:** los archivos de despliegue estan el siguiente repo https://github.com/yonn28/mlds-6-deployment
- **Rutas de acceso a los archivos:** main.py, model.joblib, railway.json, requirements.txt
- **Variables de entorno:** no se tienen variables de entorno.

## Documentación del despliegue

- **Instrucciones de instalación:** hacer push al repositorio, realizar redespliegue en railway por cada push.
- **Instrucciones de configuración:** sincronizar el repo en railway, y generar en las configuraciones una nueva URL.
- **Instrucciones de uso:** enviar el vector de caracteristicas con los siguientes valores;Latitude	Longitud,Type(codificado en valor numerico usando pd.factorize),Depth,Magnitude Type,Status(codificado en valor numerico usando pd.factorize).
- **Instrucciones de mantenimiento:** validar sí los datos con el tiempo presentan algun tipo de drift
