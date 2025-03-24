# Orbix - Sistema de Análisis Cuántico de Trayectorias Satelitales

Esta aplicación utiliza algoritmos cuánticos para predecir trayectorias de satélites y detectar posibles colisiones con alta precisión.

## Configuración

Para ejecutar esta aplicación en Streamlit Cloud:

1. Configura las siguientes variables en Secrets:
   - `space_track.username`: Tu nombre de usuario de Space-Track
   - `space_track.password`: Tu contraseña de Space-Track
   - `api_keys.noaa_api_key`: Tu clave API para NOAA (si es necesario)

2. La aplicación utilizará modelos por defecto si no encuentra modelos entrenados.

## Funcionalidades

- Predicción de trayectorias satelitales usando algoritmos cuánticos
- Análisis de riesgo de colisión
- Consulta de datos de satélites desde APIs externas
- Monitoreo de condiciones espaciales