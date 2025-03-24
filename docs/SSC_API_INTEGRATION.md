# Integración con la API del Space Science Center (SSC)

Este documento describe la integración de Orbix con la API del Space Science Center (SSC) de la NASA para obtener datos orbitales reales de satélites.

## Descripción General

La API SSC proporciona acceso a datos orbitales históricos y en tiempo real de satélites y observatorios espaciales. Esta integración permite a Orbix:

- Obtener una lista de satélites disponibles
- Recuperar datos de trayectorias orbitales para satélites específicos
- Utilizar datos reales para análisis de colisiones y optimización de rutas

## Configuración

La integración requiere las siguientes variables de entorno:

```
SSC_API_URL=https://sscweb.gsfc.nasa.gov/WS/sscr/2
SSC_API_KEY=YourApiKeyHere
```

Estas variables se pueden configurar en un archivo `.env` en la raíz del proyecto o directamente en el entorno de ejecución.

## Uso en el Código

### Inicialización del Cliente

```python
from orbix.ssc_api import SSCApi

ssc_api = SSCApi()
```

### Obtener Satélites Disponibles

```python
available_satellites = ssc_api.get_available_satellites()
print(f"Satélites disponibles: {available_satellites}")
```

### Obtener Datos de Trayectoria

```python
from datetime import datetime, timedelta

now = datetime.now()
yesterday = now - timedelta(days=1)

# Obtener datos de un satélite para las últimas 24 horas
satellite_data = ssc_api.get_satellite_data(
    satellites=["themisa"],  # ID del satélite
    start_time=yesterday,
    end_time=now,
    resolution_factor=1  # 1 = máxima resolución
)

# Procesar los datos
if "satellites" in satellite_data and "themisa" in satellite_data["satellites"]:
    trajectory = satellite_data["satellites"]["themisa"]
    print(f"Obtenidos {len(trajectory)} puntos de trayectoria")
```

## Endpoints de la API

La integración expone los siguientes endpoints en la API de Orbix:

### GET /ssc/satellites

Devuelve la lista de satélites disponibles en la API SSC.

**Respuesta:**
```json
{
  "satellites": ["themisa", "themisb", "goes13", ...]
}
```

### GET /ssc/satellite/{satellite_id}/data

Obtiene datos de trayectoria para un satélite específico.

**Parámetros:**
- `days` (opcional): Número de días hacia atrás para obtener datos (predeterminado: 1)
- `resolution` (opcional): Factor de resolución (1=máxima resolución, predeterminado: 1)

**Respuesta:**
```json
{
  "satellites": {
    "themisa": [
      {
        "time": "2