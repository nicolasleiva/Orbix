# Integración con la API de SpaceX

Este documento describe la integración de Orbix con la API pública de SpaceX para obtener información sobre lanzamientos, cohetes, cápsulas y misiones espaciales.

## Descripción General

La API de SpaceX proporciona acceso a datos históricos y actuales sobre las operaciones de SpaceX. Esta integración permite a Orbix:

- Obtener información sobre lanzamientos pasados y futuros
- Acceder a detalles técnicos de cohetes y vehículos espaciales
- Consultar datos sobre cápsulas y su historial de misiones
- Complementar los datos orbitales de la NASA con información comercial espacial

## Configuración

La integración requiere la siguiente variable de entorno:

```
SPACEX_API_URL=https://api.spacexdata.com/v3
```

Esta variable se puede configurar en un archivo `.env` en la raíz del proyecto o directamente en el entorno de ejecución.

## Uso en el Código

### Inicialización del Cliente

```python
from orbix.spacex_api import SpaceXApi

spacex_api = SpaceXApi()
```

### Obtener Lanzamientos

```python
# Obtener todos los lanzamientos
launches = spacex_api.get_launches()
print(f"Total de lanzamientos: {len(launches['data'])}")

# Obtener lanzamientos con paginación
launches_page = spacex_api.get_launches(limit=10, offset=5)
print(f"Lanzamientos (página): {len(launches_page['data'])}")
```

### Obtener Información de un Lanzamiento Específico

```python
# Obtener información del lanzamiento con número de vuelo 65
launch = spacex_api.get_launch(flight_number=65)
print(f"Misión: {launch['mission_name']}")
print(f"Fecha: {launch['launch_date_utc']}")
print(f"Éxito: {launch['launch_success']}")
```

### Obtener Información de Cohetes

```python
# Obtener todos los cohetes
rockets = spacex_api.get_rockets()
print(f"Total de cohetes: {len(rockets['data'])}")

# Obtener información de un cohete específico
rocket = spacex_api.get_rocket(rocket_id="falcon9")
print(f"Nombre: {rocket['name']}")
print(f"Descripción: {rocket['description']}")
```

### Obtener Información de Cápsulas

```python
# Obtener todas las cápsulas
capsules = spacex_api.get_capsules()
print(f"Total de cápsulas: {len(capsules['data'])}")

# Obtener información de una cápsula específica
capsule = spacex_api.get_capsule(capsule_serial="C101")
print(f"Estado: {capsule['status']}")
print(f"Tipo: {capsule['type']}")
print(f"Misiones: {len(capsule['missions'])}")
```

## Endpoints de la API

La integración expone los siguientes endpoints en la API de Orbix:

### GET /spacex/launches

Devuelve información sobre lanzamientos de SpaceX.

**Parámetros:**
- `limit` (opcional): Número máximo de lanzamientos a obtener
- `offset` (opcional): Número de lanzamientos a saltar

**Respuesta:**
```json
{
  "data": [
    {
      "flight_number": 65,
      "mission_name": "Telstar 19V",
      "launch_date_utc": "2018-07-22T05:50:00.000Z",
      "rocket": {
        "rocket_id": "falcon9",
        "rocket_name": "Falcon 9",
        "rocket_type": "FT"
      },
      "launch_site": {
        "site_name": "CCAFS SLC 40"
      },
      "launch_success": true
    },
    ...
  ]
}
```

### GET /spacex/launches/{flight_number}

Obtiene información sobre un lanzamiento específico de SpaceX.

**Respuesta:**
```json
{
  "flight_number": 65,
  "mission_name": "Telstar 19V",
  "launch_date_utc": "2018-07-22T05:50:00.000Z",
  "rocket": {
    "rocket_id": "falcon9",
    "rocket_name": "Falcon 9",
    "rocket_type": "FT"
  },
  "launch_site": {
    "site_name": "CCAFS SLC 40"
  },
  "launch_success": true,
  "details": "SSL-manufactured communications satellite intended to be placed in geostationary orbit at 63° West longitude, after an initial orbital raise and testing phase. The satellite is expected to have a useful life of about 15 years. Based on the SSL 1300 platform."
}
```

### GET /spacex/rockets

Devuelve información sobre cohetes de SpaceX.

**Respuesta:**
```json
{
  "data": [
    {
      "id": "falcon9",
      "name": "Falcon 9",
      "type": "rocket",
      "active": true,
      "stages": 2,
      "boosters": 0,
      "cost_per_launch": 50000000,
      "success_rate_pct": 97,
      "first_flight": "2010-06-04",
      "country": "United States",
      "company": "SpaceX",
      "height": {
        "meters": 70,
        "feet": 229.6
      },
      "diameter": {
        "meters": 3.7,
        "feet": 12
      },
      "mass": {
        "kg": 549054,
        "lb": 1207920
      },
      "payload_weights": [...],
      "first_stage": {...},
      "second_stage": {...},
      "engines": {...},
      "landing_legs": {...},
      "description": "Falcon 9 is a two-stage rocket designed and manufactured by SpaceX for the reliable and safe transport of satellites and the Dragon spacecraft into orbit."
    },
    ...
  ]
}
```

### GET /spacex/rockets/{rocket_id}

Obtiene información sobre un cohete específico de SpaceX.

**Respuesta:**
```json
{
  "id": "falcon9",
  "name": "Falcon 9",
  "type": "rocket",
  "active": true,
  "stages": 2,
  "boosters": 0,
  "cost_per_launch": 50000000,
  "success_rate_pct": 97,
  "first_flight": "2010-06-04",
  "country": "United States",
  "company": "SpaceX