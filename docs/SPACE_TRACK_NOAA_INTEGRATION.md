# Integración con las APIs de Space-Track y NOAA

Este documento describe la integración de Orbix con las APIs de Space-Track y NOAA para obtener datos orbitales y meteorológicos espaciales.

## Integración con Space-Track

### Descripción General

La API de Space-Track proporciona acceso a datos orbitales detallados, incluyendo TLEs (Two-Line Elements), catálogo de satélites, datos de conjunciones y decaimiento orbital. Esta integración permite a Orbix:

- Obtener el catálogo completo de satélites y objetos espaciales
- Acceder a TLEs actualizados para cualquier objeto orbital
- Consultar datos de posibles conjunciones (colisiones) entre objetos
- Obtener información sobre objetos en decaimiento orbital

### Configuración

La integración requiere las siguientes variables de entorno:

```
SPACE_TRACK_USER=your_username
SPACE_TRACK_PASS=your_password
```

Estas variables se pueden configurar en un archivo `.env` en la raíz del proyecto o directamente en el entorno de ejecución.

### Uso en el Código

#### Inicialización del Cliente

```python
from orbix.space_track_api import SpaceTrackApi

space_track_api = SpaceTrackApi()
```

#### Obtener Catálogo de Satélites

```python
# Obtener todo el catálogo
catalog = space_track_api.get_satellite_catalog()
print(f"Total de objetos en catálogo: {len(catalog['data'])}")

# Obtener catálogo limitado
limited_catalog = space_track_api.get_satellite_catalog(limit=100)
print(f"Objetos (limitado): {len(limited_catalog['data'])}")
```

#### Obtener TLE Más Reciente

```python
# Obtener TLE para la ISS (NORAD ID: 25544)
iss_tle = space_track_api.get_latest_tle(norad_id=25544)
print(f"TLE Línea 1: {iss_tle['data'][0]['TLE_LINE1']}")
print(f"TLE Línea 2: {iss_tle['data'][0]['TLE_LINE2']}")
```

#### Obtener Datos de Conjunciones

```python
# Obtener datos de conjunciones para los próximos 7 días
conjunctions = space_track_api.get_conjunction_data(days_from_now=7)
print(f"Posibles conjunciones: {len(conjunctions['data'])}")
```

#### Obtener Datos de Decaimiento Orbital

```python
# Obtener datos de decaimiento para los próximos 30 días
decay_data = space_track_api.get_decay_data(days_from_now=30)
print(f"Objetos en decaimiento: {len(decay_data['data'])}")
```

## Integración con NOAA

### Descripción General

La API de NOAA (National Oceanic and Atmospheric Administration) proporciona acceso a datos meteorológicos espaciales, incluyendo viento solar, índices geomagnéticos, llamaradas solares y alertas. Esta integración permite a Orbix:

- Obtener datos actuales del viento solar
- Acceder a índices geomagnéticos
- Consultar información sobre llamaradas solares
- Obtener pronósticos de auroras y alertas de clima espacial

### Configuración

La integración requiere la siguiente variable de entorno:

```
NOAA_TOKEN=your_api_token
```

Esta variable se puede configurar en un archivo `.env` en la raíz del proyecto o directamente en el entorno de ejecución.

### Uso en el Código

#### Inicialización del Cliente

```python
from orbix.noaa_api import NOAAApi

noaa_api = NOAAApi()
```

#### Obtener Datos del Viento Solar

```python
solar_wind = noaa_api.get_solar_wind_data()
print(f"Datos del viento solar: {len(solar_wind['data'])} puntos")
```

#### Obtener Índices Geomagnéticos

```python
geomag_indices = noaa_api.get_geomagnetic_indices()
print(f"Índices geomagnéticos: {len(geomag_indices['data'])} puntos")
```

#### Obtener Datos de Llamaradas Solares

```python
# Obtener datos de llamaradas solares para el último día
solar_flares = noaa_api.get_solar_flare_data(days=1)
print(f"Llamaradas solares: {len(solar_flares['data'])} puntos")
```

#### Obtener Datos de Flujo de Protones

```python
proton_flux = noaa_api.get_proton_flux_data()
print(f"Datos de flujo de protones: {len(proton_flux['data'])} puntos")
```

#### Obtener Pronóstico de Auroras

```python
aurora_forecast = noaa_api.get_aurora_forecast()
print(f"Pronóstico de auroras: {len(aurora_forecast['data'])} puntos")
```

#### Obtener Alertas de Clima Espacial

```python
space_weather_alerts = noaa_api.get_space_weather_alerts()
print(f"Alertas de clima espacial: {len(space_weather_alerts['data'])}")
```

## Integración con el Sistema de Predicción Cuántica

Los datos obtenidos de Space-Track y NOAA pueden utilizarse para mejorar la precisión del sistema de predicción cuántica de Orbix:

- Los TLEs actualizados de Space-Track proporcionan datos orbitales precisos para la predicción de trayectorias
- Los datos de conjunciones pueden utilizarse para validar