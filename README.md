# Orbix

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.6+](https://img.shields.io/badge/tensorflow-2.6+-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68.0+-green.svg)](https://fastapi.tiangolo.com/)

Orbix es un sistema avanzado de navegación orbital diseñado para predecir trayectorias, optimizar rutas y proporcionar alertas de colisión en tiempo real para satélites y objetos espaciales.

## Características

- **Predicción de Trayectorias:** Implementación de un modelo Transformer para predecir trayectorias orbitales con alta precisión.
- **Ingesta en Tiempo Real:** Pipeline robusto basado en Kafka para consumir y publicar datos TLE (Two-Line Element).
- **Optimización de Rutas Orbitales:** Algoritmos avanzados basados en OR-Tools para calcular rutas óptimas que eviten colisiones.
- **API en Tiempo Real:** Endpoints WebSocket con FastAPI para transmisión de datos en tiempo real.
- **Frontend Interactivo:** Visualización 3D de trayectorias y objetos espaciales con Streamlit y pydeck.
- **Sistema de Alertas de Colisión:** Cálculo de probabilidades de colisión basado en análisis de trayectorias.
- **Pipeline de ML con TFX:** Infraestructura completa para entrenamiento, validación y despliegue de modelos.
- **Análisis en Grafos:** Modelado de relaciones entre satélites y desechos espaciales utilizando StellarGraph.

## Instalación

### Usando pip

1. Clona el repositorio:

   ```bash
   git clone https://github.com/orbix-team/orbix.git
   cd orbix
   ```

2. Crea un entorno virtual e instala las dependencias:

   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Crea un archivo `.env` basado en `.env.example`:

   ```bash
   cp .env.example .env
   # Edita el archivo .env según tus necesidades
   ```

4. Instala el paquete en modo desarrollo:

   ```bash
   pip install -e .
   ```

### Usando Docker

1. Clona el repositorio:

   ```bash
   git clone https://github.com/orbix-team/orbix.git
   cd orbix
   ```

2. Construye y ejecuta los contenedores:

   ```bash
   docker-compose up -d
   ```

## Uso

### API

La API estará disponible en `http://localhost:8000`.

```bash
# Ejecutar la API directamente
python -m src.main
```

### Frontend

El frontend de Streamlit estará disponible en `http://localhost:8501`.

```bash
# Ejecutar el frontend directamente
streamlit run app.py
```

## Estructura del Proyecto

```
├── src/                  # Código fuente principal
│   ├── api.py            # API FastAPI con endpoints WebSocket
│   ├── config.py         # Configuración centralizada
│   ├── frontend.py       # Componentes de visualización con Streamlit
│   ├── graph_analysis.py # Análisis de grafos con StellarGraph
│   ├── main.py           # Punto de entrada principal
│   ├── ml_pipeline.py    # Pipeline de ML con TFX
│   ├── optimization.py   # Optimización de rutas orbitales
│   ├── prediction.py     # Predicción de trayectorias
│   ├── quantum_alerts.py # Sistema de alertas de colisión
│   ├── realtime_pipeline.py # Pipeline de datos en tiempo real
│   ├── telemetry.py      # Monitoreo de métricas
│   └── transformer_model.py # Modelo Transformer para predicciones
├── tests/                # Pruebas unitarias
├── docker-compose.yml    # Configuración de Docker Compose
├── Dockerfile            # Dockerfile para la API
├── Dockerfile.frontend   # Dockerfile para el frontend
├── requirements.txt      # Dependencias del proyecto
└── setup.py              # Configuración de instalación
```

## Desarrollo

### Ejecutar Pruebas

```bash
pytest
```

Para generar un informe de cobertura:

```bash
pytest --cov=src --cov-report=html
```

### Estilo de Código

Este proyecto sigue las convenciones de estilo PEP 8. Puedes verificar y formatear el código con:

```bash
# Verificar estilo
flake8 src tests

# Formatear código
black src tests

# Ordenar imports
isort src tests
```

## Contribución

1. Haz un fork del repositorio
2. Crea una rama para tu característica (`git checkout -b feature/amazing-feature`)
3. Haz commit de tus cambios (`git commit -m 'Add some amazing feature'`)
4. Haz push a la rama (`git push origin feature/amazing-feature`)
5. Abre un Pull Request

## Licencia

Distribuido bajo la Licencia MIT. Consulta `LICENSE` para más información.