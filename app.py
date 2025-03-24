import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import tensorflow as tf
import os
import plotly.graph_objects as go

# Importar módulos de Orbix
from src.quantum_alerts import QuantumCollisionAlertSystem
from src.ssc_api import SSCApi
from src.space_track_api import SpaceTrackApi
from src.noaa_api import NOAAApi
from src.quantum_trajectory import QuantumTrajectoryModel
from src.optimization import OrbitalPathOptimizer
from src.quantum_api_integrator import QuantumAPIIntegrator

# Configuración de la página
st.set_page_config(
    page_title="Orbix - Análisis Cuántico de Trayectorias Satelitales",
    page_icon="🛰️",
    layout="wide"
)

# Título y descripción
st.title("Orbix - Análisis Cuántico de Trayectorias Satelitales")
st.markdown("""
Esta aplicación utiliza algoritmos cuánticos reales para predecir trayectorias de satélites 
y detectar posibles colisiones con alta precisión.
""")

# Inicializar estado de sesión
if 'trajectory_df' not in st.session_state:
    st.session_state.trajectory_df = None
if 'alert' not in st.session_state:
    st.session_state.alert = None
if 'api_data' not in st.session_state:
    st.session_state.api_data = None

# Barra lateral para configuración
st.sidebar.header("Configuración")
quantum_algorithm = st.sidebar.selectbox(
    "Algoritmo Cuántico",
    ["vqe", "grover", "qaoa", "basic"],
    index=0
)

quantum_noise_model = st.sidebar.selectbox(
    "Modelo de Ruido Cuántico",
    ["none", "low", "medium", "high"],
    index=1
)

quantum_shots = st.sidebar.slider(
    "Shots Cuánticos",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100
)

# Área principal con pestañas
tab1, tab2, tab3, tab4 = st.tabs([
    "Predicción de Trayectoria", 
    "Análisis de Colisiones", 
    "Datos de Satélites",
    "Condiciones Espaciales"
])

# Pestaña 1: Predicción de Trayectoria
with tab1:
    st.header("Predicción de Trayectoria de Satélites")
    
    # Entrada para ID de satélite
    col1, col2 = st.columns([2, 1])
    with col1:
        satellite_id = st.text_input("ID del Satélite", "25544")  # ISS por defecto
    
    # Fuente de datos
    with col2:
        data_source = st.radio("Fuente de Datos", ["Space-Track API", "SSC API"])
    
    # Parámetros temporales
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Fecha de inicio", datetime.now() - timedelta(days=1))
    with col2:
        end_date = st.date_input("Fecha de fin", datetime.now() + timedelta(days=1))
    
    if st.button("Generar Trayectoria"):
        with st.spinner("Obteniendo datos y calculando trayectoria usando algoritmos cuánticos..."):
            try:
                # Inicializar el integrador de API cuántica
                api_integrator = QuantumAPIIntegrator()
                
                # Obtener datos reales de la API seleccionada
                if data_source == "Space-Track API":
                    space_track_api = SpaceTrackApi()
                    # Autenticar y obtener datos
                    space_track_api.authenticate()
                    api_data = space_track_api.get_satellite_data(
                        satellite_id=satellite_id,
                        start_time=datetime.combine(start_date, datetime.min.time()),
                        end_time=datetime.combine(end_date, datetime.max.time())
                    )
                else:  # SSC API
                    ssc_api = SSCApi()
                    api_data = ssc_api.get_satellite_data(
                        satellites=[satellite_id],
                        start_time=datetime.combine(start_date, datetime.min.time()),
                        end_time=datetime.combine(end_date, datetime.max.time())
                    )
                
                # Guardar datos de API en el estado de la sesión
                st.session_state.api_data = api_data
                
                # Convertir datos de API a formato de trayectoria
                trajectory_df = api_integrator.convert_api_data_to_trajectory(api_data)
                
                # Inicializar el modelo de trayectoria cuántica
                model = QuantumTrajectoryModel()
                model.quantum_simulator_type = quantum_algorithm
                model.quantum_noise_model = quantum_noise_model
                model.quantum_shots = quantum_shots
                
                # Cargar pesos del modelo si están disponibles
                model_path = os.path.join(os.path.dirname(__file__), "models", "quantum_trajectory_model.json")
                if os.path.exists(model_path):
                    model.load_weights(model_path)
                
                # Preparar datos para el modelo cuántico
                if not trajectory_df.empty:
                    # Extraer características para el modelo
                    features = trajectory_df[['x', 'y', 'z']].values
                    # Normalizar características
                    features_norm = features / np.max(np.abs(features))
                    
                    # Convertir a tensor de TensorFlow
                    input_tensor = tf.convert_to_tensor(
                        features_norm.reshape(1, len(features_norm), 3), 
                        dtype=tf.float32
                    )
                    
                    # Generar predicción cuántica
                    predicted_tensor = model(input_tensor)
                    predicted_trajectory = predicted_tensor.numpy()[0]
                    
                    # Escalar de vuelta a dimensiones reales
                    scale_factor = np.max(np.abs(features))
                    predicted_trajectory = predicted_trajectory * scale_factor
                    
                    # Crear DataFrame con predicción
                    future_timestamps = [
                        trajectory_df['timestamp'].iloc[-1] + timedelta(minutes=15*i) 
                        for i in range(len(predicted_trajectory))
                    ]
                    
                    prediction_df = pd.DataFrame(
                        predicted_trajectory,
                        columns=['x', 'y', 'z']
                    )
                    prediction_df['timestamp'] = future_timestamps
                    
                    # Combinar datos históricos y predicción
                    combined_df = pd.concat([trajectory_df, prediction_df], ignore_index=True)
                    st.session_state.trajectory_df = combined_df
                else:
                    st.error("No se pudieron obtener datos de trayectoria válidos de la API.")
                    return
                
                # Mostrar datos de trayectoria
                st.subheader("Datos de Trayectoria")
                st.dataframe(st.session_state.trajectory_df)
                
                # Visualizar trayectoria en 3D con Plotly
                st.subheader("Visualización 3D de Trayectoria")
                
                fig = go.Figure(data=[
                    go.Scatter3d(
                        x=trajectory_df['x'],
                        y=trajectory_df['y'],
                        z=trajectory_df['z'],
                        mode='lines+markers',
                        name='Datos Históricos',
                        marker=dict(size=4, color='blue'),
                        line=dict(color='royalblue', width=2)
                    ),
                    go.Scatter3d(
                        x=prediction_df['x'],
                        y=prediction_df['y'],
                        z=prediction_df['z'],
                        mode='lines+markers',
                        name='Predicción Cuántica',
                        marker=dict(size=4, color='red'),
                        line=dict(color='firebrick', width=2)
                    )
                ])
                
                # Añadir la Tierra como referencia
                earth_radius = 6371  # km
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = earth_radius * np.cos(u) * np.sin(v)
                y = earth_radius * np.sin(u) * np.sin(v)
                z = earth_radius * np.cos(v)
                
                fig.add_trace(go.Surface(
                    x=x, y=y, z=z,
                    colorscale='Blues',
                    opacity=0.6,
                    showscale=False,
                    name='Tierra'
                ))
                
                fig.update_layout(
                    title=f'Trayectoria para Satélite {satellite_id}',
                    scene=dict(
                        xaxis_title='X (km)',
                        yaxis_title='Y (km)',
                        zaxis_title='Z (km)',
                        aspectmode='data'
                    ),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    ),
                    margin=dict(l=0, r=0, b=0, t=30)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error al generar la trayectoria: {str(e)}")

# Pestaña 2: Análisis de Colisiones
with tab2:
    st.header("Análisis de Riesgo de Colisión")
    
    col1, col2 = st.columns(2)
    with col1:
        other_object_id = st.text_input("ID del Objeto Secundario (opcional)", "")
    
    with col2:
        confidence_threshold = st.slider(
            "Umbral de Confianza",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05
        )
    
    if st.button("Analizar Riesgo de Colisión"):
        if st.session_state.trajectory_df is None:
            st.warning("Primero debe generar una trayectoria en la pestaña 'Predicción de Trayectoria'.")
        else:
            with st.spinner("Calculando probabilidad de colisión usando algoritmos cuánticos..."):
                try:
                    # Inicializar el sistema de alertas cuánticas con los parámetros seleccionados
                    from src.quantum_alerts import QuantumAlertConfig
                    
                    config = QuantumAlertConfig(
                        simulator_type=quantum_algorithm,
                        noise_model=quantum_noise_model,
                        shots=quantum_shots,
                        confidence_threshold=confidence_threshold
                    )
                    
                    alert_system = QuantumCollisionAlertSystem(config=config)
                    
                    # Generar alerta
                    additional_metadata = {
                        "analysis_time": datetime.now().isoformat(),
                        "data_source": data_source,
                        "user_settings": {
                            "quantum_algorithm": quantum_algorithm,
                            "quantum_noise_model": quantum_noise_model,
                            "quantum_shots": quantum_shots
                        }
                    }
                    
                    alert = alert_system.generate_alert(
                        satellite_id=satellite_id,
                        trajectory=st.session_state.trajectory_df,
                        other_object_id=other_object_id if other_object_id else None,
                        additional_metadata=additional_metadata
                    )
                    
                    # Guardar alerta en el estado de la sesión
                    st.session_state.alert = alert
                    
                    # Mostrar información de alerta
                    st.subheader("Alerta de Colisión")
                    
                    # Crear columnas para mostrar la alerta
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Mostrar probabilidad con color según nivel de riesgo
                        prob = alert['collision_probability']
                        color = "green" if prob < 0.3 else "orange" if prob < 0.7 else "red"
                        st.markdown(f"<h1 style='text-align: center; color: {color};'>{prob:.4f}</h1>", unsafe_allow_html=True)
                        st.markdown("<p style='text-align: center;'>Probabilidad de Colisión</p>", unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Nivel de Alerta", alert['alert_level'])
                        st.metric("Algoritmo Cuántico", alert['quantum_algorithm'])
                    
                    with col3:
                        st.metric("Tiempo al Punto de Máximo Acercamiento", 
                                 alert['time_to_closest_approach'].split('T')[0] if alert['time_to_closest_approach'] else "N/A")
                        st.metric("Distancia Mínima Estimada", f"{alert['min_distance_estimate']:.2f} km")
                    
                    # Mostrar acciones recomendadas
                    st.subheader("Acciones Recomendadas")
                    for action in alert['recommended_actions']:
                        st.markdown(f"- {action}")
                    
                    # Mostrar intervalo de confianza
                    st.subheader("Intervalo de Confianza")
                    conf_int = alert['confidence_interval']
                    st.markdown(f"**Límite Inferior:** {conf_int['lower']:.4f}")
                    st.markdown(f"**Límite Superior:** {conf_int['upper']:.4f}")
                    
                    # Mostrar detalles completos de la alerta
                    with st.expander("Ver Detalles Completos de la Alerta"):
                        st.json(alert)
                    
                except Exception as e:
                    st.error(f"Error al analizar el riesgo de colisión: {str(e)}")

# Pestaña 3: Datos de Satélites
with tab3:
    st.header("Datos de Satélites")
    
    api_choice = st.radio("Seleccionar API", ["Space-Track", "SSC"])
    
    if st.button("Obtener Satélites Disponibles"):
        with st.spinner("Conectando a la API..."):
            try:
                if api_choice == "Space-Track":
                    api = SpaceTrackApi()
                    api.authenticate()
                    satellites = api.get_available_satellites()
                else:  # SSC
                    api = SSCApi()
                    satellites = api.get_available_satellites()
                
                if satellites:
                    st.success(f"Se encontraron {len(satellites)} satélites")
                    
                    # Mostrar satélites en una tabla
                    if isinstance(satellites, list):
                        # Convertir lista a DataFrame
                        if isinstance(satellites[0], str):
                            # Lista simple de IDs
                            df = pd.DataFrame(satellites, columns=["Satélite ID"])
                        else:
                            # Lista de diccionarios
                            df = pd.DataFrame(satellites)
                    else:
                        # Ya es un DataFrame o diccionario
                        df = pd.DataFrame(satellites)
                    
                    st.dataframe(df)
                    
                    # Opción para descargar como CSV
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Descargar como CSV",
                        data=csv,
                        file_name=f"satelites_{api_choice}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No se encontraron satélites disponibles.")
            except Exception as e:
                st.error(f"Error al obtener datos de satélites: {str(e)}")
    
    # Búsqueda de satélite específico
    st.subheader("Buscar Satélite Específico")
    search_id = st.text_input("ID o Nombre del Satélite", "")
    
    if search_id and st.button("Buscar"):
        with st.spinner("Buscando satélite..."):
            try:
                if api_choice == "Space-Track":
                    api = SpaceTrackApi()
                    api.authenticate()
                    satellite_info = api.get_satellite_info(search_id)
                else:  # SSC
                    api = SSCApi()
                    satellite_info = api.get_satellite_info(search_id)
                
                if satellite_info:
                    st.success(f"Información encontrada para {search_id}")
                    
                    # Mostrar información del satélite
                    if isinstance(satellite_info, dict):
                        # Crear dos columnas
                        col1, col2 = st.columns(2)
                        
                        # Distribuir información en las columnas
                        for i, (key, value) in enumerate(satellite_info.items()):
                            if i % 2 == 0:
                                col1.metric(key, value)
                            else:
                                col2.metric(key, value)
                    else:
                        st.write(satellite_info)
                else:
                    st.warning(f"No se encontró información para {search_id}")
            except Exception as e:
                st.error(f"Error al buscar satélite: {str(e)}")

# Pestaña 4: Condiciones Espaciales
with tab4:
    st.header("Condiciones Espaciales")
    
    data_type = st.selectbox(
        "Tipo de Datos",
        ["Viento Solar", "Clima Espacial", "Actividad Solar", "Índice Geomagnético"]
    )
    
    if st.button("Obtener Datos"):
        with st.spinner("Obteniendo datos de condiciones espaciales..."):
            try:
                # Inicializar API de NOAA
                noaa_api = NOAAApi()
                
                if data_type == "Viento Solar":
                    space_data = noaa_api.get_solar_wind_data()
                elif data_type == "Clima Espacial":
                    space_data = noaa_api.get_space_weather_data()
                elif data_type == "Actividad Solar":
                    space_data = noaa_api.get_solar_activity_data()
                else:  # Índice Geomagnético
                    space_data = noaa_api.get_geomagnetic_data()
                
                if "error" not in space_data:
                    st.success("Datos obtenidos correctamente")
                    
                    # Convertir a DataFrame si es necesario
                    if isinstance(space_data, dict):
                        if "data" in space_data:
                            df = pd.DataFrame(space_data["data"])
                        else:
                            # Convertir diccionario a DataFrame
                            df = pd.DataFrame([space_data])
                    elif isinstance(space_data, list):
                        df = pd.DataFrame(space_data)
                    else:
                        df = pd.DataFrame([{"data": space_data}])
                    
                    # Mostrar datos
                    st.dataframe(df)
                    
                    # Visualizar datos si es posible
                    if "time" in df.columns and any(col in df.columns for col in ["value", "density", "speed", "temperature"]):
                        st.subheader("Visualización de Datos")
                        
                        # Determinar qué columna visualizar
                        value_col = next((col for col in ["value", "density", "speed", "temperature"] if col in df.columns), None)
                        
                        if value_col:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(df["time"], df[value_col], marker='o', linestyle='-')
                            ax.set_title(f"{data_type} - {value_col}")
                            ax.set_xlabel("Tiempo")
                            ax.set_ylabel(value_col)
                            ax.grid(True)
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                else:
                    st.error(f"Error al obtener datos: {space_data.get('error', 'Error desconocido')}")
            except Exception as e:
                st.error(f"Error al obtener datos de condiciones espaciales: {str(e)}")

# Pie de página
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Orbix - Sistema Avanzado de Análisis Cuántico de Trayectorias Satelitales</p>
    <p>Desarrollado con algoritmos cuánticos reales para predicción de alta precisión</p>
</div>
""", unsafe_allow_html=True)