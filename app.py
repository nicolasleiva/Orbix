import streamlit as st
import pandas as pd
import numpy as np
from src.frontend import orbital_3d_map

# Configuración de la página
st.set_page_config(
    page_title="Orbix - Sistema de Navegación Orbital",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("Orbix - Sistema de Navegación Orbital")
st.markdown("""
    Sistema avanzado para la predicción de trayectorias orbitales, optimización de rutas 
    y prevención de colisiones en tiempo real.
""")

# Barra lateral con opciones
st.sidebar.title("Panel de Control")
option = st.sidebar.selectbox(
    "Seleccione una vista",
    ["Mapa Orbital 3D", "Análisis de Trayectorias", "Alertas de Colisión"]
)

# Contenido principal basado en la opción seleccionada
if option == "Mapa Orbital 3D":
    st.header("Visualización 3D de Objetos Orbitales")
    
    # Datos de ejemplo para visualización (en producción, estos vendrían de la API)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parámetros de Visualización")
        num_objects = st.slider("Número de objetos a mostrar", 1, 10, 3)
        altitude_range = st.slider("Rango de altitud (km)", 300, 1000, (400, 800))
    
    with col2:
        st.subheader("Filtros")
        object_types = st.multiselect(
            "Tipos de objetos",
            ["Satélites activos", "Satélites inactivos", "Desechos espaciales"],
            ["Satélites activos"]
        )
    
    # Generar datos de ejemplo para la visualización
    predictions = []
    for i in range(num_objects):
        altitude = np.random.randint(altitude_range[0], altitude_range[1])
        velocity = round(7.5 + np.random.random(), 2)  # Velocidad orbital aproximada
        
        # Generar trayectoria orbital simple
        radius = 6371 + altitude  # Radio de la Tierra + altitud
        theta = np.linspace(0, 2*np.pi, 100)
        inclination = np.random.random() * np.pi/4
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta) * np.cos(inclination)
        z = radius * np.sin(theta) * np.sin(inclination)
        
        coordinates = [[float(x[j]), float(y[j]), float(z[j])] for j in range(0, 100, 5)]
        
        predictions.append({
            "coordinates": coordinates,
            "altitude": altitude,
            "velocity": velocity,
            "satellite_id": f"SAT-{i+1}"
        })
    
    # Visualizar el mapa 3D
    st.pydeck_chart(orbital_3d_map(predictions))
    
    # Información adicional
    st.subheader("Detalles de los Objetos")
    details_df = pd.DataFrame([
        {"ID": p["satellite_id"], "Altitud (km)": p["altitude"], "Velocidad (km/s)": p["velocity"]} 
        for p in predictions
    ])
    st.table(details_df)

elif option == "Análisis de Trayectorias":
    st.header("Análisis de Trayectorias Orbitales")
    st.info("Esta sección permite analizar trayectorias orbitales y predecir posiciones futuras.")
    
    # Aquí iría la implementación real conectada a la API
    st.write("Funcionalidad en desarrollo. Conecte con la API para datos en tiempo real.")

elif option == "Alertas de Colisión":
    st.header("Sistema de Alertas de Colisión")
    st.info("Esta sección muestra alertas de posibles colisiones basadas en análisis de trayectorias.")
    
    # Aquí iría la implementación real conectada a la API
    st.write("Funcionalidad en desarrollo. Conecte con la API para datos en tiempo real.")

# Pie de página
st.markdown("---")
st.markdown("""
    <div style="text-align: center">
        <p>© 2023 Orbix Team | <a href="https://github.com/orbix-team/orbix">GitHub</a></p>
    </div>
""", unsafe_allow_html=True)