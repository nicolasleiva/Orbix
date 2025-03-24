# Ejemplo de uso del sistema de alertas cuánticas con algoritmos reales

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from src.quantum_alerts import QuantumCollisionAlertSystem, QuantumAlertConfig
from src.quantum_api_integrator import QuantumApiIntegrator
from plot_comparison import plot_comparison

# Configurar el sistema de alertas con diferentes algoritmos cuánticos
def test_quantum_algorithms():
    print("\n=== Comparación de Algoritmos Cuánticos para Detección de Colisiones ===")
    
    # Crear trayectorias de prueba
    # Trayectoria con alto riesgo de colisión (puntos muy cercanos)
    high_risk_trajectory = pd.DataFrame({
        "x": [950, 951, 952, 953, 954],
        "y": [1950, 1951, 1952, 1953, 1954],
        "z": [2950, 2951, 2952, 2953, 2954]
    })
    
    # Trayectoria con riesgo medio de colisión
    medium_risk_trajectory = pd.DataFrame({
        "x": np.linspace(950, 1100, 5),
        "y": np.linspace(1950, 2100, 5),
        "z": np.linspace(2950, 3100, 5)
    })
    
    # Trayectoria con bajo riesgo de colisión (puntos muy separados)
    low_risk_trajectory = pd.DataFrame({
        "x": [950, 1050, 1150, 1250, 1350],
        "y": [1950, 2050, 2150, 2250, 2350],
        "z": [2950, 3050, 3150, 3250, 3350]
    })
    
    # Algoritmos a probar
    algorithms = ["vqe", "grover", "qaoa", "basic"]
    noise_models = ["none", "low", "high"]
    
    # Almacenar resultados para comparación
    results = {}
    
    # Probar cada algoritmo con diferentes modelos de ruido
    for algo in algorithms:
        algo_results = []
        print(f"\nAlgoritmo: {algo.upper()}")
        
        for noise in noise_models:
            # Configurar sistema de alertas con este algoritmo y modelo de ruido
            config = QuantumAlertConfig(
                simulator_type=algo,
                noise_model=noise,
                shots=1000
            )
            alert_system = QuantumCollisionAlertSystem(config)
            
            # Calcular probabilidades para cada trayectoria
            high_prob = alert_system.calculate_collision_probability(high_risk_trajectory)
            medium_prob = alert_system.calculate_collision_probability(medium_risk_trajectory)
            low_prob = alert_system.calculate_collision_probability(low_risk_trajectory)
            
            print(f"  Modelo de ruido: {noise}")
            print(f"    Riesgo alto:   {high_prob:.4f}")
            print(f"    Riesgo medio:  {medium_prob:.4f}")
            print(f"    Riesgo bajo:   {low_prob:.4f}")
            
            algo_results.append((noise, high_prob, medium_prob, low_prob))
        
        results[algo] = algo_results
    
    # Visualizar resultados
    plot_comparison(results)

# Generar una alerta de colisión completa
def test_collision_alert():
    print("\n=== Generación de Alerta de Colisión con Algoritmo Cuántico ===")
    
    # Configurar sistema de alertas con VQE
    config = QuantumAlertConfig(
        simulator_type="vqe",
        noise_model="low",
        shots=1000
    )
    alert_system = QuantumCollisionAlertSystem(config)
    
    # Crear trayectoria de prueba
    trajectory = pd.DataFrame({
        "x": np.linspace(950, 1000, 10),
        "y": np.linspace(1950, 2000, 10),
        "z": np.linspace(2950, 3000, 10),
        "timestamp": [datetime.now() + timedelta(minutes=i*30) for i in range(10)]
    })
    
    # Generar alerta
    alert = alert_system.generate_alert(
        satellite_id="ORBIX-SAT-001",
        trajectory=trajectory,
        other_object_id="DEBRIS-22",
        additional_metadata={
            "satellite_type": "Observación Terrestre",
            "orbit_type": "LEO",
            "altitude_km": 550,
            "inclination_deg": 53.0
        }
    )
    
    # Mostrar información de la alerta
    print(f"\nAlerta generada:")
    for key, value in alert.items():
        if key == "generation_metadata" or key == "additional_metadata" or key == "confidence_interval":
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")
    
    # Publicar la alerta en Kafka
    success = alert_system.publish_alert(alert)
    if success:
        print("\nAlerta publicada exitosamente en Kafka")
    else:
        print("\nError al publicar la alerta en Kafka")

# Probar la integración con APIs externas
def test_api_integration():
    print("\n=== Prueba de Integración con APIs Externas ===")
    
    # Inicializar el integrador de APIs
    api_integrator = QuantumApiIntegrator()
    
    # Definir parámetros de prueba
    satellite_id = "25544"  # ISS (Estación Espacial Internacional)
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=24)
    
    # Obtener datos del satélite
    print(f"\nObteniendo datos para el satélite {satellite_id}...")
    satellite_data = api_integrator.get_satellite_data(satellite_id, start_time, end_time)
    
    # Verificar si se obtuvieron datos
    if "error" in satellite_data:
        print(f"Error: {satellite_data['error']}")
    else:
        print("Datos obtenidos correctamente:")
        for source, data in satellite_data.items():
            print(f"  {source}: {type(data)}")
    
    # Predecir trayectoria
    print("\nPrediciendo trayectoria futura...")
    trajectory = api_integrator.predict_trajectory(satellite_id, start_time, end_time, prediction_hours=48)
    
    if trajectory.empty:
        print("Error: No se pudo predecir la trayectoria")
    else:
        print(f"Trayectoria predicha con {len(trajectory)} puntos")
        print(trajectory.head())
    
    # Generar alerta de colisión
    print("\nGenerando alerta de colisión...")
    alert = api_integrator.generate_collision_alert(satellite_id, prediction_hours=48)
    
    if "error" in alert:
        print(f"Error: {alert['error']}")
    else:
        print(f"Alerta generada con ID: {alert.get('alert_id')}")
        print(f"Probabilidad de colisión: {alert.get('collision_probability', 0):.4f}")
        print(f"Nivel de alerta: {alert.get('alert_level', 'N/A')}")

# Función principal para ejecutar todas las pruebas
def main():
    print("=== Sistema de Alertas Cuánticas para Colisiones Orbitales ===")
    print("Ejecutando pruebas de funcionalidad...\n")
    
    # Ejecutar pruebas
    test_quantum_algorithms()
    test_collision_alert()
    test_api_integration()
    
    print("\n¡Todas las pruebas completadas!")

# Ejecutar si se llama directamente
if __name__ == "__main__":
    main()