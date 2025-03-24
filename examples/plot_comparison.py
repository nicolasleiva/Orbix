# Función para visualizar la comparación de algoritmos cuánticos

import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(results):
    """
    Visualiza los resultados de la comparación de algoritmos cuánticos para
    diferentes modelos de ruido y niveles de riesgo de colisión.
    
    Args:
        results: Diccionario con los resultados de cada algoritmo
    """
    # Configurar el estilo de la visualización
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Crear figura con subplots para cada nivel de riesgo
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Comparación de Algoritmos Cuánticos para Detección de Colisiones', fontsize=16)
    
    # Títulos para cada subplot
    risk_titles = ['Riesgo Alto', 'Riesgo Medio', 'Riesgo Bajo']
    
    # Colores para cada algoritmo
    colors = {'vqe': 'blue', 'grover': 'green', 'qaoa': 'red', 'basic': 'purple'}
    
    # Etiquetas para los modelos de ruido
    noise_labels = ['Sin Ruido', 'Ruido Bajo', 'Ruido Alto']
    
    # Posiciones en el eje X
    x = np.arange(len(noise_labels))
    width = 0.2  # Ancho de las barras
    
    # Iterar sobre cada nivel de riesgo (alto, medio, bajo)
    for i, risk_idx in enumerate([0, 1, 2]):  # 0=alto, 1=medio, 2=bajo
        ax = axes[i]
        ax.set_title(risk_titles[i])
        ax.set_ylim(0, 1.0)  # Probabilidad entre 0 y 1
        ax.set_ylabel('Probabilidad de Colisión')
        ax.set_xticks(x)
        ax.set_xticklabels(noise_labels)
        
        # Offset para cada algoritmo
        offsets = [-width*1.5, -width/2, width/2, width*1.5]
        
        # Iterar sobre cada algoritmo
        for j, (algo, algo_results) in enumerate(results.items()):
            # Extraer probabilidades para este nivel de riesgo
            probs = [result[risk_idx + 1] for result in algo_results]  # +1 porque el índice 0 es el nombre del modelo de ruido
            
            # Dibujar barras
            ax.bar(x + offsets[j], probs, width, label=algo.upper(), color=colors[algo], alpha=0.7)
    
    # Añadir leyenda en el último subplot
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(results), bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('quantum_algorithms_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()