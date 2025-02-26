# Neuronal Network with NumPy

Este proyecto implementa una red neuronal desde cero utilizando **NumPy** para la clasificación de datos generados sintéticamente. El objetivo es demostrar cómo construir y entrenar redes neuronales sin depender de frameworks como **TensorFlow** o **PyTorch**.

---

## 📌 Descripción del Proyecto

El proyecto consta de los siguientes componentes principales:

- **Generación de Datos**: Se crean datos sintéticos utilizando `make_gaussian_quantiles` de `sklearn.datasets` para generar un conjunto de datos de clasificación binaria.
- **Funciones de Activación**: Implementación de las funciones **ReLU** y **Sigmoid**, junto con sus derivadas.
- **Función de Pérdida**: Se emplea el **Error Cuadrático Medio (MSE)** como métrica de evaluación.
- **Entrenamiento de la Red Neuronal**: Implementación del algoritmo de **Backpropagation** y **Gradiente Descendiente**.
- **Visualización**: Se utilizan gráficos de `matplotlib` para representar los datos y los resultados del entrenamiento.

El código está organizado en dos archivos principales:

📌 **`src/code.py`** → Contiene la lógica de la red neuronal: generación de datos, funciones de activación, cálculo de pérdida y entrenamiento.

📌 **`main.py`** → Punto de entrada del programa, que ejecuta el entrenamiento de la red neuronal.

---

## 🛠️ Requisitos

Este proyecto fue desarrollado utilizando **Python 3.10**, pero es compatible con versiones de **Python 3.7 o superiores**.

Las siguientes bibliotecas son necesarias:

```bash
pip install numpy matplotlib scikit-learn
```

---

## 🚀 Clonar el Repositorio

Para obtener una copia local del proyecto, sigue estos pasos:

1. **Abrir una terminal o línea de comandos** en tu computadora.
2. **Navegar a la carpeta donde deseas clonar el repositorio**, por ejemplo:
   ```bash
   cd ~/mis-proyectos
   ```
3. **Clonar el repositorio** ejecutando el siguiente comando:
   ```bash
   git clone https://github.com/DemianM21/Neuronal_network_numpy_DemianMundo.git
   ```
4. **Acceder al directorio del proyecto**:
   ```bash
   cd Neuronal_network_numpy_DemianMundo
   ```
5. (Opcional) **Crear y activar un entorno virtual** para mantener las dependencias aisladas:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En macOS y Linux
   venv\Scripts\activate    # En Windows
   ```
6. **Instalar las dependencias del proyecto**:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Ejecución del Proyecto

Para entrenar la red neuronal y visualizar los resultados, ejecuta:

```bash
python main.py
```

Esto mostrará una gráfica con los datos y el resultado del entrenamiento.

---

## 📂 Estructura del Proyecto

```plaintext
Neuronal_network_numpy_DemianMundo/
│
├── src/
│   ├── code.py          # Lógica de la red neuronal
│
├── main.py              # Punto de entrada del programa
├── README.md            # Documentación del proyecto
└── requirements.txt     # Dependencias del proyecto (opcional)
```