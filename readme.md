# Neural Network with NumPy
## Demian Mundo

En este proyecto se implementa una red neuronal desde cero utilizando **NumPy** para la clasificación de datos generados sintéticamente. El objetivo es demostrar cómo construir y entrenar redes neuronales sin depender de módulos como **TensorFlow** o **PyTorch**.

Se incluyen funciones para la inicialización de parámetros, funciones de activación (**ReLU y Sigmoide**), cálculo del error mediante el **Error Cuadrático Medio (MSE)** y actualización de pesos utilizando el **Gradiente Descendiente**.

El modelo se entrena durante varias iteraciones y se visualizan los datos, lo que permite observar el comportamiento de aprendizaje y cómo disminuye el error, permitiendo ajustar los pesos para mejorar la precisión en la clasificación.

---

## 📌 Descripción del Proyecto

El proyecto consta de los siguientes componentes principales:

- **Generación de Datos**: Se crean datos sintéticos utilizando `make_gaussian_quantiles` de `sklearn.datasets` para generar un conjunto de datos aleatorios de clasificación binaria.
- **Funciones de Activación**: Implementación de las funciones **ReLU** y **Sigmoid**, junto con sus derivadas.
- **Función de Pérdida**: Se emplea el **Error Cuadrático Medio (MSE)** como métrica de evaluación.
- **Entrenamiento de la Red Neuronal**: Implementación del algoritmo de **Backpropagation** y **Gradiente Descendiente**.
- **Visualización**: Se utilizan gráficos de `matplotlib` para representar los datos y los resultados del entrenamiento.

El código está organizado en dos archivos principales:

📌 **`Neural/code.py`** → Contiene la lógica de la red neuronal: generación de datos, funciones de activación, cálculo de pérdida y entrenamiento.

📌 **`main.py`** → Punto de entrada del programa, que ejecuta el entrenamiento de la red neuronal, importando el módulo personalizado `code.py`.

---

## 🛠️ Requisitos

Este proyecto fue desarrollado utilizando **Python 3.10**, pero se puede utilizar con cualquier versión de python que sea compatible con las dependencias dentro de 'requirements.txt'

Las siguientes bibliotecas son necesarias:

```bash
pip install numpy matplotlib scikit-learn
```

---

## 🚀 Clonar el Repositorio

Para obtener una copia local del proyecto, sigue estos pasos:

1. **Abrir una terminal o línea de comandos** en tu computadora.
2. **Navegar a la carpeta donde deseas clonar el repositorio**:
   ```bash
   cd ~/mis-proyectos
   ```
3. **Clonar el repositorio** ejecutando el siguiente comando:
   ```bash
   git clone https://github.com/DemianM21/Neural_network_numpy_DemianMundo.git
   ```
4. **Acceder al directorio del proyecto**:
   ```bash
   cd Neural_network_numpy_DemianMundo
   ```
5. **Crear y activar un entorno virtual**:
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
# Implementación de una Red Neuronal desde Cero

Este proyecto implementa una red neuronal desde cero en NumPy para resolver un problema de clasificación binaria con datos sintéticos.

## 1. Importación de Librerías
Se importan las librerías necesarias para generar datos, realizar cálculos numéricos y visualizar los resultados.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles
```

## 2. Creación del Dataset
Se genera un conjunto de datos sintético con dos clases usando distribuciones gaussianas.

```python
def create_dataset(N=1000):
    gaussian_quantiles = make_gaussian_quantiles(
        mean=None,
        cov=0.1,
        n_samples=N,
        n_features=2,
        n_classes=2,
        shuffle=True,
        random_state=None
    )
    X, Y = gaussian_quantiles
    Y = Y[:, np.newaxis]  # Convertir a matriz columna
    return X, Y
```

## 3. Funciones de Activación
Se definen las funciones de activación Sigmoide y ReLU, junto con sus derivadas.

```python
def sigmoid(x, derivate=False):
    if derivate:
        return np.exp(-x) / (np.exp(-x) + 1)**2
    else:
        return 1 / (1 + np.exp(-x))

def relu(x, derivate=False):
    if derivate:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    else:
        return np.maximum(0, x)
```

## 4. Función de Pérdida
Se implementa el error cuadrático medio (MSE) y su derivada.

```python
def mse(y, y_hat, derivate=False):
    if derivate:
        return (y_hat - y)
    else:
        return np.mean((y_hat - y)**2)
```

## 5. Inicialización de Pesos y Sesgos
Se inicializan los pesos y sesgos de la red neuronal de manera aleatoria.

```python
def initialize_parameters_deep(layers_dims):
    parameters = {}
    L = len(layers_dims)
    for l in range(0, L-1):
        parameters['W' + str(l+1)] = (np.random.rand(layers_dims[l], layers_dims[l+1]) * 2) - 1
        parameters['b' + str(l+1)] = (np.random.rand(1, layers_dims[l+1]) * 2) - 1
    return parameters
```

## 6. Entrenamiento de la Red Neuronal
Se implementa la propagación hacia adelante y hacia atrás para actualizar los pesos mediante descenso de gradiente.

```python
def train(x_data, y_data, learning_rate, params, training=True):
    params['A0'] = x_data

    # Propagación hacia adelante
    params['Z1'] = np.matmul(params['A0'], params['W1']) + params['b1']
    params['A1'] = relu(params['Z1'])

    params['Z2'] = np.matmul(params['A1'], params['W2']) + params['b2']
    params['A2'] = relu(params['Z2'])

    params['Z3'] = np.matmul(params['A2'], params['W3']) + params['b3']
    params['A3'] = sigmoid(params['Z3'])

    output = params['A3']

    if training:
        # Backpropagation
        params['dZ3'] = mse(y_data, output, True) * sigmoid(params['A3'], True)
        params['dW3'] = np.matmul(params['A2'].T, params['dZ3'])

        params['dZ2'] = np.matmul(params['dZ3'], params['W3'].T) * relu(params['A2'], True)
        params['dW2'] = np.matmul(params['A1'].T, params['dZ2'])

        params['dZ1'] = np.matmul(params['dZ2'], params['W2'].T) * relu(params['A1'], True)
        params['dW1'] = np.matmul(params['A0'].T, params['dZ1'])

        # Actualización de parámetros
        params['W3'] -= params['dW3'] * learning_rate
        params['W2'] -= params['dW2'] * learning_rate
        params['W1'] -= params['dW1'] * learning_rate

        params['b3'] -= np.mean(params['dW3'], axis=0, keepdims=True) * learning_rate
        params['b2'] -= np.mean(params['dW2'], axis=0, keepdims=True) * learning_rate
        params['b1'] -= np.mean(params['dW1'], axis=0, keepdims=True) * learning_rate

    return output
```

## 7. Entrenamiento del Modelo
Se entrena la red neuronal sobre los datos generados y se visualizan los resultados.

```python
def train_model():
    X, Y = create_dataset()
    layers_dims = [2, 6, 10, 1]
    params = initialize_parameters_deep(layers_dims)
    error = []

    for _ in range(50000):
        output = train(X, Y, 0.001, params)
        if _ % 50 == 0:
            print(mse(Y, output))
            error.append(mse(Y, output))

    # Graficar los datos
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.show()
```


---
## 📂 Estructura del Proyecto

```plaintext
Neural_network_numpy_DemianMundo/
│
├── Neural/
│   ├── code.py          # Lógica de la red neuronal
│
├── main.py              # Punto de entrada del programa
├── README.md            # Documentación del proyecto
└── requirements.txt     # Dependencias del proyecto (opcional)
```