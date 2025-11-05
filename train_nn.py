#Simulacion de 2 entradas nuevas y un salida mas
import numpy as np

# --- Funciones de Activación ---
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_derivada(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivada(x):
    return 1.0 - x**2

# --- Clase NeuralNetwork (Tu Código) ---
class NeuralNetwork:

    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_derivada
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_derivada

        # inicializo los pesos
        self.weights = []
        self.deltas = []
        # capas = [4, 3, 5]
        # asigno valores aleatorios a capa de entrada y capa oculta (incluyendo Bias +1)
        for i in range(1, len(layers) - 1):
            r = 2 * np.random.random((layers[i-1] + 1, layers[i] + 1)) - 1
            self.weights.append(r)
        # asigno aleatorios a capa de salida (incluyendo Bias +1)
        r = 2 * np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.03, epochs=60001):
        # Agrego columna de unos (Bias)
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]] 

            # Forward Propagation
            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)
            
            # Cálculo de error y Backpropagation
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])] 
            
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))
            self.deltas.append(deltas)

            deltas.reverse() # Invertir el orden de los deltas para la actualización

            # Actualización de pesos
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

            if k % 20000 == 0: 
              print('epochs:', k)

    def predict(self, x): 
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

    def print_weights(self):
        print("LISTADO PESOS DE CONEXIONES")
        for i in range(len(self.weights)):
            print(f"\n--- PESOS DE CAPA {i+1} (Incluye Bias) ---")
            print(self.weights[i])

    def get_weights(self):
        return self.weights
    
    def get_deltas(self):
        return self.deltas


# --- DATOS DE ENTRENAMIENTO MODIFICADOS (4 Entradas, 5 Salidas) ---
# Entradas (X_new): [E1:Distancia, E2:Dirección, E3:Luz Der, E4:Luz Izq]
X_new = np.array([
    [-1, 0, -1, -1],  # Caso 1: Avanzar normal
    [ 1, 0, -1, -1],  # Caso 2: Peligro Cerca -> Retroceder + Alarma
    [-1, 0,  1, -1],  # Caso 3: Línea Der -> Avanzar + Alarma
    [-1, 0,  1,  1],  # Caso 4: Ambas Líneas -> Retroceder + Alarma
    [ 1,  1, -1, -1], # Caso 5: Obst. Cerca Izq -> Giro Der. + Alarma
    [ 1, -1, -1, -1], # Caso 6: Obst. Cerca Der -> Giro Izq. + Alarma
    [ 0,  0,  1,  1], # Caso 7: Obst. Medio y Líneas -> Retroceder
    [-1, 0, -1, -1],  # Caso 8: Avanzar normal
    [-1, 0,  1, -1],  # Caso 9: Línea Der -> Giro Der. + Alarma (Reorientar)
    [-1, 0, -1,  1],  # Caso 10: Línea Izq -> Giro Izq. + Alarma (Reorientar)
    [ 1, 0, -1, -1],  # Caso 11: Obst. Cerca -> Retroceder
    [ 0, 0, -1, -1]   # Caso 12: Avanzar normal
])

# Salidas (y_new): [S1, S2, S3, S4, S5:Alarma]
y_new = np.array([
    [1, 0, 0, 1, 0],  # Avanzar, Alarma Off
    [0, 1, 1, 0, 1],  # Retroceder, Alarma On
    [1, 0, 0, 1, 1],  # Avanzar, Alarma On
    [0, 1, 1, 0, 1],  # Retroceder, Alarma On
    [0, 1, 0, 1, 1],  # Giro Der., Alarma On
    [1, 0, 1, 0, 1],  # Giro Izq., Alarma On
    [0, 1, 1, 0, 0],  # Retroceder, Alarma Off
    [1, 0, 0, 1, 0],  # Avanzar, Alarma Off
    [0, 1, 0, 1, 1],  # Giro Der., Alarma On
    [1, 0, 1, 0, 1],  # Giro Izq., Alarma On
    [0, 1, 1, 0, 0],  # Retroceder, Alarma Off
    [1, 0, 0, 1, 0]   # Avanzar, Alarma Off
])

# --- EJECUCIÓN DEL ENTRENAMIENTO ---
print("--- INICIANDO ENTRENAMIENTO DE RED 4-3-5 ---")
# Inicializar la red: [4 entradas, 3 ocultas, 5 salidas]
nn = NeuralNetwork([4, 3, 5], activation='tanh') 
nn.fit(X_new, y_new, learning_rate=0.03, epochs=60001) 

print("\n\n--- PESOS FINALES PARA COPIAR A ARDUINO (¡LA INTELIGENCIA!) ---")
nn.print_weights()

