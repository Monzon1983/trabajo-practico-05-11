#  Trabajo Práctico: Coche Autónomo con Red Neuronal (NN 4-3-5)

##  Resumen de Modificación y Resultados

Este trabajo presenta la modificación y el entrenamiento de nuestra Red Neuronal para el control del coche robot.
El entrenamiento se realizó en Python (ver `train_nn.py`) y los pesos finales fueron transferidos al código de Arduino (ver `robot_control.ino`).

---

## Arquitectura de la Red Neuronal Modificada

Realizamos un reajuste de la arquitectura para acomodar las nuevas entradas y salidas, resultando en una estructura [4, 3, 5]:

| Capa | Neuronas | Variables | Función |
| :--- | :---: | :--- | :--- |
| **Entrada** | **4** | E1, E2, E3, E4 | Lectura de Sensores (Distancia y Luz) |
| **Oculta** | 3 | H1, H2, H3 | Procesamiento de Datos |
| **Salida** | **5** | S1-S4, S5 | Control de Motores y Alarma |

---

##  Tabla de Verdad Consolidada (Matriz de Entrenamiento)

La Lógica de Control del coche está definida por la siguiente Tabla de Verdad Consolidada, que sirvió como matriz de entrenamiento ($X$ e $y$)

| Variable | Descripción | Valores (Normalizados) |
| :---: | :---: | :---: |
| **E1** | Distancia | -1 (Lejos), 0 (Medio), 1 (Cerca) |
| **E3, E4** | Luminosidad (Línea) | -1 (Oscuro), 1 (Claro) |
| **S5** | Alarma/Luz | 0 (Off), 1 (On) |

| Caso | E1 | E2 | E3 | E4 | S1 | S2 | S3 | S4 | **S5** | **Acción Clave** |
| :---: | :---: | :---: | :---: | :---: | :-: | :-: | :-: | :-: | :---: | :--- |
| 1 | -1 | 0 | -1 | -1 | 1 | 0 | 0 | 1 | 0 | Avanzar normal |
| 2 | 1 | 0 | -1 | -1 | 0 | 1 | 1 | 0 | 1 | Retroceder + Alarma |
| 3 | -1 | 0 | 1 | -1 | 1 | 0 | 0 | 1 | 1 | Avanzar + Alarma (Detecta Línea) |
| 4 | -1 | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | Retroceder + Alarma (Ambas Líneas) |
| 5 | 1 | 1 | -1 | -1 | 0 | 1 | 0 | 1 | 1 | Giro Der. + Alarma |
| 6 | 1 | -1 | -1 | -1 | 1 | 0 | 1 | 0 | 1 | Giro Izq. + Alarma |
| 7 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | Retroceder |
| 8 | -1 | 0 | -1 | -1 | 1 | 0 | 0 | 1 | 0 | Avanzar normal |
| 9 | -1 | 0 | 1 | -1 | 0 | 1 | 0 | 1 | 1 | Giro Der. + Alarma |
| 10 | -1 | 0 | -1 | 1 | 1 | 0 | 1 | 0 | 1 | Giro Izq. + Alarma |
| 11 | 1 | 0 | -1 | -1 | 0 | 1 | 1 | 0 | 0 | Retroceder |
| 12 | 0 | 0 | -1 | -1 | 1 | 0 | 0 | 1 | 0 | Avanzar normal |

---

##  Proceso de Entrenamiento y Obtención de Pesos (`train_nn.py`)

A continuación, se muestra el fragmento clave del código en Python donde se instancia la red con la nueva arquitectura y se ejecuta el entrenamiento para obtener las matrices de pesos finales:

```python
# INICIAR ENTRENAMIENTO
nn = NeuralNetwork([4, 3, 5], activation='tanh')
nn.fit(X_new, y_new, learning_rate=0.03, epochs=60001)

# Impresión de los pesos finales para su transferencia a Arduino:
nn.print_weights()


