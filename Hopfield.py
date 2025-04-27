import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([[1], [-1], [-1], [1]])
x1_mod = np.array([[1], [1], [1], [-1]])
x2_mod = np.array([[1], [1], [-1], [-1]])  # Si más de la mitad de los píxeles están distorsionados, la imagen restaurada se invierte.
u1 = 0.25 #bias

# Matriz de pesos
W1 = (1 / len(x1)) * (x1 @ x1.T)

#Función de Energía
def energia(x, W):
    H : float = 0 #Función de Energía
    for i in range(4):
        for j in range(4):
            H += -0.5 * W[i, j] * x[i] * x[j]
    
    return H

# Función Perceptrón
def perceptron(x_mod, W, u):
    b = np.zeros(4)  # Inicializar b dentro de la función
    x_out = np.zeros(4, dtype=int)  # x_out para almacenar la salida
    energias = np.zeros(4)  # Guardar energía inicial

    for i in range(4):
        for j in range(4):
            b[i] += W[i, j] * x_mod[j, 0] - u
            if b[i] == 0:
                b[i] = 1
        x_out[i] = np.sign(b[i])
        energias[i] = energia(x_out, W)

    return x_out, b, energias # Devolvemos tanto la salida como b

# Ejecutar el perceptrón
x_restaurado, b_valores, energ_valores = perceptron(x2_mod, W1, u1)

# Convertir a forma 2x2
imagen1 = x1.flatten().reshape(2, 2)
imagen2 = x1_mod.flatten().reshape(2, 2)
imagen3 = x_restaurado.reshape(2, 2)  # Mostrar la imagen restaurada

# Mostrar imágenes
plt.figure(figsize=(6, 4))

plt.subplot(1, 4, 1)
plt.imshow(imagen1, cmap='gray', vmin=-1, vmax=1)
plt.title("Imagen Original")
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(imagen2, cmap='gray', vmin=-1, vmax=1)
plt.title("Imagen Modificada")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(imagen3, cmap='gray', vmin=-1, vmax=1)
plt.title("Imagen Restaurada")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.bar(range(len(energ_valores)), energ_valores)
plt.title("Función de Energía")
plt.xlabel("Iteración")
plt.ylabel("Energía")
plt.grid('true')
plt.axis('on')

# Mostrar resultados
print("Matriz de pesos W1:")
print(W1)
print("\nValores de b:")
print(b_valores)
print("\nImagen restaurada (x_restaurado):")
print(x_restaurado)
print(x1)

plt.tight_layout()
plt.show()
