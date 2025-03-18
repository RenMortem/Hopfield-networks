import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([[1], [-1], [-1], [1]])
x1_mod = np.array([[1], [-1], [-1], [-1]])
x2_mod = np.array([[1], [1], [-1], [-1]])  # Si más de la mitad de los píxeles están distorsionados, la imagen restaurada se invierte.
u1 = 0.25 #bias

# Matriz de pesos
W1 = (1 / len(x1)) * (x1 @ x1.T)

# Función Perceptrón
def perceptron(x, x_mod, W, u):
    b = np.zeros(4)  # Inicializar b dentro de la función
    x_out = np.zeros(4, dtype=int)  # x_out para almacenar la salida

    for i in range(4):
        for j in range(4):
            b[i] += W[i, j] * x_mod[j, 0] - u
            if b[i] == 0:
                b[i] = 1
        x_out[i] = np.sign(b[i])

    return x_out, b  # Devolvemos tanto la salida como b

# Ejecutar el perceptrón
x_restaurado, b_valores = perceptron(x1, x2_mod, W1, u1)

# Convertir a forma 2x2
imagen1 = x1.flatten().reshape(2, 2)
imagen2 = x1_mod.flatten().reshape(2, 2)
imagen3 = x_restaurado.reshape(2, 2)  # Mostrar la imagen restaurada

# Mostrar imágenes
plt.figure(figsize=(6, 3))

plt.subplot(1, 3, 1)
plt.imshow(imagen1, cmap='gray', vmin=-1, vmax=1)
plt.title("Imagen Original")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(imagen2, cmap='gray', vmin=-1, vmax=1)
plt.title("Imagen Modificada")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(imagen3, cmap='gray', vmin=-1, vmax=1)
plt.title("Imagen Restaurada")
plt.axis('off')

# Mostrar resultados
print("Matriz de pesos W1:")
print(W1)
print("\nValores de b:")
print(b_valores)
print("\nImagen restaurada (x_restaurado):")
print(x_restaurado)

plt.show()
