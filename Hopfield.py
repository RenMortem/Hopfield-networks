import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([[1], [-1], [-1], [1]])
x1_mod = np.array([[1], [-1], [-1], [-1]])
x2_mod = np.array([[1], [1], [1], [-1]]) #Si  la más de la mitad de los pixeles están distorcionados, la imagen restaruada se invierte, para arreglarlo, hay que experimentar con valores de "u".
x1_T = x1.T
u = 0.25
b = np.zeros(4)
x2 = np.zeros(4, dtype=int)

W1 = (1 / len(x1)) * (x1 @ x1.T)

for i in range(4):
    for j in range(4):
        b[i] += W1[i, j] * x2_mod[j, 0] - u
        if b[i] == 0:
            b[i] = 1
    x2[i] = np.sign(b[i])
    
# Convertir a forma 2x2
imagen1 = x1.flatten().reshape(2, 2)
imagen2 = x2_mod.flatten().reshape(2, 2)
imagen3 = x2.flatten().reshape(2, 2)

# Mostrar ambas imágenes
plt.figure(figsize=(6, 3))

plt.subplot(1, 3, 1)  # Primera imagen
plt.imshow(imagen1, cmap='gray', vmin=-1, vmax=1)
plt.title("Imagen Original")
plt.axis('off')

plt.subplot(1, 3, 2)  # Segunda imagen
plt.imshow(imagen2, cmap='gray', vmin=-1, vmax=1)
plt.title("Imagen Modificada")
plt.axis('off')

plt.subplot(1, 3, 3)  # Segunda imagen
plt.imshow(imagen3, cmap='gray', vmin=-1, vmax=1)
plt.title("Imagen Restaurada")
plt.axis('off')

print(W1)
print(b)
print(x2)

plt.show()