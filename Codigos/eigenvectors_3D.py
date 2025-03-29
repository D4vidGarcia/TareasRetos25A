import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

stl = int(3)  # Seleccionar el archivo STL a usar

# Diccionario con las rutas de los archivos STL
stl_paths = {
    # Gombocs
    1: r"C:\Users\david\Material Universidad\Retos\STL\Monostatic_Body_Release.STL", # Gomboc 1 con resolucion media
    2: r"C:\Users\david\Material Universidad\Retos\STL\GollyPoly.stl", # Gomboc 2 con resolucion alta
    # Archivos para confirmar el funcionamiento del algoritmo
    3: r"C:\Users\david\Material Universidad\Retos\STL\20000_polygon_sphere_100mm.STL", # Esfera
    4: r"C:\Users\David\Material Universidad\Retos\STL\Cilindro_3D.stl", # Cilindro
    5: r"C:\Users\David\Material Universidad\Retos\STL\Cubo_Prueba_20x20.stl", # Cubo
    }

# Usar el archivo correspondiente dependiendo de prueba
mesh = trimesh.load_mesh(stl_paths[stl])

# Calcular el volumen, centro de masa y el tensor de inercia
volume = mesh.volume
centroid = mesh.center_mass
inertia_tensor_origin = mesh.moment_inertia  # Si se desea considerar la densidad, se multiplica por la densidad

# Asumimos densidad uniforme, entonces la masa es proporcional al volumen (usaremos volumen como masa)
mass = volume  # Se multiplica por la densidad

# Vector de traslación del centro de masa
d = centroid.reshape(3, 1)  # Convertir a columna para cálculo

# Matriz identidad 3x3
I_identity = np.eye(3)

# Producto exterior de d (d * d^T)
d_outer = np.dot(d, d.T)

# Aplicar el teorema de Steiner
inertia_tensor_cm = inertia_tensor_origin - mass * (np.trace(d_outer) * I_identity - d_outer)

# Suponiendo que 'inertia_tensor' es tu tensor de inercia 3x3:
eigvals, eigvecs = np.linalg.eigh(inertia_tensor_cm)

# Crear figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Graficar la malla STL
ax.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces], alpha=0.5, edgecolor="k"))

# Graficar los vectores propios en el centro de masa
colors = ['r', 'g', 'b']  # Colores para los ejes principales
scale = np.max(eigvals) ** 0.5  # Escalar los vectores para visualización

for i in range(3):  # Para cada eje principal
    ax.quiver(centroid[0], centroid[1], centroid[2], 
              eigvecs[0, i] * scale, eigvecs[1, i] * scale, eigvecs[2, i] * scale, 
              color=colors[i], linewidth=2, label=f'Eigenvector {i+1}')

# Configurar la visualización
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.auto_scale_xyz(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2])

# Mostrar leyenda y gráfico
ax.legend()
plt.show()
