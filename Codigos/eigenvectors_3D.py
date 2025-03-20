import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Cargar archivo STL
mesh = trimesh.load_mesh(r"C:\Users\David\Documents\Material Universidad\Retos\STL\Monostatic_Body_Release.STL")

# Calcular el volumen, centro de masa y el tensor de inercia
volume = mesh.volume
centroid = mesh.center_mass
inertia_tensor = mesh.moment_inertia  # Si deseas considerar densidad, multiplica por la densidad

print("Volumen:", volume)
print("Centro de masa:", centroid)

# Suponiendo que 'inertia_tensor' es tu tensor de inercia 3x3:
eigvals, eigvecs = np.linalg.eigh(inertia_tensor)

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