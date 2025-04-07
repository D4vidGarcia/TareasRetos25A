import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyvista as pv

stl = int(1)  # Seleccionar el archivo STL a usar

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
inertia_tensor_cm = inertia_tensor_origin + mass * (np.trace(d_outer) * I_identity - d_outer)

# Suponiendo que 'inertia_tensor' es tu tensor de inercia 3x3:
eigvals, eigvecs = np.linalg.eigh(inertia_tensor_cm)

# Rotación y preparación de malla
rotation_matrix = np.array([[-1, 0, 0],
                            [ 0, 0, 1],
                            [ 0, 1, 0]])

# Rotar vértices
rotated_vertices = (rotation_matrix.T @ (mesh.vertices - centroid).T).T

# Crear malla en formato PyVista
faces_pv = np.hstack([[3] + list(face) for face in mesh.faces])  # Cada cara con 3 vértices
pv_mesh = pv.PolyData(rotated_vertices, faces_pv)

# Crear plotter
plotter = pv.Plotter()

# Añadir la malla rotada
plotter.add_mesh(pv_mesh, color='lightgray', show_edges=True, opacity=0.6)

# Rotar vectores propios
rotated_eigvecs = (rotation_matrix.T @ eigvecs.T).T

# Dibujar flechas para los ejes de la matriz identidad
colors_identity = ['magenta', 'orange', 'cyan']
line_length = 0.5 * np.linalg.norm(pv_mesh.length)  # ajustar según el tamaño de la malla

for i in range(3):
    vec = rotated_eigvecs[:, i]
    arrow = pv.Arrow(start=(0, 0, 0), direction=vec, scale=line_length)
    plotter.add_mesh(arrow, color=colors_identity[i], label=f'Eje {i+1} (Identidad)')

# Configurar visualización
plotter.set_background('white')
plotter.add_axes()
plotter.show_bounds(grid='front', location='outer')

# Título más separado de la malla (arriba centrado)
plotter.add_text("Malla rotada en punto de equilibrio", position='upper_edge', font_size=12, color='black')

# Mostrar leyenda en parte inferior derecha
plotter.add_legend(
    labels=[
        ("Eje 1 (Identidad)", "magenta"),
        ("Eje 2 (Identidad)", "orange"),
        ("Eje 3 (Identidad)", "cyan"),
    ],
    bcolor='white',
    border=True,
    size=(0.2, 0.1),
    loc='lower right'
)

# Mostrar visualización
plotter.show()