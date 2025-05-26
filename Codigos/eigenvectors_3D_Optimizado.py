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

# Calcular el centro de masa y el tensor de inercia
centroid = mesh.center_mass
# (Opcional) avisar si la malla no es hermética
if not mesh.is_watertight:
    print("⚠️ Atención: la malla no es hermética — los resultados pueden no ser fiables.")

# 1) Tensor de inercia con respecto al CM, alineado en ejes globales
I_cm = mesh.moment_inertia
# 📖 mesh.moment_inertia ya está centrado en el centro de masa y alineado con ejes cartesianos :contentReference[oaicite:0]{index=0}

# 2) Momentos principales de inercia (vector diagonal)
principal_moments = mesh.principal_inertia_components
# 📖 moment principal order corresponde a mesh.principal_inertia_vectors :contentReference[oaicite:1]{index=1}

# 3) Ejes principales de rotación (vectores propios)
principal_axes = mesh.principal_inertia_vectors
# 📖 devuelve tres vectores unitarios ordenados según principal_inertia_components :contentReference[oaicite:2]{index=2}

# Resultados
print("Tensor de inercia (centro de masa, no desplazado):\n", I_cm)
print("Momentos principales de inercia:", principal_moments)
print("Ejes principales (columnas):\n", principal_axes)

# Verificación rápida: principal_axes^T · I_cm · principal_axes debe ser diagonal
I_diag_check = principal_axes.T @ I_cm @ principal_axes
print("I_alineado con ejes principales (debe ser diagonal):\n", I_diag_check)

# Rotación y preparación de malla
rotation_matrix = np.array([[-1, 0, 0],
                            [ 0, 0, 1],
                            [ 0, 1, 0]])

# Rotar vértices
rotated_vertices = (mesh.vertices - centroid) @ rotation_matrix

# Crear malla en formato PyVista
faces_pv = np.hstack([[3] + list(face) for face in mesh.faces])  # Cada cara con 3 vértices
pv_mesh = pv.PolyData(rotated_vertices, faces_pv)

# Crear plotter
plotter = pv.Plotter()

# Añadir la malla rotada
plotter.add_mesh(pv_mesh, color='lightgray', show_edges=True, opacity=0.6)

# Rotar vectores propios
rotated_eigvecs = principal_axes @ rotation_matrix

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
        ("Eje 1", "magenta"),
        ("Eje 2", "orange"),
        ("Eje 3", "cyan"),
    ],
    bcolor='white',
    border=True,
    size=(0.2, 0.1),
    loc='lower right'
)

# Mostrar visualización
plotter.show()

# Mostrar vectores rotados por consola
print("vectores propios rotados:\n", rotated_eigvecs)