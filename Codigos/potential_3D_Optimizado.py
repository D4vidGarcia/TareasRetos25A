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

# Obtener los vértices y caras de la malla
vertices = mesh.vertices
faces = mesh.faces

# Definicion de la posicion de los vertices de cada triangulo
V1 = vertices[faces[:, 0]]
V2 = vertices[faces[:, 1]]
V3 = vertices[faces[:, 2]]

# Producto cruzado de los lados del triángulo
normals = np.cross(V2 - V1, V3 - V1)

# Normalizar los vectores normales
normals /= np.linalg.norm(normals, axis=1, keepdims=True)

midpoints = (V1 + V2 + V3) / 3  # Puntos medios de los triángulos

# Calcular los vectores desde el centro de masa a cada punto medio
vectors = midpoints - centroid

def altura_centro_masa(vectors, normals):
    """
    Calcula la proyección (producto punto) de cada vector sobre su correspondiente normal.
    Parámetros:
      vectors : np.ndarray de shape (N, 3)
          Vectores desde el centro de masa hasta los centros de los triángulos.
      normals : np.ndarray de shape (N, 3)
          Vectores normales (normalizados) de cada triángulo.
    Retorna:
      alturas : np.ndarray de shape (N,)
          Producto punto de cada par, que se interpretará como la "altura".
    """
    alturas = np.sum(vectors * normals, axis=1)
    return alturas

# Calculamos las alturas (proyecciones)
alturas = altura_centro_masa(vectors, normals)
print("Alturas (proyecciones del vector desde el centro de masa):\n", alturas)

def compute_theta_phi(vectors, rotation_matrix):
    """
    Calcula los ángulos theta y phi de cada vector usando las definiciones:
      theta = arccos(z / r)
      phi   = sgn(y) * arccos(x / sqrt(x^2 + y^2))
    donde (x, y, z) son las componentes del vector en el sistema de coordenadas (e1, e2, e3).
    
    Parámetros:
      vectors : np.ndarray de shape (N, 3)
          Vectores en coordenadas cartesianas (originales).
      eigvecs : np.ndarray de shape (3, 3)
          Matriz de eigenvectores, donde cada columna es un eigenvector.

    Retorna:
      theta : np.ndarray de shape (N,)
          Ángulo polar (definición de la imagen).
      phi   : np.ndarray de shape (N,)
          Ángulo azimutal (definición de la imagen).
    """
    
    # Para cada vector, calculamos sus componentes en el sistema (e1, e2, e3)
    #   x' = v · e1
    #   y' = v · e3
    #   z' = v · e2
    x_comp = np.dot(vectors, rotation_matrix[:, 0])  # Proyección en el eje "x"
    y_comp = np.dot(vectors, rotation_matrix[:, 1])  # Proyección en el eje "y"
    z_comp = np.dot(vectors, rotation_matrix[:, 2])  # Proyección en el eje "z"
    
    # Norma de cada vector en este nuevo sistema
    r = np.sqrt(x_comp**2 + y_comp**2 + z_comp**2)
    # Para evitar división por cero
    r_safe = np.where(r == 0, 1, r)
    
    # --------------------
    # Cálculo de theta
    # --------------------
    # theta = arccos(z / r)
    # Clipeamos el argumento de arccos a [-1,1] para evitar errores numéricos
    cos_theta = z_comp / r_safe
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    # --------------------
    # Cálculo de phi
    # --------------------
    # phi = sgn(y) * arccos(x / sqrt(x^2 + y^2))
    # Definimos el denominador para x^2 + y^2
    xy_norm = np.sqrt(x_comp**2 + y_comp**2)
    xy_norm_safe = np.where(xy_norm == 0, 1, xy_norm)  # evitar 0 en la división
    
    # arccos(x / sqrt(x^2 + y^2))
    arg_phi = x_comp / xy_norm_safe
    arg_phi = np.clip(arg_phi, -1.0, 1.0)  # clip para evitar fuera de dominio
    base_phi = np.arccos(arg_phi)
    
    # sgn(y)
    sgn_y = np.sign(y_comp)
    
    # Producto final
    phi = sgn_y * base_phi
    
    return theta, phi

# Parametrización
theta, phi = compute_theta_phi(vectors, rotation_matrix)

# Crear puntos 2D para la triangulación (θ, φ)
points_2d = np.column_stack((theta, phi))
cloud_2d = pv.PolyData(np.column_stack((theta, phi, np.zeros_like(theta))))  # plano 2D

# Generar superficie en 2D
surface = cloud_2d.delaunay_2d()

# Reemplazar Z con altura original
points_surface = surface.points
points_surface[:, 2] = alturas
surface.points = points_surface

# Configurar scalar bar en vertical (derecha)
scalar_bar_args = {
    "title": "Altura",
    "vertical": True,
    "position_x": 0.85,  # más a la derecha
    "position_y": 0.1,
    "width": 0.04,
    "height": 0.8,
    "title_font_size": 12,
    "label_font_size": 10
}

# Crear plotter
plotter = pv.Plotter()

plotter.add_mesh(
    surface,
    scalars=alturas,
    cmap='viridis',
    show_edges=False,
    smooth_shading=True,
    lighting=True,
    ambient=0.3,
    specular=0.4,
    specular_power=15,
    scalar_bar_args=scalar_bar_args
)

plotter.set_scale(1, 1, 0.5)
plotter.add_axes()
plotter.show_bounds(
    xlabel='θ',
    ylabel='φ',
    zlabel='Altura',
    grid='front',
    location='outer'
)

# Título en overlay
plotter.add_text("Superficie de altura", position="upper_edge", font_size=14)

plotter.show()
