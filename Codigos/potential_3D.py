import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata

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
inertia_tensor_cm = inertia_tensor_origin - mass * (np.trace(d_outer) * I_identity - d_outer)

# Suponiendo que 'inertia_tensor' es tu tensor de inercia 3x3:
eigvals, eigvecs = np.linalg.eigh(inertia_tensor_cm)

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

def compute_theta_phi(vectors, eigvecs):
    """
    Calcula los ángulos theta y phi de cada vector usando las definiciones:
      theta = arccos(z / r)
      phi   = sgn(y) * arccos(x / sqrt(x^2 + y^2))
    donde:
      - e2 = eigvecs[:, 1] es el eje 'z'
      - e1 = eigvecs[:, 0] es el eje 'x'
      - e3 = e2 x e1       es el eje 'y'
    
    Parámetros:
      vectors : np.ndarray de shape (N, 3)
          Vectores en coordenadas cartesianas (originales).
      eigvecs : np.ndarray de shape (3, 3)
          Matriz de eigenvectores, donde cada columna es un eigenvector.
          - eigvecs[:, 1] = e2
          - eigvecs[:, 0] = e1

    Retorna:
      theta : np.ndarray de shape (N,)
          Ángulo polar (definición de la imagen).
      phi   : np.ndarray de shape (N,)
          Ángulo azimutal (definición de la imagen).
    """
    # 1. Normalizamos e2 y e1
    e2 = eigvecs[:, 1] / np.linalg.norm(eigvecs[:, 1])  # Eje "z"
    e1 = eigvecs[:, 0] / np.linalg.norm(eigvecs[:, 0])  # Eje "x"
    e3 = eigvecs[:, 2] / np.linalg.norm(eigvecs[:, 2])  # Eje "y"
    
    # Para cada vector, calculamos sus componentes en el sistema (e1, e2, e3)
    #   x' = v · e1
    #   y' = v · e3
    #   z' = v · e2
    x_comp = np.dot(vectors, e1)
    y_comp = np.dot(vectors, e3)
    z_comp = np.dot(vectors, e2)
    
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
theta, phi = compute_theta_phi(vectors, eigvecs)

# --- INTERPOLACIÓN PARA GRAFICAR COMO SUPERFICIE ---
# Se asume que theta está en [0, π] y phi en [-π, π].
theta_lin = np.linspace(0, np.pi, 500)
phi_lin = np.linspace(-np.pi, np.pi, 1000)
theta_grid, phi_grid = np.meshgrid(theta_lin, phi_lin)

# Interpolar los valores de alturas (dispersos) sobre la malla regular.
points = np.column_stack((theta, phi))
grid_alturas = griddata(points, alturas, (theta_grid, phi_grid), method='linear')

# Definir los límites de Z
z_min, z_max = min(alturas), max(alturas) + 1

# Filtrar valores fuera del rango deseado
grid_alturas[(grid_alturas < z_min) | (grid_alturas > z_max)] = np.nan  # Convertirlos en NaN

# --- GRAFICAR LA SUPERFICIE ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficar sin los valores fuera de zlim
surf = ax.plot_surface(theta_grid, phi_grid, grid_alturas, cmap='viridis', edgecolor='none', alpha=0.8)

ax.set_xlabel("θ")
ax.set_ylabel("φ")
ax.set_zlabel("Altura")
ax.set_zlim(z_min, z_max)
ax.set_title("Superficie de h(θ, φ) (Altura del centro de masa)")
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()
