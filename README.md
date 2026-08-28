# Microproyecto 1 — Paleta de colores con Machine Learning No Supervisado

Proyecto del curso **Machine Learning No Supervisado** (Maestría en Inteligencia Artificial).

## Objetivo

Desarrollar un método, basado en técnicas de agrupación (clustering), que permita extraer los tonos dominantes de una imagen y generar un muestrario (paleta) de los colores representativos presentes en ella. El caso de uso motivador es una herramienta automática de utilidad para diseñadores gráficos, directores de arte, pintores y creadores de contenido, entre otras aplicaciones (marketing, psicología, medicina, estudios ambientales).

## Dataset

Se usa **WikiArt** (dataset [`steubk/wikiart`](https://www.kaggle.com/datasets/steubk/wikiart) en Kaggle), un conjunto de obras de arte de distintos estilos/corrientes pictóricas.

- `data/classes.csv`: metadatos de ~80.000 obras, con columnas `filename, artist, genre, description, phash, width, height, genre_count, subset`. `genre` es una lista (como string) de estilos aplicables a cada obra.
- Las imágenes **no** se descargan completas al repo: se obtienen bajo demanda con [`kagglehub`](https://github.com/Kaggle/kagglehub) (`kagglehub.dataset_download`), que las cachea localmente fuera del repo (por defecto en `~/.cache/kagglehub`).

### Credenciales de Kaggle

`kagglehub` necesita credenciales de la API de Kaggle para descargar el dataset:

1. Crear una cuenta en [kaggle.com](https://www.kaggle.com) si no se tiene.
2. En *Account → API → Create New Token* descargar `kaggle.json`.
3. Colocarlo en `~/.kaggle/kaggle.json` (Linux/Mac) o `C:\Users\<usuario>\.kaggle\kaggle.json` (Windows), o exportar `KAGGLE_USERNAME` y `KAGGLE_KEY` como variables de entorno.

## Estructura del repositorio

```
Proyecto1MLNS/
├── data/
│   └── classes.csv                          # metadatos del dataset WikiArt
├── notebooks/
│   └── microproyecto1_paleta_colores.ipynb  # notebook principal del proyecto
├── reports/
│   └── (aquí va el .html exportado antes de la entrega final)
├── requirements.txt
└── README.md
```

## Instalación y configuración del entorno

### 1. Entorno virtual

El proyecto usa un entorno virtual en `venv/` (ignorado por git). Desde la raíz del repo:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1        # cmd: .\venv\Scripts\activate.bat  |  bash: source venv/Scripts/activate
python -m pip install -r requirements.txt
```

**Decisión — versión de Python.** El entorno se probó con Python 3.14.6. Las restricciones en `requirements.txt` se dejaron como cota inferior + cota de *major* (p. ej. `numpy>=1.26,<3.0`) en lugar de pines cerrados, porque los pines antiguos (`numpy<2.0`, `scikit-learn<1.7`, ...) no tienen *wheels* para Python 3.14 y fallan al compilar. Con estas cotas `pip` resuelve a `numpy 2.5`, `scikit-learn 1.9`, `opencv-python 4.14`, que cubren todo lo que el notebook importa. Con un Python anterior (3.11–3.12) el mismo archivo resuelve eligiendo versiones más viejas dentro del rango.

### 2. Kernel de Jupyter apuntando al venv

Para que el notebook se ejecute **siempre contra este entorno** (no contra el Python global ni otro venv), se registra el venv como kernel de Jupyter con nombre propio:

```powershell
.\venv\Scripts\python.exe -m ipykernel install --user --name proyecto1mlns --display-name "Python (Proyecto1MLNS)"
```

**Decisión — por qué un kernel con nombre.** Jupyter y VS Code descubren kernels de forma global; sin uno dedicado es fácil abrir el notebook con un intérprete que no tiene las dependencias, o que tiene otras versiones. Un kernel explícito (`Python (Proyecto1MLNS)`) elimina la ambigüedad y es reproducible por el compañero de equipo y por el corrector: clonar, crear el venv, correr el comando de arriba y seleccionar ese kernel.

Verificar que quedó registrado:

```powershell
.\venv\Scripts\python.exe -m jupyter kernelspec list
# debe listar:  proyecto1mlns   C:\Users\<usuario>\AppData\Roaming\jupyter\kernels\proyecto1mlns
```

### 3. Credenciales de Kaggle

Configurar `kaggle.json` como se indica en la sección [Dataset](#dataset) antes de ejecutar el notebook.

## Cómo ejecutar el notebook

1. Abrir `notebooks/microproyecto1_paleta_colores.ipynb` en VS Code o JupyterLab.
2. Seleccionar el kernel **`Python (Proyecto1MLNS)`**:
   - VS Code: selector de kernel (arriba a la derecha) -> *Select Another Kernel...* -> *Jupyter Kernel...* -> `Python (Proyecto1MLNS)`. Si no aparece, `Ctrl+Shift+P` -> *Developer: Reload Window* y reintentar. Alternativa equivalente: *Python Environments...* -> `.\venv\Scripts\python.exe`.
   - JupyterLab: menú *Kernel -> Change Kernel...*.
3. Confirmar el entorno ejecutando en una celda: `import sys; print(sys.executable)` -> debe terminar en `...\Proyecto1MLNS\venv\Scripts\python.exe`.
4. `Run All`. La primera ejecución tarda más por las descargas vía `kagglehub`.
5. Antes de entregar, exportar a HTML con todas las celdas ya ejecutadas:
   ```powershell
   .\venv\Scripts\python.exe -m jupyter nbconvert --to html --output-dir reports notebooks/microproyecto1_paleta_colores.ipynb
   ```

## Entregable y rúbrica

El entregable son **dos archivos**: el notebook (`.ipynb`) y su export (`.html`), documentados con las justificaciones de cada decisión tomada, con las ejecuciones de todas las celdas visibles. Al final debe mostrarse la paleta generada para al menos **4 imágenes de estilos diferentes**, junto con la visualización de la distribución de colores en 2D (t-SNE sugerido). Entrega: **fin de la semana 5** del curso.

| Actividad | Peso |
|---|---|
| Recolección de imágenes diversas (≥3 estilos/pintores, 6–10 imágenes), con justificación | 10% |
| Preparación de las imágenes, con justificación de las decisiones | 10% |
| Construcción de un pipeline de preparación de imágenes | 10% |
| Modelo de clustering con búsqueda de hiperparámetros y métricas justificadas (**el número de clusters debe variar por imagen**, no ser fijo) | 30% |
| Función que transforma los clusters de color en un muestrario representativo | 15% |
| Evidencia de desempeño: paleta + visualización 2D para ≥4 imágenes de estilos distintos | 25% |

Ver [`PLAN.md`](./PLAN.md) para el desglose del trabajo pendiente mapeado a esta rúbrica.

## Estado actual del proyecto

El notebook completó la actividad de recolección de imágenes (10%); el resto sigue pendiente:

- [hecho] Entorno reproducible: venv en `venv/` (Python 3.14.6), `requirements.txt` con todas las dependencias del notebook, y kernel de Jupyter `Python (Proyecto1MLNS)` registrado apuntando al venv. Ver [Instalación y configuración del entorno](#instalación-y-configuración-del-entorno).
- [hecho] Descarga de datos vía `kagglehub` y carga de `classes.csv`.
- [hecho] Selección final: 10 imágenes en total (2 por estilo × 5 estilos: Action painting, Baroque, Cubism, Impressionism, Ukiyo e), dentro del rango 6–10 pedido por la rúbrica, con justificación en markdown del criterio de selección.
- [hecho] Descarga de las imágenes seleccionadas.
- [hecho] Visualización de la selección: grilla con las 10 imágenes agrupadas por estilo.
- [hecho] Pipeline de preparación de imágenes: conversión a RGB, resize a 150 px, conversión a espacio de color Lab, aplanado de píxeles, con justificación de cada decisión (incluida la de no normalizar) y evidencia cuantitativa/visual del resize.
- [pendiente] Notebook editado pero no re-ejecutado en esta sesión (sin `kagglehub`/credenciales de Kaggle en esta máquina) — falta un `Run All` para que los outputs queden actualizados antes de commitear/entregar.
- [pendiente] Modelo de clustering por imagen con búsqueda de hiperparámetros.
- [pendiente] Función de generación de paleta/muestrario.
- [pendiente] Visualización 2D (t-SNE) de la distribución de colores.
- [pendiente] Muestra final con ≥4 imágenes de estilos distintos.
- [pendiente] Documentación en markdown de las decisiones a lo largo de todo el notebook.
- [pendiente] Export a `.html`.

## Equipo

- bcollante98@gmail.com
- saragallegovillada@gmail.com
