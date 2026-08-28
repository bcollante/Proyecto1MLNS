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

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Configurar las credenciales de Kaggle como se indica arriba antes de ejecutar el notebook.

## Cómo ejecutar el notebook

1. `jupyter notebook notebooks/microproyecto1_paleta_colores.ipynb` (o abrir en VS Code / JupyterLab).
2. Ejecutar todas las celdas en orden (`Run All`). La primera ejecución tardará más por las descargas vía `kagglehub`.
3. Antes de entregar, exportar a HTML con todas las celdas ya ejecutadas:
   ```bash
   jupyter nbconvert --to html --output-dir reports notebooks/microproyecto1_paleta_colores.ipynb
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

El notebook está en etapa temprana (solo la actividad de recolección de imágenes, parcialmente):

- ✅ Descarga de datos vía `kagglehub` y carga de `classes.csv`.
- ✅ Selección estratificada: 6 estilos (Action painting, Baroque, Cubism, Impressionism, Romanticism, Ukiyo e), 10 imágenes por estilo.
- ✅ Descarga de las imágenes seleccionadas.
- 🟡 Visualización de la selección: solo se muestra el estilo "Cubism" (falta el resto) y sin justificación en markdown de los criterios de selección.
- ❌ Pipeline de preparación de imágenes (resize, espacio de color, normalización).
- ❌ Modelo de clustering por imagen con búsqueda de hiperparámetros.
- ❌ Función de generación de paleta/muestrario.
- ❌ Visualización 2D (t-SNE) de la distribución de colores.
- ❌ Muestra final con ≥4 imágenes de estilos distintos.
- ❌ Documentación en markdown de las decisiones (el notebook no tiene celdas markdown todavía).
- ❌ Export a `.html`.

## Equipo

- bcollante98@gmail.com
- saragallegovillada@gmail.com
