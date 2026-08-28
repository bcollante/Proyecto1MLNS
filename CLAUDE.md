# CLAUDE.md

Guía para trabajar en este repo con Claude Code. Lee esto antes de tocar el notebook.

## Contexto del proyecto

Microproyecto 1 del curso **Machine Learning No Supervisado** (Maestría en IA): construir, mediante técnicas de clustering, un método que extraiga los tonos dominantes de imágenes de obras de arte (dataset WikiArt) y genere una paleta de colores representativa por imagen, junto con una visualización 2D de la distribución de colores. Ver [`README.md`](./README.md) para el detalle del objetivo, dataset y rúbrica completa, y [`PLAN.md`](./PLAN.md) para el desglose de trabajo pendiente.

## Ubicación de archivos clave

- `notebooks/microproyecto1_paleta_colores.ipynb` — notebook principal, único artefacto de código del proyecto.
- `data/classes.csv` — metadatos WikiArt (filename, artist, genre, phash, dimensiones, subset train/test). El notebook lo carga con `pd.read_csv("../data/classes.csv")` (ruta relativa desde `notebooks/`).
- `reports/` — destino del `.html` exportado antes de la entrega. Vacío salvo `.gitkeep` hasta que se genere.
- Las imágenes en sí **no** viven en el repo: se descargan bajo demanda con `kagglehub.dataset_download(...)`, cacheadas fuera del repo (`~/.cache/kagglehub` por defecto).

## Restricción crítica de la rúbrica (fácil de pasar por alto)

El número de clusters (k) **debe recalcularse por imagen**, no ser un valor fijo aplicado a todas. El modelo de clustering necesita búsqueda de hiperparámetros y una métrica de evaluación justificada (p. ej. silhouette score, método del codo) para elegir k en cada imagen. Esto pesa el 30% de la nota, el ítem individual más alto de la rúbrica — no implementar un k fijo "que funcione bien en general".

## Restricción de formato del entregable

El entregable son solo dos archivos: `notebooks/microproyecto1_paleta_colores.ipynb` y su export `.html`. Por eso:

- Toda la lógica (preprocesamiento, clustering, generación de paleta, visualización) debe vivir **dentro del notebook**, no en módulos `.py` separados — un corrector solo abre esos dos archivos.
- Cada celda debe quedar con su output ya ejecutado y visible antes de exportar.
- Cada decisión no obvia (elección de espacio de color, método de clustering, métrica, cantidad de imágenes) necesita una celda markdown justificándola — el notebook actualmente no tiene ninguna celda markdown, hay que ir agregándolas a medida que se avanza.

## Colaboración

Repo compartido con el compañero **bcollante** (remoto `origin` → `github.com/bcollante/Proyecto1MLNS`). Antes de empezar a trabajar:

- Hacer `git pull` primero.
- Nunca `push --force` a `main`.
- El notebook es un archivo JSON con outputs de imágenes embebidos (pesa >1MB): alto riesgo de conflictos de merge si dos personas lo editan a la vez sin coordinarse. Avisar/coordinar con el compañero antes de hacer cambios grandes o reorganizaciones.

## Estado de implementación actual

El notebook tiene 8 celdas de código (0 markdown). Resumen para no tener que releerlo entero cada sesión:

1. Descarga de prueba de una imagen vía `kagglehub` (confirma que el mecanismo funciona).
2. Carga de `data/classes.csv` en un DataFrame.
3. Define 6 estilos candidatos (`Impressionism`, `Cubism`, `Baroque`, `Ukiyo e`, `Romanticism`, `Action painting`) y parsea la columna `genre` (string de lista) a una columna `style`.
4. Filtra por esos 6 estilos y muestrea 10 imágenes por estilo (`random_state=42`) → 60 imágenes seleccionadas.
5. Descarga las 60 imágenes seleccionadas vía `kagglehub`.
6. Imports para la etapa de modelado: `cv2`, `sklearn.cluster.KMeans`, `sklearn.cluster.MeanShift`, `estimate_bandwidth`, `PIL.Image` — **sin código de clustering todavía**.
7. Imprime los 6 estilos únicos seleccionados (sanity check).
8. Muestra en una grilla 2x5 las 10 imágenes del estilo "Cubism" (único estilo mostrado visualmente hasta ahora).

Lo que falta (ver `PLAN.md` para el desglose completo por peso de rúbrica): mostrar visualmente los otros 5 estilos con justificación en markdown, pipeline de preparación de imágenes, modelo de clustering por imagen con búsqueda de hiperparámetros, función de generación de paleta, visualización t-SNE 2D, sección final con ≥4 imágenes de estilos distintos, documentación markdown a lo largo de todo el notebook, y export a `.html`.
