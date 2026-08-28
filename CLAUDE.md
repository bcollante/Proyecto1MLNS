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
- Cada decisión no obvia (elección de espacio de color, método de clustering, métrica, cantidad de imágenes) necesita una celda markdown justificándola. Se viene haciendo así desde la actividad (a); mantener el patrón en las actividades siguientes.
- No usar emojis en el notebook ni en ningún archivo del proyecto (instrucción explícita del usuario) — usar marcadores de texto como `[hecho]`/`[pendiente]`/`[en progreso]` en su lugar.

## Colaboración

Repo compartido con el compañero **bcollante** (remoto `origin` → `github.com/bcollante/Proyecto1MLNS`). Antes de empezar a trabajar:

- Hacer `git pull` primero.
- Nunca `push --force` a `main`.
- El notebook es un archivo JSON con outputs de imágenes embebidos (pesa >1MB): alto riesgo de conflictos de merge si dos personas lo editan a la vez sin coordinarse. Avisar/coordinar con el compañero antes de hacer cambios grandes o reorganizaciones.

## Estado de implementación actual

El notebook tiene 20 celdas (11 código + 9 markdown). Las actividades (a), (b) y (c) de la rúbrica (recolección de imágenes 10%, preparación de imágenes 10%, pipeline de preparación 10%) están completas. Resumen para no tener que releerlo entero cada sesión:

1. *(markdown)* Introducción del proyecto y estructura del notebook.
2. Descarga de prueba de una imagen vía `kagglehub` (solo confirma que el mecanismo funciona; no es parte de la selección final).
3. Carga de `data/classes.csv` en un DataFrame (ruta `../data/classes.csv`, relativa a `notebooks/`).
4. *(markdown)* Justificación del criterio de selección de imágenes.
5. Define 5 estilos (`Impressionism`, `Cubism`, `Baroque`, `Ukiyo e`, `Action painting` — se excluyó `Romanticism` por solapar paleta cromática con Baroque) y parsea `genre` a una columna `style`.
6. Filtra por esos 5 estilos y muestrea `N_PER_STYLE=2` imágenes por estilo (`random_state=42`) → **10 imágenes seleccionadas en total**, dentro del rango 6–10 que pide la rúbrica.
7. *(markdown)* Nota sobre la descarga.
8. Descarga las 10 imágenes seleccionadas vía `kagglehub`.
9. Imports para la etapa de modelado: `cv2`, `sklearn.cluster.KMeans`, `sklearn.cluster.MeanShift`, `estimate_bandwidth`, `PIL.Image`.
10. *(markdown)* Aclara que esos imports son para las próximas etapas.
11. Imprime los 5 estilos únicos seleccionados (sanity check).
12. *(markdown)* Introduce la sección de imágenes seleccionadas.
13. Grilla `n_styles × N_PER_STYLE` (5×2) que muestra las 10 imágenes seleccionadas, cada una titulada con estilo + artista.
14. *(markdown)* Cabecera "## Preparación de las imágenes (actividades b y c)" + justificación completa del pipeline: BGR->RGB, resize a 150 px con `cv2.INTER_AREA`, conversión a Lab (8 bits de OpenCV, canales 0–255), aplanado de píxeles, decisión de no estandarizar/normalizar.
15. Define `preparar_imagen(path, max_dim=150)`: el pipeline reutilizable (carga -> BGR->RGB -> resize -> RGB->Lab -> píxeles aplanados float32). Devuelve `(img_rgb, pixeles_lab)`. Este es el punto de extensión si el algoritmo de resize/espacio de color cambia más adelante.
16. *(markdown)* Introduce la aplicación del pipeline a las 10 imágenes.
17. Aplica `preparar_imagen` a las 10 imágenes seleccionadas, guarda resultados en `imagenes_preparadas` (dict: filename -> style/artist/img_rgb/pixeles_lab), imprime tabla con resolución original -> preparada -> nº de píxeles por imagen.
18. *(markdown)* Introduce la verificación del resize (visual + cuantitativa: media por canal se conserva, desv. est. baja algo por el promediado de INTER_AREA).
19. Compara original vs. redimensionada para una imagen de ejemplo (grilla + estadísticos RGB media/desv. est.).
20. *(markdown)* Cierre apuntando al siguiente paso pendiente (modelo de clustering por imagen, actividad d).

**Importante:** el notebook fue re-ejecutado de punta a punta en esta sesión con el kernel `proyecto1mlns` (venv del repo) vía `jupyter nbconvert --execute`. Las 11 imágenes que usa (10 seleccionadas + 1 de prueba) ya estaban en la caché de `kagglehub` (`~/.cache/kagglehub/datasets/steubk/wikiart/versions/1`), así que corrió sin red ni credenciales. Todas las celdas de código tienen su output actual embebido. Excepción: la celda markdown 18 se editó después de ejecutar (no requiere ejecución). El entorno está documentado en `README.md`.

Lo que falta (ver `PLAN.md` para el desglose completo por peso de rúbrica): modelo de clustering por imagen con búsqueda de hiperparámetros y número de clusters variable (30%, el ítem de mayor peso), función de generación de paleta (15%), visualización t-SNE 2D, sección final con ≥4 imágenes de estilos distintos (25%), y export a `.html`.
