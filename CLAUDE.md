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

El notebook tiene 39 celdas (19 código + 20 markdown). Las actividades (a) a (g) de la rúbrica (recolección 10%, preparación 10%, pipeline 10%, modelo de clustering 30%, visualización 2D, función de paleta 15%, muestra final 25%) están completas, y el notebook está exportado a `reports/*.html`. Resumen para no tener que releerlo entero cada sesión:

1. *(markdown)* Introducción del proyecto y estructura del notebook.
2. Descarga de prueba de una imagen vía `kagglehub` (solo confirma que el mecanismo funciona; no es parte de la selección final).
3. Carga de `data/classes.csv` en un DataFrame (ruta `../data/classes.csv`, relativa a `notebooks/`).
4. *(markdown)* Justificación del criterio de selección de imágenes.
5. Define 5 estilos (`Impressionism`, `Cubism`, `Baroque`, `Ukiyo e`, `Action painting` — se excluyó `Romanticism` por solapar paleta cromática con Baroque) y parsea `genre` a una columna `style`.
6. Filtra por esos 5 estilos y muestrea `N_PER_STYLE=2` imágenes por estilo (`random_state=42`) → **10 imágenes seleccionadas en total**, dentro del rango 6–10 que pide la rúbrica.
7. *(markdown)* Nota sobre la descarga.
8. Descarga las 10 imágenes seleccionadas vía `kagglehub`.
9. Imports para las etapas siguientes: `numpy`, `cv2`, `matplotlib` + `matplotlib.patches.Rectangle`, `sklearn.cluster.KMeans/MeanShift/estimate_bandwidth`, `sklearn.metrics.silhouette_score`, `sklearn.manifold.TSNE`, `PIL.Image`.
10. *(markdown)* Aclara para qué actividad se usa cada import (OpenCV/PIL en b/c, KMeans/MeanShift/silhouette en d, TSNE en e, Rectangle en f).
11. Imprime los 5 estilos únicos seleccionados (sanity check).
12. *(markdown)* Introduce la sección de imágenes seleccionadas.
13. Grilla `n_styles × N_PER_STYLE` (5×2) que muestra las 10 imágenes seleccionadas, cada una titulada con estilo + artista.
14. *(markdown)* Cabecera "## Preparación de las imágenes (actividades b y c)" + justificación completa del pipeline: BGR->RGB, resize a 150 px con `cv2.INTER_AREA`, conversión a Lab (8 bits de OpenCV, canales 0–255), aplanado de píxeles, decisión de no estandarizar/normalizar.
15. Define `preparar_imagen(path, max_dim=150)`: el pipeline reutilizable (carga -> BGR->RGB -> resize -> RGB->Lab -> píxeles aplanados float32). Devuelve `(img_rgb, pixeles_lab)`. Este es el punto de extensión si el algoritmo de resize/espacio de color cambia más adelante.
16. *(markdown)* Introduce la aplicación del pipeline a las 10 imágenes.
17. Aplica `preparar_imagen` a las 10 imágenes seleccionadas, guarda resultados en `imagenes_preparadas` (dict: filename -> style/artist/img_rgb/pixeles_lab), imprime tabla con resolución original -> preparada -> nº de píxeles por imagen.
18. *(markdown)* Introduce la verificación del resize (visual + cuantitativa: media por canal se conserva, desv. est. baja algo por el promediado de INTER_AREA).
19. Compara original vs. redimensionada para una imagen de ejemplo (grilla + estadísticos RGB media/desv. est.).
20. *(markdown)* Cierre de la actividad (b)/(c), apunta a la actividad (d).
21. *(markdown)* Cabecera "## Modelo de agrupación por imagen (actividad d)" + justificación: K-Means principal, criterio de decisión = método del codo (Kneedle); silhouette se probó y se descartó como criterio (colapsa a k=2), queda como control; MeanShift como contraste.
22. Define `barrido_k_kmeans` (K-Means k=2..12, devuelve DataFrame con inercia y silhouette por k), `elegir_k_codo` (Kneedle por distancia a la cuerda de la curva de inercia) y `contraste_meanshift` (bandwidth vía `estimate_bandwidth`, submuestra 3000 px). Constantes `K_MIN/K_MAX=2/12`, `SAMPLE_SILHOUETTE=2000`, `SAMPLE_MEANSHIFT=3000`, `RANDOM_STATE=42`.
23. *(markdown)* Describe el contenido del dict `resultados_clustering`.
24. Bucle sobre `imagenes_preparadas`: arma `resultados_clustering[filename]` (curva_k, best_k=codo, k_silhouette, sil_en_best_k, centros_lab, labels, proporciones, ms_n_clusters, ms_bandwidth) e imprime tabla resumen + rango de k (da 4–5 por imagen).
25. *(markdown)* Introduce la grilla de gráficas.
26. Grilla 5x2: por imagen, inercia (codo, línea llena en el k elegido) + silhouette (control, punteada en su máximo).
27. *(markdown)* "### Lectura de resultados": k se recalcula por imagen (4 o 5), silhouette solo confirma estructura, MeanShift acompaña (con una degeneración a 1 cluster en el Pollock por bandwidth grande).
28. *(markdown)* Cabecera "## Visualización 2D de la distribución de colores (actividad e)" + justificación: t-SNE (no PCA) por preservar estructura local; submuestra de 3000 px (O(n^2)); `perplexity=30`, `init="pca"`; puntos coloreados con su RGB real (no por id de cluster) y centros de K-Means superpuestos; los centros se ubican corriendo t-SNE sobre la pila `[muestra ; centros]` porque t-SNE no tiene `transform`.
29. Define `SAMPLE_TSNE=3000`, `PERPLEXITY=30`, `lab_a_rgb(pixeles_lab)` (Lab 8-bit OpenCV -> RGB [0,1]) y `proyeccion_2d_colores(pixeles_lab, centros_lab, ...)` -> `(xy_pix, xy_cen, idx)`. Cota `perplexity < n/3` aplicada por seguridad.
30. *(markdown)* Introduce la grilla y el dict `proyecciones_2d`.
31. Bucle sobre `resultados_clustering`: arma `proyecciones_2d[filename]` (xy_pix, xy_cen, idx, labels) y dibuja la grilla 5x2 (scatter con color real por píxel + centros `X` con tamaño proporcional a la proporción y rótulo de índice de cluster).
32. *(markdown)* "### Lectura de la visualización 2D": la nube se separa en regiones coherentes, cada centro cae en una; tamaño de las `X` refleja la paleta; el Pollock (que MeanShift colapsó) aquí sí muestra estructura; recordatorio de que t-SNE es cualitativo.
33. *(markdown)* Cabecera "## Función de generación de la paleta (actividad f)" + justificación: entrada = salida de (d); conversión de centros Lab a sRGB y a hex; salida como `DataFrame` reutilizable (no solo gráfico); orden por proporción descendente; muestrario de franjas de ancho proporcional al peso.
34. Define `generar_paleta(centros_lab, proporciones)` -> `DataFrame` (`rank, hex, R, G, B, proporcion`) y `mostrar_paleta(paleta, ax=None, titulo=None)` (franjas `Rectangle` de ancho = proporción; rótulo hex + % en negro/blanco según luminancia; solo rotula franjas con ancho > 0.05).
35. *(markdown)* Introduce la grilla de paletas y el dict `paletas`.
36. Construye `paletas` (filename -> DataFrame), dibuja grilla 5x2 con la paleta de cada imagen e imprime el `DataFrame` de la primera como ejemplo del formato.
37. *(markdown)* Cabecera "## Muestra final (actividad g)" + explica que se toma un representante por estilo (5 imágenes, > el mínimo de 4).
38. Arma `seleccion_final` (primera imagen de cada estilo) y dibuja la grilla `5 x 3`: por fila, imagen preparada (`imshow`) + paleta (`mostrar_paleta`) + dispersión t-SNE (reusa `proyecciones_2d` y `paletas`).
39. *(markdown)* "### Lectura de la muestra final": el grabado monocromo de Rembrandt da paleta casi neutra; rojos de acento del Pissarro (16.4%) y el Bush (5.1%); `k` = 4 o 5 por imagen; las tres vistas son consistentes. Cierra apuntando al export `.html`.

**Importante:** el notebook fue ejecutado de punta a punta en esta sesión con el venv del repo (`venv/`, kernel `python3` que apunta a ese venv) vía `jupyter nbconvert --to notebook --execute --inplace`. Las 11 imágenes que usa (10 seleccionadas + 1 de prueba) ya estaban en la caché de `kagglehub` (`~/.cache/kagglehub/datasets/steubk/wikiart/versions/1`), así que corrió sin red ni credenciales. Las 19 celdas de código tienen su output actual embebido (execution_count 1..19), sin errores. Excepción: dos celdas markdown de cierre (lectura de la viz 2D y lectura de la muestra final) se ajustaron después de ejecutar (no requieren ejecución). El entorno está documentado en `README.md`.

**Export:** `reports/microproyecto1_paleta_colores.html` generado con `jupyter nbconvert --to html --output-dir reports notebooks/microproyecto1_paleta_colores.ipynb` (~7.8 MB, figuras embebidas). Regenerarlo tras cualquier re-ejecución del notebook.

Estado: las actividades (a) a (g) de la rúbrica están completas y el notebook está exportado. Pendiente solo: coordinar con `bcollante` y commitear (los cambios están en el working tree sin commitear).
