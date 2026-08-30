# Plan de desarrollo — Microproyecto 1

Desglose del trabajo pendiente para el notebook `notebooks/microproyecto1_paleta_colores.ipynb`, mapeado a la rúbrica del curso. Entrega: **fin de semana 5**. Ver el estado actual detallado en [`CLAUDE.md`](./CLAUDE.md) y la rúbrica completa en [`README.md`](./README.md).

Se marca cada paso como `[pendiente]`, `[en progreso]` o `[hecho]`, y se actualiza a medida que se avanza.

## (a) Selección de imágenes — 10% [hecho]

[hecho] Selección recortada a 10 imágenes en total (2 por estilo × 5 estilos: Action painting, Baroque, Cubism, Impressionism, Ukiyo e), dentro del rango 6–10 pedido por la rúbrica. Se excluyó Romanticism para maximizar diversidad cromática entre estilos.
[hecho] Evidencia visual de la selección: grilla que muestra las 10 imágenes agrupadas por estilo (antes solo se mostraba "Cubism").
[hecho] Celdas markdown agregadas explicando el criterio de selección (por qué esos 5 estilos, por qué se excluyó Romanticism, por qué 2 imágenes/estilo, por qué ese `random_state`).
[hecho] Notebook re-ejecutado de punta a punta con el kernel `proyecto1mlns` (venv del repo) vía `jupyter nbconvert --execute`; las imágenes ya estaban en la caché de `kagglehub`, así que corrió sin red ni credenciales. Todas las celdas de código tienen output embebido.

## (b) Preparación de imágenes — 10% [hecho]

[hecho] Pipeline implementado y ejecutado en el notebook (antes solo estaba descrito en la doc, no en el `.ipynb`). Estrategia justificada en markdown: BGR->RGB, resize del lado mayor a 150 px con `cv2.INTER_AREA` (con evidencia cuantitativa y visual: la media por canal se conserva, la desv. est. baja algo por el promediado), conversión a CIELAB (justificada por ser aproximadamente perceptualmente uniforme, relevante para clustering por distancia euclidiana), aplanado de píxeles, y decisión justificada de no estandarizar/normalizar.

## (c) Pipeline de preparación — 10% [hecho]

[hecho] Los pasos de (b) quedaron encapsulados en la función `preparar_imagen(path, max_dim=150)` -> `(img_rgb, pixeles_lab)`, aplicable a cualquier imagen del dataset (no solo a las 10 seleccionadas). Se aplicó a las 10 imágenes seleccionadas; los resultados quedan en el diccionario `imagenes_preparadas` (filename -> style/artist/img_rgb/pixeles_lab) para el paso de clustering.

## (d) Modelo de clustering — 30% (mayor peso de la rúbrica) [hecho]

[hecho] Algoritmo: **K-Means** (`k-means++`, `n_init=10`, `random_state=42`) como método principal, sobre los píxeles Lab, con barrido de `k` en 2..12 por imagen. Funciones `barrido_k_kmeans`, `elegir_k_codo`, `contraste_meanshift`; resultados en el dict `resultados_clustering` (curva_k, best_k, k_silhouette, centros_lab, labels, proporciones, ms_n_clusters, ms_bandwidth).
[hecho] Búsqueda de `k` **por imagen**: criterio primario = **método del codo** (Kneedle por distancia a la cuerda sobre la curva de inercia). Da `k` = 4 o 5 según la obra. Se descartó silhouette como criterio primario porque colapsa a `k=2` en 9/10 imágenes (sesgo conocido hacia pocos clusters); queda como control (valores 0.38–0.58 en el `k` elegido, confirman estructura).
[hecho] Contraste independiente con **MeanShift** (`bandwidth` por imagen vía `estimate_bandwidth`, `quantile=0.2`, submuestra de 3000 px): 2–6 clusters en 9/10 imágenes, en el mismo orden de magnitud que el codo (una degeneración a 1 cluster en el Pollock por `bandwidth` grande, documentada).
[hecho] Gráfica 5x2: inercia (codo, `k` elegido) + silhouette (control) por imagen.

## (e) Visualización 2D de la distribución de colores [hecho]

[hecho] Reducción de dimensionalidad con **t-SNE** (`sklearn.manifold.TSNE`, `perplexity=30`, `init="pca"`, `random_state=42`) sobre una submuestra de 3.000 píxeles Lab por imagen (t-SNE es O(n^2)). Funciones `lab_a_rgb` (Lab 8-bit -> RGB [0,1] para colorear los puntos con su color real) y `proyeccion_2d_colores` (corre t-SNE sobre la pila `[muestra de píxeles ; centros de K-Means]` para situar los centros en el mismo plano, ya que t-SNE no tiene `transform`). Grilla 5x2: por imagen, scatter de la nube de color (cada punto con su RGB real) + centros de K-Means como `X` con tamaño proporcional a su peso y rótulo con el índice de clúster. Resultados en el dict `proyecciones_2d` (xy_pix, xy_cen, idx, labels) para reutilizar en la actividad (g). Celdas markdown justifican la elección de t-SNE sobre PCA, el submuestreo, `perplexity`, y el criterio de coloreado; celda "Lectura de la visualización 2D" al cierre.

## (f) Función de generación de paleta — 15% [hecho]

[hecho] `generar_paleta(centros_lab, proporciones)` devuelve un `DataFrame` (`rank, hex, R, G, B, proporcion`) ordenado por proporción descendente: los centros de K-Means en Lab se pasan a sRGB (vía `lab_a_rgb`) y a código `#rrggbb`. `mostrar_paleta(paleta, ax, titulo)` dibuja el muestrario como franjas horizontales de ancho proporcional al peso de cada color, rotuladas con hex + porcentaje (texto negro/blanco según luminancia). Grilla 5x2 con la paleta de las 10 imágenes + impresión del `DataFrame` de una de ellas. Resultados en el dict `paletas` (filename -> DataFrame). Celda markdown justifica el formato de salida, el orden y el ancho proporcional.

## (g) Muestra final — 25% [hecho]

[hecho] Sección "## Muestra final (actividad g)": se toma un representante por estilo (5 imágenes, > el mínimo de 4) y se dibuja una grilla `5 x 3` con, por fila, imagen preparada + paleta generada (actividad f) + distribución de color 2D t-SNE (actividad e). Celda markdown "### Lectura de la muestra final" conecta las tres vistas (el grabado monocromo de Rembrandt da una paleta casi neutra; los rojos de acento del Pissarro y el Bush; `k` = 4 o 5 por imagen).

## (h) Documentación — requisito transversal del entregable [hecho]

[hecho] Celdas markdown de justificación a lo largo de todo el notebook: (a) selección de imágenes, (b)/(c) espacio de color Lab, resize, no normalizar y encapsulado en función, (d) K-Means, codo vs silhouette, MeanShift de contraste y lectura de resultados, (e) t-SNE vs PCA, submuestreo, perplexity, coloreado por color real y lectura de la viz 2D, (f) formato de salida de la paleta, orden y ancho proporcional, (g) lectura de la muestra final. 20 celdas markdown / 19 de código.

## (i) Exportación y verificación final [hecho]

[hecho] Notebook ejecutado de principio a fin con `jupyter nbconvert --to notebook --execute --inplace` (kernel = venv del repo). Las 19 celdas de código tienen output embebido, `execution_count` 1..19, sin errores ni tracebacks. Excepción: dos celdas markdown de cierre (lectura de la viz 2D y lectura de la muestra final) se ajustaron después de ejecutar (no requieren ejecución).
[hecho] Exportado a `reports/microproyecto1_paleta_colores.html` con `jupyter nbconvert --to html --output-dir reports ...` (~7.8 MB, 7 figuras embebidas como data-URI).
[hecho] `.html` revisado: contiene todas las secciones (a)-(g), sin `Traceback` ni salidas de error. Único warning de nbconvert: "Alternative text is missing on 6 image(s)" (texto alt de las figuras; no afecta el contenido).

## Notas de coordinación

- Repo compartido con `bcollante` — avisar antes de hacer cambios grandes en el notebook (alto riesgo de conflicto de merge por ser JSON con outputs embebidos).
- Actualizar este archivo (`[pendiente]` -> `[en progreso]` -> `[hecho]`) a medida que se completa cada paso, para que cualquiera (persona o sesión de Claude Code) sepa el estado real sin tener que releer todo el notebook.
