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

## (d) Modelo de clustering — 30% (mayor peso de la rúbrica)

[pendiente] Elegir y justificar el algoritmo de clustering (K-Means, MeanShift, u otro — ya están importados ambos candidatos en el notebook).
[pendiente] Implementar **búsqueda de hiperparámetros por imagen**: el número de clusters (k) no puede ser fijo, debe determinarse individualmente para cada imagen (p. ej. método del codo o silhouette score sobre un rango de k, o MeanShift con bandwidth estimado vía `estimate_bandwidth`).
[pendiente] Justificar la métrica de evaluación usada para elegir k (o el equivalente en MeanShift).

## (e) Visualización 2D de la distribución de colores

[pendiente] Aplicar una técnica de reducción de dimensionalidad (t-SNE sugerido, `sklearn.manifold.TSNE`) sobre los colores de la imagen y graficar en 2D, coloreando por cluster asignado.

## (f) Función de generación de paleta — 15%

[pendiente] Función que, a partir de los centros de los clusters (colores dominantes), genere un muestrario/swatch visual (p. ej. barras de color ordenadas, con su proporción en la imagen).

## (g) Muestra final — 25%

[pendiente] Sección final del notebook que, para **al menos 4 imágenes de estilos distintos**, corra el pipeline completo y muestre: imagen original, paleta generada, y visualización 2D de la distribución de colores.

## (h) Documentación — requisito transversal del entregable

[en progreso] Agregar celdas markdown a lo largo de todo el notebook justificando cada decisión. Ya cubierto para las actividades (a) (selección de imágenes) y (b)/(c) (espacio de color Lab, resize, no normalizar, encapsulado en función); falta para algoritmo y métricas de clustering, y diseño de la función de paleta, a medida que se implementan.

## (i) Exportación y verificación final

[pendiente] Ejecutar el notebook completo de principio a fin (`Run All`) y confirmar que todas las celdas muestran su output.
[pendiente] Exportar con `jupyter nbconvert --to html --output-dir reports notebooks/microproyecto1_paleta_colores.ipynb`.
[pendiente] Revisar el `.html` generado antes de entregar ambos archivos (`.ipynb` + `.html`).

## Notas de coordinación

- Repo compartido con `bcollante` — avisar antes de hacer cambios grandes en el notebook (alto riesgo de conflicto de merge por ser JSON con outputs embebidos).
- Actualizar este archivo (`[pendiente]` -> `[en progreso]` -> `[hecho]`) a medida que se completa cada paso, para que cualquiera (persona o sesión de Claude Code) sepa el estado real sin tener que releer todo el notebook.
