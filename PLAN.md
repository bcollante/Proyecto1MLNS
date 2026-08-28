# Plan de desarrollo — Microproyecto 1

Desglose del trabajo pendiente para el notebook `notebooks/microproyecto1_paleta_colores.ipynb`, mapeado a la rúbrica del curso. Entrega: **fin de semana 5**. Ver el estado actual detallado en [`CLAUDE.md`](./CLAUDE.md) y la rúbrica completa en [`README.md`](./README.md).

Se marca cada paso como pendiente (⬜), en progreso (🟡) o hecho (✅), y se actualiza a medida que se avanza.

## (a) Selección de imágenes — 10%

⬜ Completar la evidencia visual de la selección: hoy solo se muestra el estilo "Cubism" (10 de 60 imágenes). Falta mostrar los otros 5 estilos (Action painting, Baroque, Impressionism, Romanticism, Ukiyo e).
⬜ Agregar celdas markdown al inicio explicando el criterio de selección: por qué esos 6 estilos, por qué 10 imágenes por estilo, y por qué ese `random_state`.

## (b) Preparación de imágenes — 10%

⬜ Definir y justificar la estrategia de preprocesamiento por imagen: tamaño de resize (y por qué), espacio de color a usar (RGB vs HSV vs Lab — Lab suele ser más perceptualmente uniforme para clustering de color), y cómo se muestrean/aplanan los píxeles para alimentar el modelo (todos los píxeles vs. una muestra aleatoria, por costo computacional).
⬜ Justificar cualquier normalización aplicada.

## (c) Pipeline de preparación — 10%

⬜ Encapsular los pasos de (b) en una función o `Pipeline` reutilizable que reciba una imagen (o su ruta) y devuelva la matriz de píxeles lista para clustering, aplicable a cualquier imagen del dataset.

## (d) Modelo de clustering — 30% (mayor peso de la rúbrica)

⬜ Elegir y justificar el algoritmo de clustering (K-Means, MeanShift, u otro — ya están importados ambos candidatos en el notebook).
⬜ Implementar **búsqueda de hiperparámetros por imagen**: el número de clusters (k) no puede ser fijo, debe determinarse individualmente para cada imagen (p. ej. método del codo o silhouette score sobre un rango de k, o MeanShift con bandwidth estimado vía `estimate_bandwidth`).
⬜ Justificar la métrica de evaluación usada para elegir k (o el equivalente en MeanShift).

## (e) Visualización 2D de la distribución de colores

⬜ Aplicar una técnica de reducción de dimensionalidad (t-SNE sugerido, `sklearn.manifold.TSNE`) sobre los colores de la imagen y graficar en 2D, coloreando por cluster asignado.

## (f) Función de generación de paleta — 15%

⬜ Función que, a partir de los centros de los clusters (colores dominantes), genere un muestrario/swatch visual (p. ej. barras de color ordenadas, con su proporción en la imagen).

## (g) Muestra final — 25%

⬜ Sección final del notebook que, para **al menos 4 imágenes de estilos distintos**, corra el pipeline completo y muestre: imagen original, paleta generada, y visualización 2D de la distribución de colores.

## (h) Documentación — requisito transversal del entregable

⬜ Agregar celdas markdown a lo largo de todo el notebook (actualmente tiene 0) justificando cada decisión: selección de imágenes, preprocesamiento, algoritmo y métricas de clustering, diseño de la función de paleta.

## (i) Exportación y verificación final

⬜ Ejecutar el notebook completo de principio a fin (`Run All`) y confirmar que todas las celdas muestran su output.
⬜ Exportar con `jupyter nbconvert --to html --output-dir reports notebooks/microproyecto1_paleta_colores.ipynb`.
⬜ Revisar el `.html` generado antes de entregar ambos archivos (`.ipynb` + `.html`).

## Notas de coordinación

- Repo compartido con `bcollante` — avisar antes de hacer cambios grandes en el notebook (alto riesgo de conflicto de merge por ser JSON con outputs embebidos).
- Actualizar este archivo (marcar ⬜ → 🟡 → ✅) a medida que se completa cada paso, para que cualquiera (persona o sesión de Claude Code) sepa el estado real sin tener que releer todo el notebook.
