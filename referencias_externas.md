# Referencias de datos externos y uso de IA generativa

**Grupo 11 — Proyecto 2 (Entrega 2)**
**Aprendizaje de Máquina 2026-10 · Universidad de los Andes**

Este es uno de los tres entregables en Bloque Neón, junto con el notebook final y el modelo guardado. Cubre dos puntos del enunciado: las referencias a los datos externos (fuente, enlace y licencia) y el uso que le dimos a la IA generativa durante el proyecto.

---

## 1. Datos externos

Preparamos un dataset de Project Gutenberg, pero al final no entró al modelo entregado. La idea era integrarlo en NB12 (con el flag `USE_EXTERNAL_DATA=True`), pero no alcanzamos a entrenar ese notebook a tiempo, así que el modelo final reutiliza el bundle de NB11, sin datos externos. Lo documentamos igual porque la curaduría, el filtrado y la deduplicación quedaron implementados y son reproducibles (`scripts/fetch_external.py`).

| Fuente | URL | Licencia | Volumen | Décadas | Estado |
|---|---|---|---|---|---|
| Project Gutenberg (API gutendex) | gutenberg.org · gutendex.com | Dominio público | 525 párrafos (~524 tras dedup; ~1.9% del train) | 13 de 39 | Preparado, no usado en el modelo final |
| Wikisource ES | es.wikisource.org | CC BY-SA 4.0 | — | — | Propuesto, no implementado |
| Internet Archive (libros en español) | archive.org | Dominio público | — | — | Propuesto, no implementado |

De las tres fuentes solo Project Gutenberg llegó a descargarse: 525 párrafos de obras de dominio público con la década de publicación curada a mano. Sabíamos desde el diseño que tenía limitaciones. Algunas ediciones traen prólogos modernos del editor mezclados con el texto de época; los textos están modernizados ortográficamente y sin ruido OCR, lo que los aleja del corpus del concurso (OCR histórico sucio); y la cobertura es parcial, 13 de 39 décadas, porque varias búsquedas no devolvieron edición en español.

Para NB12 habíamos previsto cómo mitigar esto: augmentación de ruido realista durante el entrenamiento (`utils.augment_realistic`, sobre todo dobles espacios y cortes con guion, que es el ruido dominante del corpus) para acercar el dominio; topes por década y global para no sesgar; deduplicación por n-gramas contra eval y holdout (`utils.overlap_with_reference`), que respeta la prohibición de §5.1 de tocar los datos de prueba; y un ablation que habría descartado los externos si empeoraban el F1 en holdout. Como NB12 no se entrenó, nada de esto se aplicó al modelo final.

**Filtros aplicados al descargar:**
- Longitud de 50 a 500 palabras por párrafo, en línea con `train.csv`.
- Solo párrafos con fecha de publicación entre 1500 y 1899.
- Tope del 20% del tamaño del train interno.
- Idioma español, verificado con `langdetect` y el campo `language` de la fuente.

**Procedimiento (`scripts/fetch_external.py`):** partimos de una lista curada a mano de obras con década conocida, priorizando los siglos XVI y XVII por ser los más escasos del corpus. Para cada obra buscamos en gutendex, descargamos el `text/plain`, quitamos el boilerplate de Gutenberg, segmentamos en párrafos de 50–500 palabras y los etiquetamos con la década. La salida queda en `data_external/textos_externos.csv`.

---

## 1.b Modelos preentrenados y embeddings

El enunciado (§5.1) pide usar transferencia de aprendizaje y modelos preentrenados. Todos los recursos son públicos y descargables desde HuggingFace (`huggingface.co/<nombre>`), con licencias que permiten uso académico.

| Recurso | Licencia | Uso en el proyecto |
|---|---|---|
| FacebookAI/xlm-roberta-base | MIT | Encoder de NB09 y miembro del ensemble (NB07, NB10, NB11). Multilingüe, cubre el latín, italiano y catalán que aparecen en el corpus. |
| FacebookAI/xlm-roberta-large | MIT | Encoder principal del modelo final (bundle de NB11). Entrenado en T4 con gradient checkpointing, fp16 y batch efectivo de 32. |
| dccuchile/bert-base-spanish-wwm-cased (BETO) | CC BY 4.0 | Modelo B en NB07 y miembro de NB10; aporta una vista monolingüe del castellano y diversidad frente a RoBERTa. |
| microsoft/mdeberta-v3-base | MIT | Intento en NB10. Diverge a NaN ya en el primer paso, incluso en fp32, así que el código lo excluye del ensemble si falla. |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | Apache 2.0 | Embeddings congelados para el baseline transformer (NB03). |
| FastText cc.es.300 | CC BY-SA 3.0 | Embeddings preentrenados (subword) para NB00–02. |

**Modelos que consideramos pero descartamos:**
- `PlanTL-GOB-ES/roberta-base-bne`: repositorio deprecated (redirige a BSC-LT).
- `BSC-LT/roberta-base-bne`: devuelve HTTP 401 (gated o caído).
- `bertin-project/bertin-roberta-base-spanish`: bug de carga de LayerNorm en transformers ≥4.45 (`gamma`/`beta` no mapean a `weight`/`bias`), visto en los logs de NB04.
- `Qwen/Qwen3-Embedding-8B`: el más capaz, pero no cabe en una T4 de 16 GB.

Sobre mDeBERTa-v3: intentamos usarlo como encoder primario, pero diverge a NaN en el primer paso de optimización incluso en fp32 con LR 1e-5, la inestabilidad conocida de su disentangled attention. `dl_utils.train_model` tiene una guardia NaN (reintenta en fp32 si fp16 falla) y `adam_eps` configurable, pero contra la divergencia en fp32 la única salida fiable fue volver a XLM-R, que es estable y ya había rendido bien en NB06.

---

## 2. Uso de IA generativa

Usamos Claude (Anthropic) como herramienta de apoyo. El diseño del proyecto fue nuestro: la estructura y la escritura de los notebooks, la elección de modelos e hiperparámetros, las hipótesis y las decisiones de arquitectura las tomamos como grupo. La IA aportó sobre todo en cuatro frentes:

- **Funciones auxiliares.** Borradores de utilidades en `utils.py` y `dl_utils.py` (limpieza de texto, cálculo de métricas, la guardia NaN del loop de entrenamiento, el cargador de datos externos), que revisamos y ajustamos antes de integrarlas.
- **Análisis de resultados intermedios.** Apoyo para leer las curvas de entrenamiento y los diagnósticos: por ejemplo, ver que el large quedó subentrenado en NB11 (el LR decayó a casi cero con el loss todavía bajando) o que el smoothing gaussiano bajaba el F1 exacto de 0.24 a 0.20.
- **Investigación.** Verificar en HuggingFace qué modelos siguen activos y cuáles están deprecated, y contrastar alternativas antes de descartarlas (bertin por el bug de LayerNorm, Qwen3-Embedding-8B por tamaño).
- **Documentación.** Redacción de docstrings, celdas markdown de hipótesis y de este documento.

**Lo que no hizo la IA:**
- Etiquetar datos: las etiquetas vienen del `train.csv` que entregó el curso.
- Generar texto sintético para entrenar. La augmentación es ruido determinista de superficie (`utils.augment_realistic`), no texto producido por un LLM; los datos externos son obras reales de dominio público.
- Ejecutar los modelos: los entrenamientos y submissions corrieron en nuestras cuentas de Colab y Kaggle.
- La sustentación oral.

Todo el código que salió de la IA pasó por nuestra revisión y prueba antes de ejecutarse, y las decisiones de fondo se discutieron en grupo.

---

## 3. Estado final

Actualizado el 2026-05-23. El modelo final es el híbrido clásico + DL sobre el bundle de NB11 (`notebooks/diego/nb_final_ejecutado.ipynb`); los datos externos quedaron preparados pero no se incorporaron, porque NB12 no se entrenó a tiempo. Resultado: F1 de 0.3383 en holdout y 0.33810 en el Kaggle público (7.º puesto), con el blend calibrado por log y temperatura.

Integrantes: Diego Molano · Andrés Cáceres · Diego Ojeda · Pablo Ramírez
