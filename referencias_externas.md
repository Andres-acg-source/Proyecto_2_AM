# Referencias de Datos Externos y Uso de IA Generativa

**Grupo 11 — Proyecto 2 (Entrega 2)**
**Curso:** Aprendizaje de Máquina 2026-10 — Universidad de los Andes

> Este documento es uno de los tres entregables formales en Bloque Neón
> (junto al notebook final y al modelo guardado). Cubre el requisito:
> *"Un documento adicional que incluya las referencias a los datos externos
> utilizados, especificando la fuente, el enlace y la licencia... Además
> debe incluir el uso dado a la IA generativa en el desarrollo del proyecto."*

---

## 1. Datasets externos

> **Estado en la entrega final:** el dataset externo de Project Gutenberg quedó **PREPARADO pero NO
> incorporado al modelo final**. Se diseñó su activación en NB12 (`USE_EXTERNAL_DATA=True`), pero ese
> entrenamiento **no se completó a tiempo para la entrega**; el modelo final reutiliza el bundle DL de
> NB11 (sin datos externos). Se documenta igualmente por trazabilidad: la curaduría, el filtrado y la
> deduplicación vs eval/holdout están implementados y son reproducibles (`scripts/fetch_external.py`).

| Fuente | URL | Licencia | Volumen (#textos) | Décadas cubiertas | Uso |
|---|---|---|---|---|---|
| Project Gutenberg (vía gutendex API) | https://www.gutenberg.org / https://gutendex.com | Public Domain | **525 párrafos** (≈524 tras dedup vs eval/holdout; ~1.9% del train interno) | **13 de 39**: 150, 155, 158, 160, 161, 162, 165, 169, 180, 183, 184, 187, 188 | **Preparada pero NO usada en el modelo final** (NB12 no se entrenó a tiempo): el pipeline de concatenación al train con tope por década + dedup vs eval/holdout está implementado y es reproducible (`scripts/fetch_external.py`). El modelo final usa el bundle DL de NB11 (sin externos) |
| Wikisource ES | https://es.wikisource.org/ | CC BY-SA 4.0 | (propuesta, no implementada) | — | Documentos históricos fechados |
| Internet Archive Spanish Books | https://archive.org/details/texts?and[]=language%3A%22Spanish%22 | Public Domain (filtrar por metadata) | (propuesta, no implementada) | — | Décadas XVI–XVII (más escasas) |

> **Estado:** solo Project Gutenberg está implementado y descargado (525 párrafos de obras de
> dominio público con década de publicación curada a mano), pero **no incorporado al modelo final**
> (NB12 no se entrenó a tiempo). **Limitaciones
> conocidas** (mitigadas en el diseño, ver más abajo): (1) algunas ediciones incluyen **prólogos/introducciones
> modernas** del editor (español contemporáneo, no de época) mezclados con el texto; (2) los textos
> están **modernizados ortográficamente y sin ruido OCR** → desajuste con el corpus del concurso;
> (3) cobertura parcial (13/39 décadas; varias búsquedas no devolvieron edición en español).
>
> **Mitigaciones diseñadas (para NB12, no ejecutado en la entrega):** (a) `utils.augment_realistic`
> durante el entrenamiento para acercar el dominio (dobles espacios, cortes con guion — el ruido
> dominante real del corpus); (b) tope por década (`per_decade_cap`) + tope global (`cap_frac=0.15`)
> para no sesgar; (c) **deduplicación por n-gramas de palabras contra eval Y holdout**
> (`utils.overlap_with_reference`) → cumple la prohibición de §5.1 de usar datos de prueba y evita
> inflar la métrica; (d) **ablation previsto**: NB12 reportaría el F1 del large en holdout y, si los
> externos lo empeoraban, se reentrenaría con `USE_EXTERNAL_DATA=False`. Como NB12 no se entrenó a
> tiempo, los externos **no entraron al modelo final** (volumen ~1.9% del train, contribución modesta).

**Filtros aplicados:**
- Longitud: 50–500 palabras por párrafo (alineado con la distribución del `train.csv` interno).
- Fecha: solo párrafos con metadato de publicación entre 1500 y 1899.
- Volumen total: ≤ 20% del tamaño del `train.csv` interno (≈ 6,280 ejemplos máx).
- Idioma: español (verificado con `langdetect` o el campo `language` de cada fuente).

**Procedimiento de obtención (implementado en `scripts/fetch_external.py`):**
1. Lista **curada a mano** de obras de dominio público con **década de publicación conocida**
   (no se infiere la fecha automáticamente — sería ruidoso). Se priorizan décadas tempranas
   (XVI–XVII), las más escasas del corpus.
2. Para cada obra: búsqueda en la API gutendex (`languages=es`), descarga del `text/plain`,
   eliminación del boilerplate de Gutenberg, segmentación en párrafos de 50–500 palabras,
   muestreo determinista (≤ `max_per_work`), etiquetado con la década curada.
3. Salida `data_external/textos_externos.csv` con columnas `text, decade, source, license`.

**CAVEAT importante (riesgo de distribución):** los textos de Project Gutenberg son ediciones
modernas con ortografía normalizada y **sin ruido OCR**, mientras que el train/eval del concurso
es OCR histórico sucio. En **todas las iteraciones entregadas (NB07–NB11) el toggle `USE_EXTERNAL_DATA`
quedó OFF**. La activación se diseñó para NB12 —`utils.augment_realistic` sobre todo el train
(incluidos los externos) para simular el ruido del corpus, más el **ablation** en holdout descrito
arriba—, pero **NB12 no se entrenó a tiempo**, así que los externos no llegaron al modelo final. Además
persiste un residuo de contaminación (prólogos modernos en algunas obras tempranas, p.ej. una intro
editorial de *La Celestina* etiquetada como década 150) que la dedup no elimina; por eso su inclusión
habría dependido del resultado del ablation, no de fe.

---

## 1.b Modelos preentrenados y embeddings externos

El enunciado §5.1 obliga al uso de "técnicas de transferencia de aprendizaje y modelos preentrenados". Todos los recursos aquí listados son **públicos, descargables y citables** vía HuggingFace o el repositorio oficial del proyecto. Las licencias permiten redistribución y uso académico.

| Recurso | URL (HuggingFace u origen) | Licencia | Uso en este proyecto |
|---|---|---|---|
| **FacebookAI/xlm-roberta-base** | https://huggingface.co/FacebookAI/xlm-roberta-base | MIT | Encoder de NB09 (single), modelo A de NB07 y miembro de NB10; **reusado** como miembro del ensemble en NB11/NB12. Multilingüe (94 idiomas) — cubre el latín/italiano/catalán del corpus; estable (entrenó bien en NB06). Validado activo en HF (mayo 2026) |
| **FacebookAI/xlm-roberta-large** | https://huggingface.co/FacebookAI/xlm-roberta-large | MIT | **Encoder primario del modelo final** (bundle de NB11). 550M, hidden 1024 — misma arquitectura que el base que ya rendía, con más capacidad y mejor preentrenamiento multilingüe. Entrenado en T4 con gradient checkpointing + fp16 + batch efectivo 32. NB12 se diseñó para corregir su subentrenamiento (más épocas + augmentación + datos externos) pero **no se entrenó a tiempo**; el modelo final usa el large de NB11 |
| **dccuchile/bert-base-spanish-wwm-cased** (BETO) | https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased | CC BY 4.0 | **Modelo B** de NB07 y miembro de NB10 (BERT español-específico, vista monolingüe sobre la mayoría castellana; diversidad RoBERTa vs BERT) |
| **microsoft/mdeberta-v3-base** | https://huggingface.co/microsoft/mdeberta-v3-base | MIT | Intento **best-effort** en NB10 (multilingüe, teóricamente superior). **Diverge a NaN al primer paso incluso en fp32** (LR 5e-6 + adam_eps 1e-6 no siempre basta) → se excluye automáticamente del ensemble si falla. Validado activo en HF (mayo 2026) |
| **sentence-transformers/paraphrase-multilingual-mpnet-base-v2** | https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | Apache 2.0 | Embeddings frozen para NB03 (baseline transformer) |
| **FastText cc.es.300** | https://fasttext.cc/docs/en/crawl-vectors.html | CC BY-SA 3.0 | Embeddings preentrenados (subword-aware) para NB00–02 |

**Modelos considerados pero NO usados** (documentados para trazabilidad de decisiones):
- `PlanTL-GOB-ES/roberta-base-bne` — repositorio **deprecated** (la página redirige a la organización BSC-LT); descartado.
- `BSC-LT/roberta-base-bne` — devuelve HTTP 401 Unauthorized (gated o caído).
- `bertin-project/bertin-roberta-base-spanish` — checkpoint 2021 con **bug de carga LayerNorm** (`gamma`/`beta` no mapean a `weight`/`bias` en transformers ≥4.45 → los LayerNorm preentrenados quedan sin cargar). Observado en los logs de la corrida real de NB04; descartado.
- `Qwen/Qwen3-Embedding-8B` — SOTA actual pero no cabe en T4 16 GB.

**Nota sobre mDeBERTa-v3 + estabilidad (verificado empíricamente, mayo 2026):** se intentó usarlo como primario pero **diverge a NaN en el primer paso de optimización incluso en fp32** con LR 1e-5 (NaN en step 1). Es la inestabilidad conocida de la disentangled attention. `dl_utils.train_model` incluye una **guardia NaN** (reintenta fp32 si fp16 falla) y `adam_eps` configurable, pero contra la divergencia en fp32 la única salida fiable fue **usar XLM-R** (multilingüe y estable, probado en NB06) como primario. mDeBERTa queda como intento best-effort en NB10 (LR 5e-6 + adam_eps 1e-6), excluido del ensemble si falla.
