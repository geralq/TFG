# Trabajo de Fin de Grado: Interpretación del lenguaje de signos español basada en visión por computador y aprendizaje profundo  

Este repositorio contiene el **código**, **cuadernos** y la **memoria** del TFG orientado a la interpretación de la **Lengua de Signos Española (LSE)** con técnicas de visión por computador y aprendizaje profundo. Incluye **preprocesado de datos de vídeo**, **modelado secuencial (RNN/TCN)**, **detección temporal (spotting/segmentación)** y **experimentos de detección de novedad (open-set)**, además de la memoria en PDF.

This repository contains the **code**, **notebooks**, and the **thesis document** for a project on **Spanish Sign Language (LSE) interpretation** using computer vision and deep learning. It covers **video data preprocessing**, **sequence models (RNN/TCN)**, **temporal detection (spotting/segmentation)**, and **open-set (novelty) detection** experiments, plus the thesis PDF.

---

## Descripción breve / Quick Overview

**ES:** Se implementan pipelines para extraer características por frame a partir de vídeo, entrenar modelos secuenciales (RNN/TCN) para **gestos aislados** y **detección/segmentación temporal**, y explorar **detección de novedad** mediante **embeddings**.

**EN:** Pipelines are implemented to extract **per-frame features** from video, train **sequence models (RNN/TCN)** for **isolated signs** and **temporal detection/segmentation**, and explore **open-set detection** via **embeddings**.

---

## Estructura / Repository Structure

### Carpetas / Folders
- `data_preparations/` — Scripts y utilidades para **preparación de datasets** (limpieza, normalización, generación de ventanas/etiquetas).
- `data_processing/` — **Procesado de datos**: lectura de anotaciones y extracción/transformación de características.
- `video_processing/` — **Preprocesado de vídeo** y extracción de landmarks/poses por frame.
- `train_test_val_split/` — Lógica para **división de datos** en train/val/test.
- `models/` — Definiciones de **arquitecturas** utilizadas (p. ej., RNN/TCN).
- `loss_functions/` — Utilidades y **funciones de pérdida** para entrenamiento.
- `evaluaciones/` — Recursos para **evaluación** y análisis.

### Archivos principales / Key Files
- `train_model.py` — Entrenamiento para **gestos aislados** con modelos RNN.
- `train_TCN.py` — Entrenamiento de **TCN** para **detección/segmentación temporal**.
- `triplet_train.py` — Entrenamiento con **triplet loss** para aprender **embeddings**.
- `test_models.py`, `test_iuc.py` — **Pruebas** y comprobaciones rápidas.
- `spot_detection.ipynb` — **Spotting** de gestos en secuencias largas.
- `segmentation_dataset.ipynb` — Construcción/inspección de **datasets de segmentación temporal**.
- `novelty_dataset.ipynb`, `novelty_detection.ipynb` — Dataset y experimentos de **open-set/novelty**.
- `Memoria_TFG.pdf` — **Memoria** del proyecto (PDF).
