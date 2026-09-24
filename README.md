#  GDG Hermosillo: Radiografía de una Comunidad en WhatsApp

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Contexto del proyecto

Las comunidades tecnológicas modernas se construyen y dinamizan a través de canales de mensajería instantánea. Este proyecto consiste en un **pipeline de análisis de datos y procesamiento de lenguaje natural (PLN)** aplicado al historial de chat de la comunidad **GDG Hermosillo**. 

El objetivo principal es transformar miles de mensajes crudos en **métricas accionables y visualizaciones limpias** que permitan entender el comportamiento, los intereses y los patrones de interacción de la comunidad.

> **Nota de Privacidad:** Toda la información procesada ha sido **estrictamente anonimizada** (removiendo nombres reales, números telefónicos y datos sensibles) para proteger la privacidad de los miembros, cumpliendo con buenas prácticas de manejo de datos.
> 
>  **Rango de Tiempo Analizado:** De **Noviembre 2025** a **Septiembre 2026**

---

### Dimensiones clave del análisis

* **Patrones Temporales:** Identificación de los momentos de mayor actividad de la comunidad (destacando picos los jueves por la noche y los sábados).
* **Perfiles de Participación:** Análisis del volumen de mensajes, longitud promedio, y diferenciación entre miembros activos, debates profundos y reacciones rápidas (emojis).
* **Procesamiento de Lenguaje Natural (PLN):** Extracción semántica de adjetivos clave y generación de nubes de palabras mediante `spaCy` para mapear el enfoque técnico del grupo (foco en IA, agentes y arquitectura).
* **Ecosistema de Recursos:** Mapeo de enlaces y plataformas externas más compartidas, revelando una preferencia por recursos audiovisuales y de desarrollo (`YouTube`, `LinkedIn` y `GitHub`).

---

### Stack tecnológico

El proyecto está desarrollado en **Python** bajo una arquitectura modular y limpia, utilizando librerías estándar de ciencia de datos:

* **Procesamiento y análisis:** `pandas`, `spaCy`, `NLTK`, `emoji`
* **Visualización:** `matplotlib`, `seaborn`, `wordcloud`
* **Automatización y arquitectura:** `Typer` (CLI), `loguru` (Logging estructurado)

---

## Organización del proycto

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         my_whatsapp_group_analytics and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── modules   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes modules folder a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    └── plots.py                <- Code to create visualizations
```

--------

