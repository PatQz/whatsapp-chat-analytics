from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
# logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Configuration lists for chat data anonymization
COMMUNITY_EXCEPTIONS = {
    "Hermosillo Community",
    "Nuevo Video",
    "Claude Code",
    "PyCharm",
    "Rider",
    "Pydantic",
    "Custom MCP Servers",
    "Gemini Multimodal",
    "Semicon Desert",
    "AiSalon",
    "Knowledge Graph",
    "Vibe Engineering",
    "Loop Engineering",
    "Google",
    "Antigravity",
    "Python",
    "Vibe Video Studio",
    "Taller Práctico",
    "Multi-Agentes de IA",
    "Refinar el Blueprint",
    "Resumen Técnico",
    "IDEs Agentic-first",
    "Workshop Especializado",
    "Auditoría de PRs",
    "Arquitectura y Desarrollo de Agentes de IA",
    "Analyst Agent",
    "Researcher Agent",
    "Critic & Refiner Agents",
    "Calendario Inteligente: Tabla",
    "Cloud Infra",
    "Custom Workflows",
    "Maura Health",
    "[LIVE CODING]",
    "Validar Atributos de Calidad",
    "Evaluación Continua",
    "Mención Honorífica",
    "Control de Comportamiento y Capacidades",
    "Senior Backend Developer",
    "Spec-Driven Development",
}

BLOCKED_WORDS = {
    "Pfff",
    "Rigor",
    "Aprenderás",
    "Aprenderas",
    "Explora",
    "Solo",
    "Caleño",
    "Fork",
    "Adicional",
    "Comparte",
    "Inglés",
    "Miércoles",
    "H2H",
    "Coding",
    "Sistemas",
    "Zonas",
    "Mucho",
    "daré",
    "Skills",
    "Fecha",
    "Captura",
    "Justificar",
    "Patrones",
    "Workshop",
    "Ingles",
    "Miercoles",
}

DEFAULT_BLOCKLIST = [
    "multimedia",
    "omitido",
    "numero",
    "oculto",
    "cambio",
    "toca",
    "obtener",
    "mas",
    "informacion",
    "seguridad",
    "unio",
    "enlace",
    "grupo",
    "comunidad",
    "sticker",
    "audio",
    "codigo",
    "nombre",
    "añadio",
    "si",
    "link",
    "hola",
    "dia",
    "solo",
    "aqui",
    "edito",
    "mensaje",
    "jaja",
    "jajaja",
    "jejeje",
    "utm",
    "source",
    "medium",
    "campaign",
    "content",
    "sourceshareutm",
    "http",
    "https",
    "www",
]

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
