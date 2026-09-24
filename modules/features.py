from pathlib import Path
import emoji
from loguru import logger
import pandas as pd
import typer
import spacy
from modules.config import PROCESSED_DATA_DIR
import re

app = typer.Typer()

# Load spaCy model
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    logger.error("El modelo de spaCy 'es_core_news_sm' no está instalado.")
    nlp = None

# =====================================================================
#  USER STATISTICS AND METRICS
# =====================================================================


def get_top_active_members(df, top_n=10) -> pd.Series:
    """
    Calculates and returns the message count
    of the most active members (excluding the System).
    """
    logger.info("Calculando miembros más activos...")
    return df[df["sender"] != "Sistema"]["sender"].value_counts().head(top_n)


def get_average_user_word_count(df, top_n=10) -> pd.DataFrame:
    """
    Calculate average number of words per message for the top N users.
    """
    logger.info("Calculando promedio de palabras por usuario...")

    # Copy and clean the DataFrame text.
    df_temp = df.copy()

    if "word_count" not in df_temp.columns:
        df_temp["word_count"] = df_temp["message"].apply(lambda x: len(str(x).split()))

    return (
        df_temp[df_temp["sender"] != "Sistema"]
        .groupby("sender")["word_count"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .to_frame()
        .reset_index()
    )


def get_user_statistics(df) -> pd.DataFrame:
    """
    Calculates key metrics per user (total messages, characters, and averages),
    excluding System.
    """
    logger.info("Calculando estadísticas de mensajes por usuario...")

    # Filter system messages
    users = df[df["sender"] != "Sistema"].copy()

    # Calculate the message length in characters
    users["message_size"] = users["message"].apply(lambda x: len(str(x)))

    # Group by user and calculate key metrics
    df_user_stats = (
        users.groupby("sender")
        .agg(
            total_messages=("message", "count"),  # Total messages
            total_characters=("message_size", "sum"),  # Total characters
            avg_message_size=("message_size", "mean"),  # Average characters per message
        )
        .reset_index()
    )

    # Sort from highest to lowest by message volume
    df_user_stats = df_user_stats.sort_values(by="total_messages", ascending=False)

    return df_user_stats


def get_users_short_messages(df, top_n=10) -> pd.DataFrame:
    """
    Calculates and returns the top N users with the lowest
    average number of characters per message (excluding System).
    """
    logger.info("Calculando usuarios con mensajes más cortos...")

    # Copy and clean the DataFrame text.
    df_temp = df.copy()

    # Ensure the message size column exists
    if "message_size" not in df_temp.columns:
        df_temp["message_size"] = df_temp["message"].apply(lambda x: len(str(x)))

    # Group by sender, calculate average size and total message count
    df_short_users = (
        df_temp[df_temp["sender"] != "Sistema"]
        .groupby("sender")
        .agg(
            avg_message_size=("message_size", "mean"),
            total_messages=("message", "count"),
        )
        .reset_index()
        .sort_values(by="avg_message_size", ascending=True)
        .head(top_n)
    )

    return df_short_users


# =====================================================================
#  TEMPORAL PATTERNS (DAYS, HOURS, AND HEAT MAP)
# =====================================================================


def _preprocess_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Internal function to ensure the datetime format
    and map days to Spanish in the correct order.
    """
    # Copy and clean the DataFrame text.
    df_temp = df.copy()

    if "timestamp" in df_temp.columns:
        df_temp["timestamp"] = pd.to_datetime(
            df_temp["timestamp"], format="mixed", errors="coerce"
        )

    # Extract time and day in English
    df_temp["hour"] = df_temp["timestamp"].dt.hour
    df_temp["day_name"] = df_temp["timestamp"].dt.day_name()

    # Translate and arrange the days in Spanish.
    translation_days = {
        "Monday": "Lunes",
        "Tuesday": "Martes",
        "Wednesday": "Miércoles",
        "Thursday": "Jueves",
        "Friday": "Viernes",
        "Saturday": "Sábado",
        "Sunday": "Domingo",
    }

    df_temp["day_name"] = df_temp["day_name"].map(translation_days)

    sorted_days_es = [
        "Lunes",
        "Martes",
        "Miércoles",
        "Jueves",
        "Viernes",
        "Sábado",
        "Domingo",
    ]

    df_temp["day_name"] = pd.Categorical(
        df_temp["day_name"], categories=sorted_days_es, ordered=True
    )

    return df_temp


def get_messages_by_day(df: pd.DataFrame) -> pd.Series:
    """
    Processes the timestamps and groups the messages
    by day of the week in Spanish.
    """
    logger.info("Procesando mensajes por día de la semana...")
    df_temp = _preprocess_timestamp(df)
    return df_temp.groupby("day_name", observed=False).size()


def get_messages_by_hour(df: pd.DataFrame) -> pd.Series:
    """Extract the times and group the messages by time of day (0-23)."""
    logger.info("Procesando mensajes por hora del día...")
    df_temp = _preprocess_timestamp(df)
    return df_temp.groupby("hour").size()


def get_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Generate the pivot table to cross-reference hours and days of the week."""
    logger.info("Generando tabla pivote para el mapa de calor...")
    df_temp = _preprocess_timestamp(df)
    val_col = "message" if "message" in df_temp.columns else df_temp.columns[0]
    return df_temp.pivot_table(
        index="hour", columns="day_name", values=val_col, aggfunc="count", fill_value=0
    )


# =====================================================================
#  TEXT PROCESSING (SPACY)
# =====================================================================


def get_adjective_frequency(clean_text: str) -> pd.DataFrame:
    """
    Recives a cleaned text string, extracts the adjectives using spaCy,
    and returns a Pandas DataFrame containing the adjectives and their counts, sorted.
    """
    if nlp is None:
        raise RuntimeError("SpaCy no pudo cargarse.")

    logger.info("Extrayendo adjetivos con spaCy desde el texto procesado...")

    # Process the cleaned text with spaCy.
    doc = nlp(clean_text)

    # Extract adjectives
    adjectives = [
        token.text.lower() for token in doc if token.pos_ == "ADJ" and not token.is_stop
    ]

    # Count frequencies and convert to a clean DataFrame
    df_result = pd.Series(adjectives).value_counts().reset_index()
    df_result.columns = ["adjetivo", "count"]

    return df_result


# =====================================================================
#  EXTRACTION OF EMOJIS AND RESOURCES/DOMAINS
# =====================================================================


def get_emoji_frequency(df, top_n=10) -> pd.DataFrame:
    """
    Extracts emojis from the original DataFrame, breaks them down,
    and returns the top N most used emojis along with their frequencies.
    """
    logger.info("Extrayendo y contando emojis del chat...")

    # Internal function to extract a list of emojis
    def extraer_emojis(texto):
        return [c["emoji"] for c in emoji.emoji_list(str(texto))]

    # Apply extraction to a copy of the dataframe
    df_temp = df.copy()
    df_temp["emoji_list"] = df_temp["message"].apply(extraer_emojis)

    # Explode so that each emoji has its own row
    df_emojis_expanded = df_temp.explode("emoji_list").dropna(subset=["emoji_list"])

    # Count frequencies and take the Top N
    top_emojis = (
        df_emojis_expanded["emoji_list"].value_counts().head(top_n).reset_index()
    )

    top_emojis.columns = ["emoji", "count"]

    return top_emojis


def get_domain(url):
    """Internal function to extract domain urls."""
    try:
        u = re.sub(r"https?://(www\.)?", "", str(url).strip().lower())
        domain = u.split("/")[0].split("?")[0].split("#")[0]

        if "youtube" in domain or "youtu.be" in domain:
            return "youtube.com"
        if "linkedin" in domain or "lnkd.in" in domain:
            return "linkedin.com"
        if "github" in domain:
            return "github.com"
        if "x.com" in domain or "twitter" in domain:
            return "x.com"

        return domain or "Desconocido"
    except Exception:
        return "Desconocido"


def get_top_shared_domains(df, top_n=10) -> pd.DataFrame:
    """Extracts URLs, consolidates key domains, and returns the top N."""
    logger.info("Extrayendo y procesando enlaces del chat...")

    # Select the 'message' column or use the first one by default.
    msg_col = "message" if "message" in df.columns else df.columns[0]

    # URL extractor
    url_pattern = r'https?://[^\s<>"]+?(?=[A-Z¡!¿?,\s]|$|https?://)'

    links = (
        df[msg_col]
        .astype(str)
        .apply(lambda x: re.findall(url_pattern, x))
        .explode()
        .dropna()
        .str.rstrip(".,;:!?")
        .apply(get_domain)
    )

    if links.empty:
        logger.warning("No se encontraron enlaces válidos.")
        return pd.DataFrame(columns=["dominio", "count"])

    return links.value_counts().head(top_n).reset_index(name="count")


# =====================================================================
#  CLI (TYPER)
# =====================================================================


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
):
    logger.info("Iniciando ejecución de características vía CLI...")
    df = pd.read_csv(input_path)

    # Example of direct processing of user statistics
    df_features = get_user_statistics(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)
    logger.success(f"Características guardadas exitosamente en {output_path}")


if __name__ == "__main__":
    app()
