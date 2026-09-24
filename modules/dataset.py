from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import re
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
import unicodedata
from modules.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    COMMUNITY_EXCEPTIONS,
    BLOCKED_WORDS,
    DEFAULT_BLOCKLIST,
)
import emoji

app = typer.Typer()

# Load spaCy
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    nlp = None
    logger.warning(
        "El modelo de spaCy no está disponible. La anonimización de nombres dependerá solo de reglas secundarias."
    )

# =====================================================================
#  TEXT PROCESSING AND SEMANTIC ANALYSIS (SPACY / NLTK)
# =====================================================================


def _sanitize_name(line):
    """Cleans invisible Unicode control characters and normalizes spaces in a WhatsApp chat line."""
    if not line:
        return "[Sistema]"
    clean_line = re.sub(r"[\u200e\u200f\u202f\u2068\u2069]", " ", line)
    clean_line = re.sub(r"(\s*-\s*)[\s~]+", r"\1", clean_line).strip()
    return clean_line


def redact_sensitive_info(message: str) -> str:
    """
    Cleans invisible characters and anonymizes sensitive information in the message.
    """
    message = str(message)

    # Basic initial cleaning
    message = re.sub(r"[\u200e\u200f]", "", message)
    message = re.sub(
        r"(?:\w*\[(?!NOMBRE_OCULTO|NUMERO_OCULTO|CORREO_OCULTO)[^\]]+\]\w*)+",
        "[NOMBRE_OCULTO]",
        message,
    )
    message = re.sub(r"\+\d{1,3}(?:[\s().-]*\d+)+", "[NUMERO_OCULTO]", message)
    message = re.sub(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[CORREO_OCULTO]", message
    )
    message = re.sub(
        r"~[\s\u202f]*[A-Za-zÁÉÍÓÚáéíóúÑñ]+(?:\s+[A-Za-zÁÉÍÓÚáéíóúÑñ]+)*",
        "[NOMBRE_OCULTO]",
        message,
    )

    # Isolate URLs and Emojis
    url_pattern = r"https?://\S+|www\.\S+"
    urls_found = re.findall(url_pattern, message)
    for i, url in enumerate(urls_found):
        message = message.replace(url, f"___URL_PLACEHOLDER_{i}___")

    emojis_found = [c["emoji"] for c in emoji.emoji_list(message)]
    for i, emo in enumerate(emojis_found):
        message = message.replace(emo, f"___EMOJI_PLACEHOLDER_{i}___")

    # Rule for scripted presentations
    pattern_presentation_script = r"\b(?:Dra?|Lic|Ing|Mtro?|Prof?)\.?\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+(?:de\s+la|del|de|la|Los|Las|[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+))+\s*-\s*"
    if re.search(pattern_presentation_script, message):
        match_desc = re.search(
            r"\b(?:Dra?|Lic|Ing|Mtro?|Prof?)\.?\s+(?:.*?\s*)-\s*(.*)", message
        )
        if match_desc:
            description = match_desc.group(1)
            message = f"[NOMBRE_OCULTO] - {description}"
    else:
        if nlp is not None:
            prefixes_pattern = r"\b(?:Dra?|Lic|Ing|Mtro?|Prof?)\.\s*"
            message_spacy = re.sub(prefixes_pattern, "", message)
            doc = nlp(message_spacy)

            entities_per = sorted(
                [e for e in doc.ents if e.label_ == "PER"],
                key=lambda x: x.start_char,
                reverse=True,
            )

            for ent in entities_per:
                entity_text = ent.text.strip()

                is_false_positive = (
                    entity_text in COMMUNITY_EXCEPTIONS
                    or entity_text in BLOCKED_WORDS
                    or any(p in BLOCKED_WORDS for p in entity_text.split())
                    or any(
                        entity_text.lower() in phrase.lower()
                        or phrase.lower() in entity_text.lower()
                        for phrase in COMMUNITY_EXCEPTIONS
                    )
                    or len(entity_text.split()) < 2
                )

                if not is_false_positive:
                    message = message.replace(entity_text, "[NOMBRE_OCULTO]")

    # Restore Emojis and URLs
    for i, emo in enumerate(emojis_found):
        message = message.replace(f"___EMOJI_PLACEHOLDER_{i}___", emo)

    for i, url in enumerate(urls_found):
        message = message.replace(f"___URL_PLACEHOLDER_{i}___", url)

    return message


def anonymize_chat(input_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Processes and anonymizes a WhatsApp chat history: assigns consistent pseudonyms (`Miembro_GDG_N`)
    to senders, masks sensitive information, and processes multi-line messages.
    """
    pattern = re.compile(
        r"^\s*[\u200e\u200f]?(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\s*(?:[ap]\.?\s?m\.?)?)\s*-\s*(?:([^:]+):\s*)?(.*)$",
        re.IGNORECASE,
    )

    # Initializes structures to map real names to anonymous pseudonyms
    name_mapping = {}
    user_counter = 1
    anonymized_lines = []
    list_anonymized = []

    input_path_obj = Path(input_path)
    logger.info(f"Leyendo archivo crudo: {input_path_obj.name}")

    # Open and iterate through the WhatsApp chat file line by line
    with open(input_path, "r", encoding="utf-8") as file:
        for line in file:
            clean_text = _sanitize_name(line)
            match = pattern.match(clean_text)
            # If the line matches the message format (Timestamp - Sender: Message)
            if match:
                timestamp, sender, message = match.groups()
                sender = sender.strip() if sender else "Sistema"
                # Assigns a unique anonymous identifier for each new user
                if sender not in name_mapping:
                    name_mapping[sender] = (
                        f"Miembro_GDG_{user_counter}"
                        if sender != "Sistema"
                        else "Sistema"
                    )
                    user_counter += 1

                anonymous_sender = name_mapping[sender]
                message_redacted = redact_sensitive_info(message)
                # Reconstruct the line with the anonymized sender and the protected content
                anonymized_line = (
                    f"[{timestamp}] {anonymous_sender}: {message_redacted}\n"
                )

                anonymized_lines.append(anonymized_line)
                list_anonymized.append([timestamp, anonymous_sender, message_redacted])
            # If the line does not match, it is a line break (multi-line message)
            else:
                if line.strip():
                    san = redact_sensitive_info(clean_text)
                    anonymized_lines.append(san + "\n")
                    if list_anonymized:
                        last_index = len(list_anonymized) - 1
                        list_anonymized[last_index][2] = (
                            list_anonymized[last_index][2] + "\n" + san
                        )

    df_anonymized = pd.DataFrame(
        list_anonymized, columns=["timestamp", "sender", "message"]
    )

    # Secure output directories
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save anonymized plain-text backup
    with open(output_path.with_suffix(".txt"), "w", encoding="utf-8") as file:
        file.writelines(anonymized_lines)

    output_path_obj = Path(input_path)
    logger.success(f"¡Chat anonimizado con éxito! Guardado en: {output_path_obj.name}")
    logger.info(f"Mapeo de usuarios generado con éxito.")

    return df_anonymized


def _remove_accents(_texto):
    """Internal function to remove accents."""
    # 1. Temporarily protect the 'ñ' with a unique placeholder.
    temp_texto = _texto.replace("ñ", "__ENYE_TEMP__")

    # 2. Normalize and remove accent marks (Mn)
    nfkd_form = unicodedata.normalize("NFD", temp_texto)
    temp_texto_wo_accents = "".join(
        [c for c in nfkd_form if not unicodedata.category(c) == "Mn"]
    )

    # 3. Restore the original 'ñ'
    return temp_texto_wo_accents.replace("__ENYE_TEMP__", "ñ")


def clean_chat_text(df, custom_stopwords=None, text_column=None):
    """Clean the text, remove URLs, accents, and filter stopwords from a DataFrame."""
    # Ensure download of stopwords from nltk
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    # Configure base stop words and add custom blocks
    my_stopwords = set(stopwords.words("spanish"))

    if custom_stopwords:
        DEFAULT_BLOCKLIST.extend(custom_stopwords)

    my_stopwords.update(DEFAULT_BLOCKLIST)

    # Copy and clean the DataFrame text.
    df_temp = df.copy()

    # Dynamically detect the text column if not specified.
    if not text_column:
        text_column = "message" if "message" in df_temp.columns else df_temp.columns[0]

    df_temp[text_column] = (
        df_temp[text_column].astype(str).str.replace("_", " ", regex=False)
    )

    # Replace URLs
    patron_url = r"https?://\S+|www\.\S+|\S+\.\S+/\S+|(?:\w+\.)+(?:com|org|net|edu|gob|mx|es)\b\S*"
    df_temp[text_column] = df_temp[text_column].str.replace(
        patron_url, "", regex=True, case=False
    )

    # Join all the text into one giant string
    join_text = " ".join(df_temp[text_column].tolist())

    # Cleaning of symbols, lowercase letters, and accents
    join_text = join_text.lower()
    join_text = re.sub(r"[^a-zA-ZáéíóúÁÉÍÓÚñÑ0-9\s]", "", join_text)
    join_text = _remove_accents(join_text)

    # Filter words using stop words.
    clean_join_text = " ".join(
        word for word in join_text.split() if word not in my_stopwords
    )

    if not clean_join_text.strip():
        logger.warning("El texto resultante después de la limpieza está vacío.")

    return clean_join_text


# =====================================================================
#  CLI (TYPER)
# =====================================================================


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "chat_whatsapp.txt",
    output_path: Path = PROCESSED_DATA_DIR / "chat_anonymized.txt",
):
    """Processes and anonymizes raw WhatsApp chat data."""
    logger.info("Iniciando pipeline de procesamiento de datos...")
    anonymize_chat(input_path, output_path)
    logger.success("¡Proceso completado!")


if __name__ == "__main__":
    app()
