from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import typer
import pandas as pd
from modules.config import FIGURES_DIR, PROCESSED_DATA_DIR
from wordcloud import WordCloud

app = typer.Typer()
sns.set_theme(style="white")


# =====================================================================
#  USER AND ENGAGEMENT CHARTS
# =====================================================================


def _clean_spines(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_top_active_members(top_users, top_n=10, output_path: Path = None):
    """Generate a horizontal bar chart of the top N most active members."""
    plt.figure(figsize=(10, 5))

    ax = sns.barplot(
        x=top_users.values,
        y=top_users.index,
        hue=top_users.index,
        palette="Blues_r",
        legend=False,
        edgecolor="none",
    )

    ax = plt.gca()
    _clean_spines(ax)
    plt.title(
        f"Top {top_n} de Miembros Más Activos", fontsize=13, fontweight="bold", pad=15
    )
    plt.xlabel("Cantidad de mensajes", fontsize=11)
    plt.ylabel("Remitente", fontsize=11)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.success(f"Gráfico de top usuarios guardado en {output_path}")

    return plt.gcf()


def plot_top_average_word_count(promedio_palabras, top_n=10, output_path: Path = None):
    """Generate bar chart for the average number of words per user."""
    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        x="word_count",
        y="sender",
        data=promedio_palabras,
        hue="sender",
        palette="Purples_r",
        legend=False,
        edgecolor="none",
    )

    _clean_spines(ax)
    plt.title(
        f"Top {top_n} promedio de palabras por mensaje de cada usuario",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Frecuencia promedio de palabras", fontsize=11)
    plt.ylabel("Usuario", fontsize=11)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.success(f"Gráfico de promedio de palabras guardado en {output_path}")

    return plt.gcf()


def plot_boxplot_message_size(
    df_original, df_user_stats, top_n=10, output_path: Path = None
):
    """
    Generate a boxplot showing the message size distribution
    for the top N most active users.
    """
    plt.figure(figsize=(11, 6))

    # Get the list of the top N most active users
    top_n_usuarios = df_user_stats.head(top_n)["sender"].tolist()

    # Filter the original dataframe to include only messages from those users
    usuarios = df_original[df_original["sender"] != "Sistema"].copy()
    usuarios["message_size"] = usuarios["message"].apply(lambda x: len(str(x)))
    df_top_n = usuarios[usuarios["sender"].isin(top_n_usuarios)]

    ax = sns.boxplot(
        data=df_top_n,
        x="message_size",
        y="sender",
        hue="sender",
        order=top_n_usuarios,
        palette="Blues",
        fliersize=3,
        linewidth=1.2,
        showfliers=False,
        legend=False,
    )

    _clean_spines(ax)
    plt.title(
        f"Distribución del tamaño de mensajes por miembro del GDG (Top {top_n} Activos)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Longitud del mensaje (cantidad de caracteres)", fontsize=11)
    plt.ylabel("Usuario", fontsize=11)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.success(f"Boxplot de tamaño de mensajes guardado en {output_path}")

    return plt.gcf()


def plot_short_users_scatter(df_short_users, output_path: Path = None):
    """
    Generate a horizontal dot plot to visualize the users
    with the shortest messages on average.
    """
    plt.figure(figsize=(10, 5))

    # Sort from shortest to longest
    df_plot = df_short_users.sort_values(by="avg_message_size", ascending=True)

    # Create a dot plot (Lollipop / Dot plot)
    ax = sns.scatterplot(
        data=df_plot,
        x="avg_message_size",
        y="sender",
        s=120,
        color="indianred",
        zorder=3,
    )

    # Add lollipop-style horizontal lines
    plt.hlines(
        y=df_plot["sender"],
        xmin=0,
        xmax=df_plot["avg_message_size"],
        color="lightgray",
        linestyle="--",
        linewidth=1.5,
        zorder=2,
    )

    _clean_spines(ax)
    plt.title(
        "Top miembros con mensajes más cortos", fontsize=13, fontweight="bold", pad=15
    )
    plt.xlabel("Promedio de caracteres por mensaje", fontsize=11)
    plt.ylabel("Miembros GDG", fontsize=11)
    plt.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.success(f"Gráfico de mensajes cortos guardado en {output_path}")

    return plt.gcf()


# =====================================================================
#  TEMPORAL PATTERN GRAPHS
# =====================================================================


def plot_messages_by_day(data_grouped, output_path: Path = None):
    """
    Generate a line chart showing the distribution
    of messages by day of the week.
    """
    plt.figure(figsize=(10, 4))

    ax = sns.lineplot(data=data_grouped, marker="o", color="royalblue", linewidth=2)
    _clean_spines(ax)
    plt.title(
        "Distribución de mensajes por día de la semana",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Día de la semana", fontsize=11)
    plt.ylabel("Cantidad de mensajes", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.success(f"Gráfico por día guardado en {output_path}")

    return plt.gcf()


def plot_messages_by_hour(data_grouped, output_path: Path = None):
    """
    Generate a line graph showing the distribution of messages
    by hour of the day.
    """
    plt.figure(figsize=(10, 4))

    ax = sns.lineplot(data=data_grouped, marker="o", color="seagreen", linewidth=2)

    _clean_spines(ax)
    plt.title(
        "Distribución de mensajes por hora del día",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Hora (24 hrs)", fontsize=11)
    plt.ylabel("Cantidad de mensajes", fontsize=11)
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.success(f"Gráfico por hora guardado en {output_path}")

    return plt.gcf()


def plot_messages_heatmap(heatmap_data, output_path: Path = None):
    """
    Generate a heat map showing the concentration of messages
    by hour and day.
    """
    plt.figure(figsize=(12, 6))

    ax = sns.heatmap(heatmap_data, cmap="YlGnBu", annot=False, fmt="d", linewidths=0.5)

    _clean_spines(ax)
    plt.title(
        "Mapa de Calor: Concentración de Mensajes por Hora y Día de la Semana",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Día de la semana", fontsize=11)
    plt.ylabel("Hora del día (0 - 23)", fontsize=11)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.success(f"Mapa de calor guardado en {output_path}")

    return plt.gcf()


# =====================================================================
#  TEXT GRAPHICS, EMOJIS, AND CONTENT
# =====================================================================


def plot_wordcloud(texto_limpio: str, output_path: Path = None):
    """Generate WordCloud."""
    if not texto_limpio or not texto_limpio.strip():
        logger.error("No se puede generar la nube de palabras: el texto está vacío.")
        return None

    wordcloud = WordCloud(width=800, height=600, background_color="white").generate(
        texto_limpio
    )

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.success(f"Nube de palabras guardada en {output_path}")

    return plt.gcf()


def plot_top_adjectives(df_adjetivos, top_n=10, output_path: Path = None):
    """Generates the horizontal bar graph for the Top of adjectives."""
    plt.figure(figsize=(10, 5))

    top_10 = df_adjetivos.head(top_n)

    ax = sns.barplot(
        data=top_10,
        x="count",
        y="adjetivo",
        hue="adjetivo",
        palette="Greens_r",
        legend=False,
        edgecolor="none",
    )

    _clean_spines(ax)
    plt.title(
        f"Top {top_n} Adjetivos más frecuentes en el chat",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Frecuencia", fontsize=12)
    plt.ylabel("Adjetivos", fontsize=12)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.success(f"Gráfico de adjetivos guardado en {output_path}")

    return plt.gcf()


def get_emoji_table(top_emojis_df):
    """
    Take the top emojis and apply the gradient style
    to display them formatted in the Jupyter Notebook.
    """
    df_tabla_emojis = top_emojis_df.copy()
    df_tabla_emojis.columns = ["Emoji", "Frecuencia"]

    return df_tabla_emojis.style.hide(axis="index").background_gradient(
        subset=["Frecuencia"], cmap="YlOrRd"
    )


def plot_top_domains(
    top_dominios_df, community_name="GDG Hermosillo", output_path: Path = None
):
    """
    Generate the horizontal bar chart of the most shared domains securely.
    """
    plt.figure(figsize=(10, 5))

    # Copy and ensure correct column names based on position
    df_plot = top_dominios_df.copy()

    if len(df_plot.columns) >= 2:
        df_plot.columns = ["dominio", "count"]

    df_plot = df_plot.sort_values(by="count", ascending=False)

    ax = sns.barplot(
        data=df_plot,
        x="count",
        y="dominio",
        hue="dominio",
        palette="Reds_r",
        legend=False,
        edgecolor="none",
    )

    _clean_spines(ax)
    plt.title(
        f"Plataformas y recursos más compartidos en {community_name}",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Cantidad de enlaces compartidos", fontsize=11)
    plt.ylabel("Dominio / Plataforma", fontsize=11)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.success(f"Gráfico de dominios guardado en {output_path}")

    return plt.gcf()


# =====================================================================
#  CLI (TYPER)
# =====================================================================

if __name__ == "__main__":
    app()
