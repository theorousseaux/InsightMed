import streamlit as st
import altair as alt
import pandas as pd
import random
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("Trends and Sentiments :chart_with_upwards_trend:")

features_df = pd.read_csv("generation/features.csv")
features_df["cancer_types"] = features_df["cancer_types"].apply(
    eval
)  # Convert string to list

articles_df = (
    features_df.drop(
        columns=["treatment_name", "treatment_description", "treatment_sentiment"]
    )
    .drop_duplicates(subset=["article_name"])
    .reset_index(drop=True)
)
cancer_df = articles_df.explode("cancer_types").reset_index(drop=True)

st.markdown("## Cancer types repartition")

cancer_repartition_chart = (
    alt.Chart(cancer_df[cancer_df["cancer_types"].isna() == False])
    .mark_bar()
    .encode(
        y=alt.Y(
            "cancer_types",
            title="Cancer Type",
            sort="-x",
            axis=alt.Axis(labelLimit=200, labelPadding=10),
        ),
        x="count()",
        color="article_type",
    )
    .properties(title="Number of articles per cancer type")
)

st.altair_chart(cancer_repartition_chart, use_container_width=True)

st.markdown("---")

st.markdown("## Articles repartition")

articles_repartition_barchart = (
    alt.Chart(articles_df)
    .mark_bar()
    .encode(x="article_type", y="count()")
    .properties(title="Number of articles per type")
)

articles_repartition_piechart = (
    alt.Chart(articles_df)
    .mark_arc()
    .encode(
        theta="count()",
        color="article_type",
    )
    .properties(title="Number of articles per type")
)

articles_date_repartition_chart = (
    alt.Chart(articles_df)
    .mark_bar()
    .encode(x="article_date", y="count()", color="article_type")
    .properties(title="Number of articles per date")
)

col1, col2 = st.columns(2)
with col1:
    st.altair_chart(articles_repartition_barchart, use_container_width=True)
with col2:
    st.altair_chart(articles_repartition_piechart, use_container_width=True)
st.altair_chart(articles_date_repartition_chart, use_container_width=True)

st.markdown("---")

st.markdown("## Treatments")

k = st.number_input(
    "Number of top mentioned treatments to display", min_value=1, value=5
)

sentiment_order = ["Positive", "Optimistic", "Neutral", "Cautious", "Pessimistic"]
sentiment_colors_dict = {
    "Positive": "#A2D9A4",  # Vert doux
    "Optimistic": "#C5E1A5",  # Vert citronné
    "Neutral": "#FFECB3",  # Beige
    "Cautious": "#FFCC80",  # Orange pastel
    "Pessimistic": "#FF8A80",  # Rouge clair
}

sentiment_colors = [sentiment_colors_dict[sentiment] for sentiment in sentiment_order]

top_k = features_df["treatment_name"].value_counts()[:k]
top_k_features_df = features_df[features_df["treatment_name"].isin(top_k.index)]

# Créer le graphique Altair
treatment_repartition_chart = (
    alt.Chart(top_k_features_df)
    .mark_bar(size=30)
    .encode(
        y=alt.Y("treatment_name", title="Treatment name", sort="-x"),
        x=alt.X("count()").title("Number of mentions"),
        color=alt.Color(
            "treatment_sentiment:N",
            scale=alt.Scale(domain=sentiment_order, range=sentiment_colors),
        ).title("Sentiment"),
    )
    .properties(title="Number of articles mentioning treatment", height=400)
)

st.altair_chart(treatment_repartition_chart, use_container_width=True)

# Compter les occurrences de chaque paire traitement-sentiment
treatment_sentiment_counts = (
    top_k_features_df.groupby(["treatment_name", "treatment_sentiment"])
    .size()
    .reset_index(name="count")
)
# Assurer que les sentiments sont dans l'ordre fixe
treatment_sentiment_counts["treatment_sentiment"] = pd.Categorical(
    treatment_sentiment_counts["treatment_sentiment"],
    categories=sentiment_order,
    ordered=True,
)
treatment_sentiment_counts = treatment_sentiment_counts.sort_values(
    "treatment_sentiment"
)

# Créer des listes pour les sources, cibles et valeurs
source = []
target = []
value = []

# Créer un dictionnaire pour mapper les traitements et les sentiments à des indices
treatment_dict = {
    treatment: i
    for i, treatment in enumerate(top_k_features_df["treatment_name"].unique())
}
sentiment_dict = {
    sentiment: i + len(treatment_dict) for i, sentiment in enumerate(sentiment_order)
}

# Remplir les listes source, target et value
for _, row in treatment_sentiment_counts.iterrows():
    source.append(treatment_dict[row["treatment_name"]])
    target.append(sentiment_dict[row["treatment_sentiment"]])
    value.append(row["count"])

# Créer une liste de couleurs pour les nœuds
colors = [
    "#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])
    for _ in range(len(treatment_dict))
]

colors.extend(sentiment_colors)

# Créer le texte personnalisé pour les nœuds de traitement
node_labels = list(treatment_dict.keys()) + list(sentiment_dict.keys())
node_customdata = []
for treatment in treatment_dict.keys():
    articles = top_k_features_df[top_k_features_df["treatment_name"] == treatment][
        "article_name"
    ].unique()
    node_customdata.append("<br>".join(articles))
node_customdata.extend([""] * len(sentiment_dict))

# Créer le diagramme de Sankey
fig = go.Figure(
    data=[
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line={"color": "black", "width": 0.5},
                label=node_labels,
                customdata=node_customdata,
                hovertemplate="%{label}<br>%{customdata}<extra></extra>",
                color=colors,
            ),
            link=dict(source=source, target=target, value=value),
        )
    ]
)

# Mettre en forme le diagramme de Sankey
fig.update_layout(
    title_text="Top {} most mentionned Treatments and sentiments assiociated".format(k),
    font_size=10,
)

st.plotly_chart(fig)

with st.expander("Show features"):
    st.write(features_df)
