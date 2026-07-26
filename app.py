import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ============================================================
# Section 1 — Page Configuration
# ============================================================

st.set_page_config(
    page_title="Steam Game Review Sentiment Analysis",
    page_icon="🎮",
    layout="wide"
)




# ============================================================
# Section 3 — Load Data
# ============================================================

@st.cache_data
def load_data():
    return (
        pd.read_csv("data/game_statistics.csv"),
        pd.read_csv("data/top20_most_reviewed_games.csv"),
        pd.read_csv("data/review_distribution.csv"),
        pd.read_csv("data/highest_rated_games.csv"),
        pd.read_csv("data/lowest_rated_games.csv"),
        pd.read_csv("data/price_statistics.csv"),
        pd.read_csv("data/playtime_statistics.csv"),
        pd.read_csv("data/review_length_statistics.csv"),
    )

(game_stats, top_games, review_distribution,
 highest_rated, lowest_rated,
 price_stats, playtime_stats,
 review_length_stats) = load_data()

# ============================================================
# Section 4 — Load Machine Learning Model
# ============================================================

@st.cache_resource
def load_model():
    return (
        joblib.load("models/sentiment_model.pkl"),
        joblib.load("models/tfidf_vectorizer.pkl")
    )

model, vectorizer = load_model()

# ============================================================
# Section 5 — Sidebar
# ============================================================

st.sidebar.title("🎮 Dashboard Navigation")

page = st.sidebar.radio(
    "Select a Section",
    [
        "🏠 Home",
        "📊 Dataset Analytics",
        "🎮 Game Statistics",
        "🤖 Sentiment Prediction",
        "ℹ️ About",
    ],
)

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Project Summary")

st.sidebar.metric(
    "🎮 Games",
    len(game_stats)
)

st.sidebar.metric(
    "📝 Reviews",
    f"{int(game_stats['Total_Reviews'].sum()):,}"
)

st.sidebar.metric(
    "🎯 Accuracy",
    "90.06%"
)

st.sidebar.write("🤖 **Model:** Logistic Regression")
st.sidebar.write("📝 **Vectorizer:** TF-IDF")

st.sidebar.markdown("---")

st.sidebar.caption(
    "Built with Python • Streamlit • Scikit-Learn"
)
# ============================================================
# Section 6 — Home
# ============================================================

if page == "🏠 Home":

    st.title("🎮 Steam Game Review Sentiment Analysis")

    st.markdown("""
Welcome to the **Steam Game Review Sentiment Analysis Dashboard**.

This application analyzes over **730,000 Steam reviews** using
Natural Language Processing (NLP) and Machine Learning.

### Features

- 📊 Dataset Analytics
- 🎮 Interactive Game Statistics
- 🤖 AI Sentiment Prediction
- 📈 Machine Learning Insights
""")

    st.divider()

    total_reviews = int(game_stats["Total_Reviews"].sum())
    total_games = len(game_stats)

    positive_reviews = int(game_stats["Positive_Reviews"].sum())
    negative_reviews = total_reviews - positive_reviews

    positive_pct = positive_reviews / total_reviews * 100
    avg_price = game_stats["Average_Price"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("📝 Reviews", f"{total_reviews:,}")
    c2.metric("🎮 Games", total_games)
    c3.metric("😊 Positive", f"{positive_pct:.1f}%")
    c4.metric("😞 Negative", f"{100-positive_pct:.1f}%")
    c5.metric("💰 Avg Price", f"${avg_price:.2f}")

    st.divider()

    st.subheader("📊 Top 10 Most Reviewed Games")

    chart = (
        top_games
        .head(10)
        .set_index("name")["Total_Reviews"]
    )

    st.bar_chart(chart)

# ============================================================
# Section 7 — Dataset Analytics
# ============================================================

elif page == "📊 Dataset Analytics":

    st.header("📊 Dataset Analytics")

    # ------------------------------------------------------------
    # Top 20 Most Reviewed Games
    # ------------------------------------------------------------

    st.subheader("🎮 Top 20 Most Reviewed Games")

    top_chart = (
        top_games
        .head(20)
        .set_index("name")["Total_Reviews"]
    )

    st.bar_chart(top_chart)

    st.dataframe(
        top_games,
        use_container_width=True
    )

    st.divider()

    # ------------------------------------------------------------
    # Highest Rated Games
    # ------------------------------------------------------------

    st.subheader("⭐ Highest Rated Games")

    highest_chart = (
        highest_rated
        .head(10)
        .set_index("name")["Positive_Percentage"]
    )

    st.bar_chart(highest_chart)

    st.dataframe(
        highest_rated.head(20),
        use_container_width=True
    )

    st.divider()

    # ------------------------------------------------------------
    # Lowest Rated Games
    # ------------------------------------------------------------

    st.subheader("👎 Lowest Rated Games")

    lowest_chart = (
        lowest_rated
        .head(10)
        .set_index("name")["Positive_Percentage"]
    )

    st.bar_chart(lowest_chart)

    st.dataframe(
        lowest_rated.head(20),
        use_container_width=True
    )

    st.divider()

    # ------------------------------------------------------------
    # Review Distribution
    # ------------------------------------------------------------

    st.subheader("📈 Review Distribution")

    review_chart = review_distribution.set_index("Sentiment")

    st.bar_chart(review_chart)

    st.dataframe(
        review_distribution,
        use_container_width=True
    )
# ============================================================
# Section 8 — Game Statistics
# ============================================================

elif page == "🎮 Game Statistics":

    st.header("🎮 Game Statistics")

    selected = st.selectbox(
        "Select a Game",
        sorted(game_stats["name"].unique())
    )

    game = game_stats[
        game_stats["name"] == selected
    ].iloc[0]

    st.subheader(f"🎮 {selected}")

    st.caption("Detailed statistics for the selected Steam game.")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "📝 Total Reviews",
            f"{int(game['Total_Reviews']):,}"
        )

    with col2:
        st.metric(
            "😊 Positive Rating",
            f"{game['Positive_Percentage']:.1f}%"
        )

    with col3:
        st.metric(
            "💰 Average Price",
            f"${game['Average_Price']:.2f}"
        )

    st.divider()

    col4, col5 = st.columns(2)

    with col4:
        st.metric(
            "👍 Positive Reviews",
            f"{int(game['Positive_Reviews']):,}"
        )

    with col5:
        st.metric(
            "👎 Negative Reviews",
            f"{int(game['Negative_Reviews']):,}"
        )

    st.divider()

    review_chart = pd.DataFrame(
        {
            "Reviews": [
                int(game["Positive_Reviews"]),
                int(game["Negative_Reviews"])
            ]
        },
        index=["Positive", "Negative"]
    )

    st.subheader("📊 Review Breakdown")

    st.bar_chart(review_chart)
# ============================================================
# Section 9 — Sentiment Prediction
# ============================================================
elif page == "🤖 Sentiment Prediction":

    st.header("🤖 AI Sentiment Prediction")

    st.markdown("""
    Enter a Steam game review below and let the trained Machine Learning model
    predict whether the sentiment is **Positive** or **Negative**.

    The model was trained using **TF-IDF** text vectorization and a
    **Logistic Regression** classifier.
    """)

    review = st.text_area(
        "✍️ Enter a Review",
        value=st.session_state.get("review", ""),
        height=200,
        placeholder="Example: This game has amazing graphics and addictive gameplay..."
    )

    if st.button("Predict Sentiment"):

        if not review.strip():
            st.warning("Please enter a review.")

        else:
            vec = vectorizer.transform([review])
            prediction = model.predict(vec)[0]
            confidence = max(model.predict_proba(vec)[0]) * 100

            st.divider()

            st.subheader("Prediction Result")

            c1, c2 = st.columns(2)

            if prediction:
                c1.success("😊 Positive Review")
            else:
                c1.error("😞 Negative Review")

            c2.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.divider()

            st.subheader("Model Information")

            st.write("**Algorithm:** Logistic Regression")
            st.write("**Vectorizer:** TF-IDF")
            st.write("**Training Accuracy:** 90.06%")

# ============================================================
# Section 10 — About
# ============================================================

elif page == "ℹ️ About":
    st.header("ℹ️ About This Project")
    st.markdown("""
### Steam Game Review Sentiment Analysis

- Python
- Streamlit
- Pandas
- Scikit-Learn
- TF-IDF
- Logistic Regression

Dataset:
- 730,945 Steam Reviews
- 735 Games

Model Accuracy: **90.06%**
""")

