import pandas as pd

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("data/steam_reviews.csv")

print("=" * 60)
print("STEAM GAME DATA EXPLORATION")
print("=" * 60)

print(f"Dataset Size   : {len(df):,} Reviews")
print(f"Games Analyzed : {df['name'].nunique():,}")

# ============================================================
# Section 1 — Top 20 Most Reviewed Games:
# ============================================================

print("\n" + "=" * 60)
print("SECTION 1 — TOP 20 MOST REVIEWED GAMES")
print("=" * 60)

top20_games = (
    df.groupby("name")
      .size()
      .sort_values(ascending=False)
      .head(20)
      .reset_index(name="Total_Reviews")
)

print(top20_games.to_string(index=False))

top20_games.to_csv(
    "data/top20_most_reviewed_games.csv",
    index=False
)

print("\nSaved:")
print("data/top20_most_reviewed_games.csv")

# ============================================================
# Section 2 — Review Distribution:
# ============================================================

print("\n" + "=" * 60)
print("SECTION 2 — REVIEW DISTRIBUTION")
print("=" * 60)

review_distribution = (
    df["voted_up"]
    .value_counts()
    .rename_axis("Sentiment")
    .reset_index(name="Total_Reviews")
)

review_distribution["Percentage"] = (
    review_distribution["Total_Reviews"]
    /
    review_distribution["Total_Reviews"].sum()
    * 100
).round(2)

print(review_distribution.to_string(index=False))

review_distribution.to_csv(
    "data/review_distribution.csv",
    index=False
)

print("\nSaved:")
print("data/review_distribution.csv")

# ============================================================
# Section 3 — Review Distribution:
# ============================================================

print("\n" + "=" * 60)
print("SECTION 3 — HIGHEST RATED GAMES")
print("=" * 60)

game_stats = pd.read_csv("data/game_statistics.csv")

highest_rated = (
    game_stats
    .sort_values(
        by="Positive_Percentage",
        ascending=False
    )
    .head(20)
)

print(highest_rated.to_string(index=False))

highest_rated.to_csv(
    "data/highest_rated_games.csv",
    index=False
)

print("\nSaved:")
print("data/highest_rated_games.csv")

# ============================================================
# Section 4 — Lowest Rated Games
# ============================================================

print("\n" + "=" * 60)
print("SECTION 4 — LOWEST RATED GAMES")
print("=" * 60)

lowest_rated = (
    game_stats[
        game_stats["Total_Reviews"] >= 100
    ]
    .sort_values(
        by="Positive_Percentage",
        ascending=True
    )
    .head(20)
)

print(lowest_rated.to_string(index=False))

lowest_rated.to_csv(
    "data/lowest_rated_games.csv",
    index=False
)

print("\nSaved:")
print("data/lowest_rated_games.csv")



# ============================================================
# Section 5 — Price Analysis
# ============================================================

print("\n" + "=" * 60)
print("SECTION 5 — PRICE ANALYSIS")
print("=" * 60)

print(f"Minimum Game Price : ${game_stats['Average_Price'].min():.2f}")
print(f"Maximum Game Price : ${game_stats['Average_Price'].max():.2f}")
print(f"Average Game Price : ${game_stats['Average_Price'].mean():.2f}")

price_statistics = (
    game_stats[
        [
            "name",
            "Average_Price"
        ]
    ]
    .sort_values(
        by="Average_Price",
        ascending=False
    )
)

display_price = price_statistics.head(20).copy()

display_price["Average_Price"] = (
    "$" +
    display_price["Average_Price"].map("{:.2f}".format)
)

print("\nTop 20 Most Expensive Games:\n")
print(display_price.to_string(index=False))

price_statistics.to_csv(
    "data/price_statistics.csv",
    index=False
)

print("\nSaved:")
print("data/price_statistics.csv")

# ============================================================
# Section 6 — Playtime Analysis
# ============================================================

print("\n" + "=" * 60)
print("SECTION 6 — PLAYTIME ANALYSIS")
print("=" * 60)

# Function to convert minutes into days, hours, and minutes
def format_playtime(minutes):
    minutes = int(round(minutes))

    days = minutes // (24 * 60)
    hours = (minutes % (24 * 60)) // 60
    mins = minutes % 60

    if days > 0:
        return f"{days} days {hours} hours"

    elif hours > 0:
        return f"{hours} hours {mins} minutes"

    else:
        return f"{mins} minutes"

# Summary Statistics
print(f"Minimum Average Playtime : {format_playtime(game_stats['Average_Playtime_Minutes'].min())}")
print(f"Maximum Average Playtime : {format_playtime(game_stats['Average_Playtime_Minutes'].max())}")
print(f"Average Playtime         : {format_playtime(game_stats['Average_Playtime_Minutes'].mean())}")

# Sort by playtime
playtime_statistics = (
    game_stats[
        [
            "name",
            "Average_Playtime_Minutes"
        ]
    ]
    .sort_values(
        by="Average_Playtime_Minutes",
        ascending=False
    )
)

# Create a display version for printing
display_playtime = playtime_statistics.head(20).copy()

display_playtime["Average_Playtime"] = (
    display_playtime["Average_Playtime_Minutes"]
    .apply(format_playtime)
)

display_playtime.drop(
    columns=["Average_Playtime_Minutes"],
    inplace=True
)

print("\nTop 20 Games by Average Playtime:\n")
print(display_playtime.to_string(index=False))

# Save original numeric data
playtime_statistics.to_csv(
    "data/playtime_statistics.csv",
    index=False
)

print("\nSaved:")
print("data/playtime_statistics.csv")



# ============================================================
# Section 7 — Review Length Analysis
# ============================================================

print("\n" + "=" * 60)
print("SECTION 7 — REVIEW LENGTH ANALYSIS")
print("=" * 60)

print(f"Shortest Average Review : {game_stats['Average_Review_Length'].min()} words")
print(f"Longest Average Review  : {game_stats['Average_Review_Length'].max()} words")
print(f"Average Review Length   : {game_stats['Average_Review_Length'].mean():.1f} words")

review_length_statistics = (
    game_stats[
        [
            "name",
            "Average_Review_Length"
        ]
    ]
    .sort_values(
        by="Average_Review_Length",
        ascending=False
    )
)

print("\nTop 20 Games with the Longest Average Reviews:\n")
print(review_length_statistics.head(20).to_string(index=False))

review_length_statistics.to_csv(
    "data/review_length_statistics.csv",
    index=False
)

print("\nSaved:")
print("data/review_length_statistics.csv")

