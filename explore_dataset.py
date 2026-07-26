import pandas as pd

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("data/steam_reviews.csv")

# ============================================================
# STEAM GAME REVIEW DATASET EXPLORATION
# ============================================================

print("=" * 60)
print("STEAM GAME REVIEW DATASET EXPLORATION")
print("=" * 60)

print(f"Dataset Size   : {len(df):,} Reviews")
print(f"Games Analyzed : {df['name'].nunique():,}")

# ============================================================
# Section 1 — Dataset Overview
# ============================================================

print("\n" + "=" * 60)
print("SECTION 1 — DATASET OVERVIEW")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

if df["release_date"].isna().all():
    print("\nNote:")
    print("The release_date column is empty for all records and will not be used.")

print("\nData Types:")
print(df.dtypes)

print(f"\nTotal Unique Games: {df['name'].nunique():,}")

print("\nReview Label Distribution:")

review_counts = df["voted_up"].value_counts()

positive = review_counts.get(True, 0)
negative = review_counts.get(False, 0)
total = positive + negative

print(f"Positive Reviews : {positive:,} ({positive/total:.1%})")
print(f"Negative Reviews : {negative:,} ({negative/total:.1%})")

# ============================================================
# Section 2 — Top 20 Most Reviewed Games
# ============================================================

print("\n" + "=" * 60)
print("SECTION 2 — TOP 20 MOST REVIEWED GAMES")
print("=" * 60)

top_games = (
    df.groupby("name")
      .size()
      .sort_values(ascending=False)
      .head(20)
)

for rank, (game, reviews) in enumerate(top_games.items(), start=1):
    print(f"{rank:2d}. {game}")
    print(f"    Reviews in Dataset : {reviews:,}")
    print()

# ============================================================
# Section 3 — Game Price Statistics
# ============================================================

print("\n" + "=" * 60)
print("SECTION 3 — GAME PRICE STATISTICS")
print("=" * 60)

prices = df["price"] / 100

print(f"Minimum Price : ${prices.min():.2f}")
print(f"Maximum Price : ${prices.max():.2f}")
print(f"Average Price : ${prices.mean():.2f}")

# ============================================================
# Section 4 — Player Playtime Statistics
# ============================================================

print("\n" + "=" * 60)
print("SECTION 4 — PLAYER PLAYTIME STATISTICS")
print("=" * 60)

avg_minutes = round(df["author_playtime_forever"].mean())

hours = avg_minutes // 60
minutes = avg_minutes % 60

print(f"Average Playtime : {hours} hours {minutes} minutes")

# ============================================================
# Section 5 — Review Length Statistics
# ============================================================

print("\n" + "=" * 60)
print("SECTION 5 — REVIEW LENGTH STATISTICS")
print("=" * 60)

avg_words = round(df["word_count"].mean())

print(f"Average Review Length : {avg_words} words")

# ============================================================
# Section 6 — Game Statistics Summary
# ============================================================

print("\n" + "=" * 60)
print("SECTION 6 — GAME STATISTICS SUMMARY")
print("=" * 60)

game_stats = (
    df.groupby("name")
      .agg(
          Total_Reviews=("review", "count"),
          Positive_Reviews=("voted_up", "sum"),
          Average_Playtime_Minutes=("author_playtime_forever", "mean"),
          Average_Price=("price", "mean"),
          Average_Review_Length=("word_count", "mean")
      )
)

game_stats["Negative_Reviews"] = (
    game_stats["Total_Reviews"] -
    game_stats["Positive_Reviews"]
)

game_stats["Positive_Percentage"] = (
    game_stats["Positive_Reviews"] /
    game_stats["Total_Reviews"] * 100
)

# Format values
game_stats["Average_Price"] = (game_stats["Average_Price"] / 100).round(2)
game_stats["Average_Playtime_Minutes"] = (
    game_stats["Average_Playtime_Minutes"]
    .round()
    .astype(int)
)
game_stats["Average_Review_Length"] = (
    game_stats["Average_Review_Length"]
    .round()
    .astype(int)
)
game_stats["Positive_Percentage"] = (
    game_stats["Positive_Percentage"]
    .round(2)
)

print("\nFirst 5 Games Summary:\n")
print(game_stats.head().to_string())

# Save summary dataset
game_stats.to_csv("data/game_statistics.csv")

print("\nGame statistics saved to:")
print("data/game_statistics.csv")

# ============================================================
# Dataset Exploration Complete
# ============================================================

print("\n" + "=" * 60)
print("DATASET EXPLORATION COMPLETE")
print("=" * 60)
