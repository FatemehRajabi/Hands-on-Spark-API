# main.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("MusicAnalysis").getOrCreate()

# Load datasets
logs_df = spark.read.csv("input/listening_logs.csv", header=True, inferSchema=True)
songs_df = spark.read.csv("input/songs_metadata.csv", header=True, inferSchema=True)

# logs_df.show(5)
# songs_df.show(5)
# Task 1: User Favorite Genres
joined_df = logs_df.join(songs_df, on="song_id", how="inner")
genre_counts = joined_df.groupBy("user_id", "genre").count()
window_spec = Window.partitionBy("user_id").orderBy(col("count").desc())

favorite_genres = genre_counts.withColumn("rank", row_number().over(window_spec)) \
    .filter(col("rank") == 1) \
    .select("user_id", "genre", col("count").alias("listen_count"))

favorite_genres.show(10)

# favorite_genres.coalesce(1).write.mode("overwrite").option("header", True).csv(
#     "D:/1.UNCC/1.CloudComputing/Hands-on-Spark-API/Hands-on-Spark-API/outputs/task1_user_favorite_genres"
# )
# Task 2: Average Listen Time
average_listen_time = logs_df.groupBy("user_id") \
    .agg(round(avg("duration_sec"), 2).alias("average_duration_sec")) \
    .orderBy("user_id")

average_listen_time.show(10)


# Task 3: Create your own Genre Loyalty Scores and rank them and list out top 10
# (Listens in favorite genre / Total listens) × 100
total_listens = joined_df.groupBy("user_id") \
    .agg(count("*").alias("total_listens"))

genre_listens = joined_df.groupBy("user_id", "genre") \
    .agg(count("*").alias("genre_listens"))

window_spec = Window.partitionBy("user_id").orderBy(col("genre_listens").desc())

top_genre = genre_listens.withColumn(
    "rank",
    row_number().over(window_spec)
).filter(col("rank") == 1)

loyalty_scores = top_genre.join(total_listens, "user_id") \
    .withColumn(
        "genre_loyalty_score",
        round(col("genre_listens") / col("total_listens") * 100, 2)
    ) \
    .select(
        "user_id",
        "genre",
        "genre_listens",
        "total_listens",
        "genre_loyalty_score"
    ) \
    .orderBy(col("genre_loyalty_score").desc()) \
    .limit(10)

loyalty_scores.show()
# Task 4: Identify users who listen between 12 AM and 5 AM
night_owl_users = logs_df.withColumn("hour", hour(col("timestamp"))) \
    .filter((col("hour") >= 0) & (col("hour") <= 5)) \
    .select("user_id", "song_id", "timestamp", "duration_sec", "hour") \
    .orderBy("user_id")

night_owl_users.show(10)