import multiprocessing
import socket
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, lower, regexp_replace, split, explode, window, udf, count, when
from pyspark.sql.types import StructType, StringType, TimestampType, ArrayType
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Function to simulate social media posts
def simulate_events():
    HOST = '0.0.0.0'
    PORT = 9999
    messages = [
        {"text": "Love the sunny weather! #sunshine", "timestamp": "2023-05-01T14:00:00Z", "location": "San Francisco"},
        {"text": "It's raining cats and dogs here. #rain", "timestamp": "2023-05-01T14:05:00Z", "location": "Seattle"},
        {"text": "Just had the best coffee ever! #coffee", "timestamp": "2023-05-01T14:10:00Z"},
        {"text": "Can't believe how great that concert was! #music", "timestamp": "2023-05-01T14:15:00Z", "location": "New York"},
        {"text": "Stuck in traffic, ugh! #traffic", "timestamp": "2023-05-01T14:20:00Z"},
    ]
    # Set up server to send simulated posts
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen(1)
    print("Server started listening on port", PORT)

    conn, addr = sock.accept()
    print("Connected by", addr)

    for message in messages:
        data = json.dumps(message) + '\\n'
        conn.sendall(data.encode('utf-8'))
        time.sleep(1)  # Simulate real-time posting delay

    conn.close()

# Function for sentiment analysis using VADER
def sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Function to extract hashtags from text
def extract_hashtags(text):
    return list(set(part[1:] for part in text.split() if part.startswith('#')))

# Function to save plots with Matplotlib
def save_plot(data, filename='sentiment_plot.png'):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['timestamp'], data['Positive'], label='Positive', marker='o', linestyle='-')
    ax.plot(data['timestamp'], data['Negative'], label='Negative', marker='o', linestyle='-')
    ax.plot(data['timestamp'], data['Neutral'], label='Neutral', marker='o', linestyle='-')
    ax.set_title('Sentiment Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True)
    fig.savefig(filename)
    plt.close(fig)  # Close the figure to free up memory

# Function to start Spark streaming and process the data
def start_spark_streaming():
    schema = StructType().add("text", StringType()).add("timestamp", TimestampType()).add("location", StringType())

    spark = SparkSession.builder \
        .appName("SocialMediaStream") \
        .master("local[*]") \
        .config("spark.ui.enabled", "false") \
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem") \
        .getOrCreate()

    sentiment_udf = udf(sentiment_analysis, StringType())
    extract_hashtags_udf = udf(extract_hashtags, ArrayType(StringType()))

    lines = spark \
        .readStream \
        .format("socket") \
        .option("host", "localhost") \
        .option("port", 9999) \
        .load()

    cleaned_lines = lines.select(
        from_json(col("value"), schema).alias("data")
    ).select("data.*")

    cleaned_lines = cleaned_lines.withColumn("text", lower(col("text")))
    cleaned_lines = cleaned_lines.withColumn("text", regexp_replace(col("text"), "[^\\w\\s#]", ""))
    cleaned_lines = cleaned_lines.withColumn("words", split(col("text"), "\\s+"))

    cleaned_lines = cleaned_lines.withColumn("sentiment", sentiment_udf(col("text")))

    hashtags = cleaned_lines.withColumn("hashtag", explode(extract_hashtags_udf(col("text"))))
    hashtag_counts = hashtags.groupBy(
        window(col("timestamp"), "1 minute"),
        col("hashtag")
    ).count().orderBy('window')

    location_sentiment = cleaned_lines.filter(cleaned_lines.location.isNotNull()) \
        .groupBy("location") \
        .agg(
            count(when(col("sentiment") == "Positive", True)).alias("Positive"),
            count(when(col("sentiment") == "Negative", True)).alias("Negative"),
            count(when(col("sentiment") == "Neutral", True)).alias("Neutral")
        )

    sentiment_query = location_sentiment.writeStream \
        .outputMode("complete") \
        .format("console") \
        .option("truncate", "false") \
        .start()

    hashtag_query = hashtag_counts.writeStream \
        .outputMode("complete") \
        .format("console") \
        .option("truncate", "false") \
        .start()

    sentiment_query.awaitTermination()
    hashtag_query.awaitTermination()

if __name__ == "__main__":
    p = multiprocessing.Process(target=simulate_events)
    p.start()
    start_spark_streaming()
    p.join()
