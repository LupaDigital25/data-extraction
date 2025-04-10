# %%
import pickle
from bs4 import BeautifulSoup
import requests
from pybloom_live import BloomFilter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from deep_translator import GoogleTranslator # deep-translator
import spacy # python -m spacy download pt_core_news_sm
from datetime import datetime
import random


# tfidf vectorizer and ML model for news classification task
with open("assets/newsClassVect.pkl", "rb") as vec_file:
    tfidf_vectorizer = pickle.load(vec_file)

with open("assets/newsClassML.pkl", "rb") as model_file:
    tfidf_ML = pickle.load(model_file)

# define the path to the news urls file
urls_file = "data/urls.csv"

# define the path to the checkpoint file
checkpoint_file = "data/newsCheckpoint.json"

# create a bloom filter to store the extracted texts
bloom = BloomFilter(capacity=1000000, error_rate=0.001)

# load the portuguese spacy model
nlp = spacy.load("pt_core_news_sm")

# %%
def fetch_url(url):
    """
    Fetches the content of a given URL and returns the visible text.

    This function sends a GET request to the specified URL, parses the HTML content,
    removes script and style elements, and extracts the visible text.

    Args:
        url (str): The URL to fetch content from.

    Returns:
        str: The visible text extracted from the HTML content of the URL.
             If an error occurs, returns a string describing the error.
    """
    try:
        response = requests.get(url, timeout=60)  # Timeout set to 10 seconds
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for tag in soup(['script', 'style']):
            tag.decompose()

        visible_text = soup.get_text(separator=' ')
        lines = [line.strip() for line in visible_text.splitlines()]
        visible_text = ' '.join(line for line in lines if line)
        return visible_text
    except Exception as e:
        return f"Error: {str(e)}"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def compute_news_probability(text, tfidf_vectorizer=tfidf_vectorizer, tfidf_ML=tfidf_ML):
    """
    Compute the probability of the given text being classified as news.

    Args:
        text (str): The input text to be classified.
        tfidf_vectorizer (TfidfVectorizer): The TF-IDF vectorizer used to transform the text.
        tfidf_ML (sklearn.base.BaseEstimator): The machine learning model used to predict the probability.

    Returns:
        float: The probability of the text being classified as news.
    """
    feature = tfidf_vectorizer.transform([text])
    probability = tfidf_ML.predict_proba(feature)[0,1]
    return probability

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def process_keywords(text):
    """
    Process the given text to extract named entities and significant words.
    Args:
        text (str): The input text to process.
    Returns:
        dict: A dictionary containing organized named entities and significant words/verbs.
              The values are number of mentions.
    """
    doc = nlp(text)

    tempos = {"janeiro", "fevereiro", "março", "abril", "maio", "junho",
              "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
              "segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
              "sexta-feira", "sábado", "domingo", "fim-de-semana"}
    lixo = {"REFER", "Pedro Cunha Especial Saramago", "Angra do Heroísmo Aveiro Beja",
            "Braga Bragança Castelo Branco Coimbra"}
    noticias = {"comentários", "notícias", "artigo", "jornal", "assine", "facebook", "assinaturas",
                "online", "twitter", "instagram", "linkedin", "whatsapp", "telegram", "youtube",
                "início", "partilhar", "contactos", "newsletter", "enviar", "termos", "privacidade",
                "iniciar", "inicie", "sessão", "subscrever", "subscreva", "subscrever", "tv", "rádio",
                "conteúdos", "cookies", "opinião", "assinantes", "publicação", "edição", "edições",
                "informações", "patrocínio", "jornalismo", "entrevista", "lusa", "antena",
                "partilhe", "expresso", "público", "observador", "sábado", "paypal", "limites",
                "login", "vídeo", "entre", "programação", "newsletters", "informação", "site",
                "partilhas", "notícia", "clique", "comentário", "conteúdo", "email", "password",
                "jn", "cm", "dn", "rtp", "sic", "tvi", "limites", "índice", "imprimir",
                "perguntas frequentes", "utilizador", "sair", "reportagem", "ongoing",
                "localidade", "cinecartaz", "meteo", "perfil", "siga-nos", "dados pessoais",
                " alterar", "desporto ", "pesquisar "}

    
    # Extract named entities and filter keywords
    named_entities = [
        (ent.text, ent.label_) for ent in doc.ents
        if ent.label_ in {"PER", "ORG", "LOC", "MISC"}
        and ent.text.lower() not in tempos
        and ent.text not in lixo
        and all(w not in ent.text.lower() for w in noticias)
        and all(char.isalnum() or char in {" ", "-"} for char in ent.text)
        and len(ent.text) <= 50
    ]
    palavras_significativas = [
        token.lemma_ for token in doc 
        if token.pos_ in {"NOUN", "VERB"} 
        and not token.is_stop
        and (token.text.islower() or token.text.isupper())
        and token.text.lower() not in tempos 
        and token.text not in lixo 
        and token.text.lower() not in noticias
        and token.text.isalnum()
        and len(token.text) > 3
        and random.random() < 0.44 # instead of .5 to give some penalty to "normal" words
    ]

    if palavras_significativas == []:
        return None
    
    keywords = {}
    for entity, category in named_entities:
        if entity not in keywords:
            keywords[entity] = 1
        else:
            keywords[entity] += 1
    for palavra in palavras_significativas:
        if palavra not in keywords:
            keywords[palavra] = 2
        else:
            keywords[palavra] += 2
    
    return keywords

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def process_sentiment(text):
    """
    Processes the sentiment of a given text by translating it from Portuguese to English
    and then analyzing the sentiment using VADER and TextBlob.

    Args:
        text (str): The input text in Portuguese.

    Returns:
        float: A weighted sentiment score combining VADER and TextBlob results.
    """
    texto_pt = [text[i:i + 2500] for i in range(0, len(text), 2500)][:5]

    en_text = []
    for chunk in texto_pt:
        traducao = GoogleTranslator(source='pt', target='en').translate(chunk)
        en_text.append(traducao)
    texto_en = "".join([text for text in en_text if text is not None])

    # Initialize VADER and TextBlob
    analyzer = SentimentIntensityAnalyzer()
    blob = TextBlob(texto_en)

    # Get VADER and TextBlob sentiment scores
    vader_scores = analyzer.polarity_scores(texto_en)
    textblob_sentiment = blob.sentiment

    return vader_scores["compound"] * 0.6 + textblob_sentiment.polarity * 0.4

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def process_url(url,
                hash_filter=bloom,
                compute_news_probability=compute_news_probability,
                compute_news_keywords=process_keywords,
                compute_news_sentiment=process_sentiment):
    
    # Fetch the text from the URL
    text = fetch_url(url)

    # Check if an error occurred
    if text.startswith("Error: "):
        return {
            "probability": None,
            "keywords": None,
            "sentiment": None,
            "status": "error",
            "error": text,
        }
    
    # Check if the text is a duplicate
    if text in hash_filter:
        return {
            "probability": None,
            "keywords": None,
            "sentiment": None,
            "status": "duplicate",
            "error": "Duplicate text",
        }
    hash_filter.add(text)

    # Check if the text is news
    probability = compute_news_probability(text)
    if probability < 0.5:
        return {
            "probability": None,
            "keywords": None,
            "sentiment": None,
            "status": "notnews",
            "error": f"{probability:.2f} < 0.5",
        }
    
    # Extract keywords and verify if any were found
    keywords = compute_news_keywords(text)
    if keywords is None:
        return {
            "probability": None,
            "keywords": None,
            "sentiment": None,
            "status": "notnews",
            "error": "No keywords found",
        }
    
    # Extract sentiment and return the results
    sentiment = compute_news_sentiment(text)
    return {
        "probability": float(probability),
        "keywords": keywords,
        "sentiment": float(sentiment),
        "status": "success",
        "error": None,
    }

# %%
import os
import json
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, StringType, MapType, IntegerType
from pyspark.sql.functions import udf, col, substring
from pyspark.sql.functions import monotonically_increasing_id

# Create a Spark session
spark = SparkSession.builder \
    .appName("URLProcessing") \
    .config("spark.executor.extraJavaOptions", "-XX:ReservedCodeCacheSize=512m") \
    .config("spark.driver.extraJavaOptions", "-XX:ReservedCodeCacheSize=512m") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "500") \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.dynamicAllocation.minExecutors", "1") \
    .config("spark.dynamicAllocation.maxExecutors", "10") \
    .getOrCreate()



df = spark.read.csv(urls_file, header=True, inferSchema=True)

# Preprocess the DataFrame
df = df.drop("url")
df = df.withColumn("timestamp", substring(col("timestamp"), 1, 6).cast("int"))

result_schema = StructType([
    StructField("probability", FloatType(), True),
    StructField("keywords", MapType(StringType(), IntegerType()), True),
    StructField("sentiment", FloatType(), True),
    StructField("status", StringType(), True),
    StructField("error", StringType(), True)
])

# Wrap your function in a UDF
process_url_udf = udf(process_url, result_schema)

def get_last_processed_status():
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return json.load(f)["last_processed_status"]  # Return the last processed checkpoint
    return None

def update_checkpoint(status):
    # Save the checkpoint status to a JSON file
    with open(checkpoint_file, "w") as f:
        json.dump({"last_processed_status": status}, f)

while_loop = True
while while_loop:
    i = get_last_processed_status() or -1
    df_selected = df.filter((df["id"] > i) & (df["id"] <= i + 5000)) #2400 10000 500 2500
    while_loop = df_selected.count() > 0

    # Divide the DataFrame into chunks
    df_selected = df_selected.repartition(4) #4 22 4 4

    # Apply the UDF to the DataFrame
    df_selected = df_selected.withColumn("result", process_url_udf(col("archive")))

    # Unpack the result struct into separate columns
    df_selected = df_selected.withColumn("probability", df_selected["result"]["probability"]) \
        .withColumn("keywords", df_selected["result"]["keywords"]) \
        .withColumn("sentiment", df_selected["result"]["sentiment"]) \
        .withColumn("status", df_selected["result"]["status"]) \
        .withColumn("error", df_selected["result"]["error"])
    df_selected = df_selected.drop("result")

    # Save the partition results
    df_selected.write.partitionBy("status").mode("append").json("data/news")

    # Update the checkpoint with the last processed status
    last_processed_status = df_selected.agg({"id": "max"}).collect()[0][0]
    update_checkpoint(last_processed_status)
    print(f"Processed partition up to ID: {last_processed_status}. {datetime.now().strftime('%H:%M:%S')}")
    
print("Processing completed.")
spark.stop()
print("Spark session stopped and everything is done :)")