# Data Extraction

This repository is responsible for extracting and aggregating all raw data used in the main project.

![Pipeline Schema](assets/schema.svg)

---

- `newsurls.ipynb`

Responsible for interacting with the Arquivo.pt CDX Server API to extract archived links related to various news sources, and store them in `data/urls.csv` for further processing.

It extracted around 3M URLs.

- `dataExtraction.py`

Handles the processing of each stored URL using the following tools and techniques:

PySpark for parallel processing and scalability;

BeautifulSoup (bs4) for parsing the content of each URL;

Bloom Filter to check for and avoid duplicates;

Machine Learning Models to determine the relevance of the news content.

The script stores the extracted news articles (at `data/news`) along with metadata such as: timestamp, source, sentiment analysis, and mentioned topics.