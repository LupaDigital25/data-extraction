# Data Extraction

This repository is responsible for extracting and aggregating all raw data used in the main project. It focuses on collecting, processing, and filtering news articles from archived web sources to support downstream machine learning tasks.

![Data Extraction Overview](assets/schema.svg)

---
## Project Structure

The repository is organized as follows:

- `newsUrls.ipynb`:
  - Interacts with the Arquivo.pt CDX Server API to extract archived links related to various news sources.
  - Stores the extracted URLs in `data/urls.csv` for further processing.
  - Approximately 3 million URLs were extracted.

- `dataExtraction.py`:
  - Processes each stored URL using the following tools and techniques:
    - **PySpark** for parallel processing and scalability.
    - **BeautifulSoup (bs4)** for parsing the content of each URL.
    - **Bloom Filter** to check for and avoid duplicates.
    - **Machine Learning Models** to determine the relevance of the news content.
  - Stores the extracted news articles in the `data/news` directory.

- `variables.py`:
  - Includes links to news sources used to feed the API.


- `data/`:
  - Directory containing the extracted URLs (`urls.csv`) and the processed news articles.

- `assets/`:
  - Contains images and other assets related to the project.
  - Also includes machine learning models used for classifying news content.