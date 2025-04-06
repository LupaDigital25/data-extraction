# data-extraction

- newsurls.ipynb to extract inumerous news related links (around 3M) by using the variables.py links as base

- extracts the news related links by using the cdx api from arquivo.pt

- saves all the links to data/urls.csv for further processing

- dataExtraction.py to extract the news from the links in data/urls.csv

    - uses bloom filter to avoid duplicates

    - uses multiprocessing (spark) to speed up the process

    - use bs4 to parse news contents

    - uses nlp to extract entities, sentiment, and other data

    - processes the news in chuncks using checkpoints

    - saves the data into folders according to the news category in data/news/