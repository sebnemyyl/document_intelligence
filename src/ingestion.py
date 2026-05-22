# run_ingestion.py
from data_fetch import fetch_wikipedia_articles, chunk_documents, embed_and_store

topics = [
    "Apple Inc.", "Microsoft", "Amazon (company)",
    "Alphabet Inc.", "Meta Platforms", "Tesla, Inc.",
    "NVIDIA", "JPMorgan Chase", "Goldman Sachs", "S&P 500",
]

docs   = fetch_wikipedia_articles(topics)
chunks = chunk_documents(docs, chunk_size=512, chunk_overlap=50)
db     = embed_and_store(chunks)