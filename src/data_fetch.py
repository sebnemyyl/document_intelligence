import wikipedia
from langchain_core.documents import Document
from pathlib import Path
import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

def fetch_wikipedia_articles(topics: list[str], save_dir: str = "data/raw") -> list[Document]:
    """
    Fetch Wikipedia articles by topic and return as LangChain Documents.
    Each article becomes one Document with metadata preserved.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    docs = []

    for topic in topics:
        try:
            page = wikipedia.page(topic, auto_suggest=False)
            doc = Document(
                page_content=page.content,
                metadata={
                    "source": page.url,
                    "title": page.title,
                    "topic": topic,
                    "type": "wikipedia"
                }
            )
            docs.append(doc)

            # Save raw text locally so you don't re-fetch every run
            safe_name = topic.replace(" ", "_").replace("/", "-")
            with open(f"{save_dir}/{safe_name}.txt", "w", encoding="utf-8") as f:
                f.write(page.content)

            print(f"✓ {page.title} — {len(page.content):,} chars")

        except wikipedia.DisambiguationError as e:
            print(f"⚠ '{topic}' is ambiguous. Options: {e.options[:3]}")
        except wikipedia.PageError:
            print(f"✗ '{topic}' not found")

    print(f"\nFetched {len(docs)} articles total")
    return docs


# Example: build a corpus around a focused theme
topics = [
    "Apple Inc.",
    "Microsoft",
    "Amazon (company)",
    "Alphabet Inc.",
    "Meta Platforms",
    "Tesla, Inc.",
    "NVIDIA",
    "JPMorgan Chase",
    "Goldman Sachs",
    "S&P 500",
]


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import tiktoken

# ── helpers ──────────────────────────────────────────────────────────────────

def count_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    """Count tokens using the same tokenizer as the embedding model."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def clean_wikipedia_text(text: str) -> str:
    """
    Remove Wikipedia boilerplate that adds noise to chunks:
    section headers with only '==' markers, citation markers like [1],
    and excessive blank lines.
    """
    import re
    # Remove citation markers e.g. [1], [23]
    text = re.sub(r'\[\d+\]', '', text)
    # Remove edit section markers e.g. [edit]
    text = re.sub(r'\[edit\]', '', text)
    # Collapse 3+ blank lines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ── main chunking function ────────────────────────────────────────────────────

def chunk_documents(
    docs: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    verbose: bool = True,
) -> list[Document]:
    """
    Clean and chunk a list of Documents using RecursiveCharacterTextSplitter.

    Separator priority (tried in order):
      1. Double newline  → paragraph boundary  (best)
      2. Single newline  → line boundary
      3. Period + space  → sentence boundary
      4. Space           → word boundary       (last resort)

    Each output chunk inherits the parent document's metadata,
    plus its own chunk_index and token_count.

    Args:
        docs:          Output of fetch_wikipedia_articles()
        chunk_size:    Target chunk size in characters
        chunk_overlap: Characters shared between adjacent chunks
        verbose:       Print per-article and summary stats

    Returns:
        List of LangChain Document objects ready for embedding
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks = []

    for doc in docs:
        # 1. Clean boilerplate before splitting
        cleaned_text = clean_wikipedia_text(doc.page_content)

        # 2. Split into raw chunks
        raw_chunks = splitter.create_documents(
            texts=[cleaned_text],
            metadatas=[doc.metadata],   # parent metadata passed to every chunk
        )

        # 3. Enrich each chunk with position and token metadata
        for i, chunk in enumerate(raw_chunks):
            chunk.metadata.update({
                "chunk_index":  i,
                "chunk_total":  len(raw_chunks),
                "char_count":   len(chunk.page_content),
                "token_count":  count_tokens(chunk.page_content),
            })

        all_chunks.extend(raw_chunks)

        if verbose:
            token_counts = [c.metadata["token_count"] for c in raw_chunks]
            print(
                f"  {doc.metadata['title']:<35} "
                f"→ {len(raw_chunks):>3} chunks | "
                f"avg {sum(token_counts)//len(token_counts)} tok | "
                f"max {max(token_counts)} tok"
            )

    if verbose:
        _print_summary(all_chunks)

    return all_chunks


def _print_summary(chunks: list[Document]) -> None:
    """Print aggregate stats across all chunks — useful for your ablation notebook."""
    token_counts = [c.metadata["token_count"] for c in chunks]
    char_counts  = [c.metadata["char_count"]  for c in chunks]
    print(f"""
── Chunking summary ─────────────────────────────
  Total chunks : {len(chunks)}
  Avg tokens   : {sum(token_counts) // len(token_counts)}
  Min tokens   : {min(token_counts)}
  Max tokens   : {max(token_counts)}
  Avg chars    : {sum(char_counts) // len(char_counts)}
─────────────────────────────────────────────────""")




def embed_and_store(chunks: list, persist_dir: str = ".chroma") -> Chroma:
    """Embed chunks and persist to ChromaDB."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="doc_intelligence"
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB at {persist_dir}")
    return vectorstore