"""Download the prebuilt ArxivSage data (papers, chunks, embeddings) from the Hugging Face Hub into data/.

With these files, notebook 01 can be skipped, and notebook 02 rebuilds the Weaviate index in ~2 minutes
because the cached embeddings are reused instead of recomputed.

usage: python download_data.py
"""
from huggingface_hub import hf_hub_download

from common import DATA_DIR, HF_DATASET

FILES = ["papers.jsonl", "chunks.jsonl", "chunk_embeddings.npy"]

for name in FILES:
    path = hf_hub_download(repo_id=HF_DATASET, filename=name, repo_type="dataset", local_dir=DATA_DIR)
    print("downloaded", path)
