from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="jinaai/jina-embeddings-v5-text-nano",
    local_dir="./jina-embeddings-v5-text-nano"
)