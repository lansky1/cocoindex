# Text Embedding with Milvus

[![GitHub](https://img.shields.io/github/stars/cocoindex-io/cocoindex?color=5B5BD6)](https://github.com/cocoindex-io/cocoindex)

This example demonstrates how to build a text embedding index using CocoIndex with [Milvus](https://milvus.io/) as the vector database backend. The pipeline ingests markdown files, chunks them, generates embeddings using SentenceTransformer, and stores them in Milvus for semantic search.

If this helps, a star at [CocoIndex Github](https://github.com/cocoindex-io/cocoindex) is appreciated.

## What It Does

This example flow:
1. **Loads** markdown documents from local filesystem
2. **Chunks** documents recursively with configurable chunk size and overlap
3. **Embeds** each chunk using a pre-trained SentenceTransformer model
4. **Stores** embeddings and metadata in a Milvus collection
5. **Queries** the index for semantic search using the same embedding model

The example uses:
- **Milvus** - Open-source vector database with HNSW indexing
- **SentenceTransformer** - Pre-trained models for semantic text embeddings
- **CocoIndex** - Data synchronization framework for incremental updates

## Prerequisites

### 1. Milvus Server

You need a running Milvus server. The easiest way is to use Docker:

```bash
# Start Milvus with Docker Compose
docker run -d \
  --name milvus \
  -p 19530:19530 \
  -p 9091:9091 \
  -e COMMON_STORAGETYPE=local \
  milvusdb/milvus:latest
```

Or use docker-compose:

```yaml
version: '3'
services:
  milvus:
    image: milvusdb/milvus:latest
    ports:
      - "19530:19530"
      - "9091:9091"
    environment:
      COMMON_STORAGETYPE: local
    volumes:
      - milvus_data:/var/lib/milvus

volumes:
  milvus_data:
```

Then start with:

```bash
docker-compose up -d
```

### 2. PostgreSQL

CocoIndex requires PostgreSQL for metadata storage. Follow [the installation guide](https://cocoindex.io/docs/getting_started/installation#-install-postgres) if not already set up.

### 3. Python Dependencies

Install dependencies from this directory:

```bash
pip install -e .
```

This installs:
- `cocoindex` - Core framework
- `pymilvus>=2.4.0` - Milvus Python client
- `sentence-transformers>=2.2.0` - Embedding models

## Configuration

Create a `.env` file to customize settings:

```bash
cp .env.example .env
```

Available environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MILVUS_HOST` | `localhost` | Milvus server hostname |
| `MILVUS_PORT` | `19530` | Milvus gRPC port |
| `MILVUS_DB_NAME` | `default` | Milvus database name |
| `MILVUS_COLLECTION` | `text_embeddings` | Collection name for embeddings |
| `MILVUS_API_KEY` | (empty) | Optional API key for authentication |

## Running the Example

### 1. Prepare Sample Data

Create a `markdown_files` directory with sample markdown files:

```bash
mkdir markdown_files
echo "# Sample Document\nThis is a sample document for embedding." > markdown_files/sample.md
```

### 2. Run the Flow

```bash
python main.py
```

The flow will:
1. Load markdown files from `markdown_files/`
2. Process and embed them
3. Store embeddings in Milvus
4. Make the collection available for queries

### 3. Query the Index

Once the flow is running, you can query it semantically:

```python
# Query example
results = flow.query("What is machine learning?")
for result in results:
    print(f"Score: {result['score']}")
    print(f"Text: {result['text']}")
    print(f"File: {result['filename']}")
```

## Architecture

```
┌─────────────────────┐
│  Markdown Files     │
│  (Local Directory)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  CocoIndex Flow     │
│  ├─ SplitRecursive  │
│  ├─ SentenceTransform
│  └─ Collect         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Milvus Vector    │
│    Database         │
│  ├─ Collection      │
│  └─ HNSW Index      │
└─────────────────────┘
```

## Key Features

### Incremental Updates
- CocoIndex tracks changes to source files
- Only modified documents are re-processed
- Efficient metadata tracking in PostgreSQL

### Semantic Search
- Uses same embedding model for index and queries
- Supports various similarity metrics (cosine, L2, inner product)
- Fast vector search with HNSW indexing

### Scalability
- Milvus supports distributed deployments
- Handles millions of vectors efficiently
- Partitioning and sharding support

## Milvus Index Configuration

The example uses HNSW (Hierarchical Navigable Small World) indexing:

```python
cocoindex.VectorIndexDef(
    "text_embedding",
    cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
    method=cocoindex.HnswVectorIndexMethod(m=16),
)
```

Configuration options:
- **m**: Number of bidirectional links (default: 16)
- **ef_construction**: Size of dynamic list (default: 200)
- **query_ef**: (At query time - tuned in Milvus)

## Troubleshooting

### Milvus Connection Error
```
Error: failed to connect to Milvus
```
- Ensure Milvus is running: `docker ps | grep milvus`
- Check host/port configuration in `.env`
- Verify firewall allows access to port 19530

### Out of Memory
```
Error: vector index building failed
```
- Reduce chunk size in flow
- Use IVFFlat indexing for larger datasets
- Increase available memory

### Slow Embedding Generation
- Consider using a smaller SentenceTransformer model
- Use GPU acceleration (requires CUDA)
- Pre-compute embeddings offline

## Advanced Usage

### Custom Embedding Models

Replace the model in `text_to_embedding`:

```python
model="sentence-transformers/all-mpnet-base-v2"  # Larger, more accurate
model="sentence-transformers/paraphrase-MiniLM-L6-v2"  # Faster
```

### Different Index Methods

Use IVF_FLAT for larger datasets:

```python
cocoindex.VectorIndexDef(
    "text_embedding",
    cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
    method=cocoindex.IvfFlatVectorIndexMethod(lists=1024),
)
```

### Multiple Vector Fields

For more complex scenarios, consider LanceDB which supports multiple vectors per collection.

## Performance Tips

1. **Batch Processing**: Process multiple files together
2. **Index Parameters**: Tune m and ef_construction based on dataset size
3. **Connection Pooling**: CocoIndex automatically pools Milvus connections
4. **Chunk Size**: Balance between granularity and vector count (512 tokens typical)

## References

- [Milvus Documentation](https://milvus.io/docs)
- [SentenceTransformer Models](https://www.sbert.net/docs/pretrained_models.html)
- [CocoIndex Documentation](https://cocoindex.io/)
- [HNSW Algorithm](https://arxiv.org/abs/1802.02413)

## License

Apache License 2.0 - See LICENSE file for details
