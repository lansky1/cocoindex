"""
Text embedding example using Milvus vector database.

This example creates a text embedding pipeline that:
1. Reads markdown files from local directory
2. Chunks them recursively
3. Embeds chunks using SentenceTransformer
4. Stores embeddings in Milvus vector database
5. Provides query handler for semantic search
"""

import os
import datetime
import cocoindex
import cocoindex.targets.milvus as coco_milvus
from cocoindex.auth_registry import add_auth_entry

MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", "19530"))
MILVUS_DB_NAME = os.environ.get("MILVUS_DB_NAME", "default")
MILVUS_COLLECTION = os.environ.get("MILVUS_COLLECTION", "text_embeddings")
MILVUS_API_KEY = os.environ.get("MILVUS_API_KEY", None)


@cocoindex.transform_flow()
def text_to_embedding(
    text: cocoindex.DataSlice[str],
) -> cocoindex.DataSlice[list[float]]:
    """
    Embed the text using a SentenceTransformer model.
    
    This is a shared logic between indexing and querying, so extract it as a function
    to ensure consistency between flow and query paths.
    """
    return text.transform(
        cocoindex.functions.SentenceTransformerEmbed(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    )


@cocoindex.flow_def(name="TextEmbeddingWithMilvus")
def text_embedding_flow(
    flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope
) -> None:
    """
    Define a text embedding flow using Milvus as the vector database.
    
    The flow:
    1. Loads documents from local markdown files
    2. Chunks documents recursively with overlap
    3. Embeds each chunk using SentenceTransformer
    4. Exports embeddings to Milvus with metadata
    """
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(path="markdown_files"),
        refresh_interval=datetime.timedelta(seconds=5),
    )

    doc_embeddings = data_scope.add_collector()

    with data_scope["documents"].row() as doc:
        doc["chunks"] = doc["content"].transform(
            cocoindex.functions.SplitRecursively(),
            language="markdown",
            chunk_size=500,
            chunk_overlap=100,
        )

        with doc["chunks"].row() as chunk:
            chunk["embedding"] = text_to_embedding(chunk["text"])
            
            doc_embeddings.collect(
                id=cocoindex.GeneratedField.UUID,
                filename=doc["filename"],
                location=chunk["location"],
                text=chunk["text"],
                text_embedding=chunk["embedding"],
            )

    milvus_conn = add_auth_entry(
        "MilvusConnection",
        coco_milvus.MilvusConnection(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            db_name=MILVUS_DB_NAME,
            api_key=MILVUS_API_KEY,
        ),
    )

    doc_embeddings.export(
        "doc_embeddings",
        coco_milvus.Milvus(
            collection_name=MILVUS_COLLECTION,
            connection=milvus_conn,
        ),
        primary_key_fields=["id"],
        vector_indexes=[
            cocoindex.VectorIndexDef(
                "text_embedding",
                cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
                method=cocoindex.HnswVectorIndexMethod(m=16),
            )
        ],
    )


@text_embedding_flow.query_handler(
    result_fields=cocoindex.QueryHandlerResultFields(
        embedding=["text_embedding"],
        score="score",
    ),
)
def query_handler(
    query: str,
) -> list[dict]:
    """
    Query handler for semantic search.
    
    Embeds the query using the same model and searches Milvus for similar chunks.
    """
    # TODO: Implement query handler with embedding and search logic
    return []


if __name__ == "__main__":
    flow = text_embedding_flow
    
    print("Text Embedding Flow with Milvus")
    print(f"Target Collection: {MILVUS_COLLECTION}")
    print(f"Milvus Server: {MILVUS_HOST}:{MILVUS_PORT}")
