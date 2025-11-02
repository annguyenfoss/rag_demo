"""Debug and test script for RAG pipeline.

Usage: python test_rag.py
"""
import logging
from pathlib import Path
import json

from engine.engine import init_settings, get_query_engine, _load_or_create_index
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_storage():
    """Check if documents are stored."""
    logger.info("=== Checking Storage ===")
    
    storage_path = Path(config.STORAGE_DIR)
    if not storage_path.exists():
        logger.error(f"❌ Storage directory doesn't exist: {storage_path}")
        return False
    
    docstore_path = storage_path / "docstore.json"
    if not docstore_path.exists():
        logger.error(f"❌ Docstore doesn't exist: {docstore_path}")
        return False
    
    with open(docstore_path) as f:
        docstore = json.load(f)
    
    doc_count = len(docstore.get("docstore/data", {}))
    logger.info(f"Documents in docstore: {doc_count}")
    
    if doc_count == 0:
        logger.error("❌ No documents found! Run: python ingest.py")
        return False
    
    logger.info(f"✓ Found {doc_count} document chunks")
    
    # Show sample documents
    logger.info("\nSample documents:")
    for i, (doc_id, doc_data) in enumerate(
        list(docstore.get("docstore/data", {}).items())[:3]
    ):
        metadata = doc_data.get("metadata", {})
        text_preview = doc_data.get("text", "")[:100]
        logger.info(f"\n  Document {i+1}:")
        logger.info(f"    ID: {doc_id}")
        logger.info(f"    File: {metadata.get('file_name', 'unknown')}")
        logger.info(f"    Page: {metadata.get('page', 'N/A')}")
        logger.info(f"    Text preview: {text_preview}...")
    
    return True


def test_retrieval(query: str):
    """Test document retrieval for a query."""
    logger.info(f"\n=== Testing Retrieval ===")
    logger.info(f"Query: {query}")
    
    engine = get_query_engine()
    response = engine.query(query)
    
    # Get the response text
    text = getattr(response, "response", None) or str(response)
    logger.info(f"\nResponse length: {len(text)} chars")
    logger.info(f"Response: {text}")
    
    # Check source nodes (retrieved documents)
    if hasattr(response, "source_nodes"):
        logger.info(
            f"\n✓ Retrieved {len(response.source_nodes)} documents:"
        )
        for i, node in enumerate(response.source_nodes, 1):
            score = node.score if hasattr(node, "score") else "N/A"
            metadata = (
                node.node.metadata
                if hasattr(node.node, "metadata")
                else {}
            )
            text_preview = (
                node.node.text[:150]
                if hasattr(node.node, "text")
                else ""
            )
            
            logger.info(f"\n  Source {i} (score: {score}):")
            logger.info(
                f"    File: {metadata.get('file_name', 'unknown')}"
            )
            logger.info(f"    Page: {metadata.get('page', 'N/A')}")
            logger.info(f"    Text: {text_preview}...")
    else:
        logger.warning("⚠ No source_nodes found in response")
    
    return text


def test_direct_index_query():
    """Test querying the index directly."""
    logger.info("\n=== Testing Direct Index Query ===")
    
    index = _load_or_create_index()
    
    # Try to get some stats
    logger.info(f"Index type: {type(index).__name__}")
    
    # Test a simple retrieval
    retriever = index.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve("seatbelt Ford Transit")
    
    logger.info(f"Retrieved {len(nodes)} nodes")
    for i, node in enumerate(nodes[:3], 1):
        score = node.score if hasattr(node, "score") else "N/A"
        metadata = (
            node.node.metadata
            if hasattr(node.node, "metadata")
            else {}
        )
        logger.info(f"\n  Node {i} (score: {score}):")
        logger.info(f"    File: {metadata.get('file_name', 'unknown')}")
        logger.info(
            f"    Text preview: {node.node.text[:100]}..."
        )


def check_milvus_collection():
    """Check Milvus collection statistics."""
    logger.info("\n=== Checking Milvus Collection ===")
    
    try:
        from pymilvus import MilvusClient
        
        client = MilvusClient(uri=config.MILVUS_URI)
        
        # Check if collection exists
        collections = client.list_collections()
        logger.info(f"Collections: {collections}")
        
        if config.MILVUS_COLLECTION in collections:
            # Get collection stats
            stats = client.get_collection_stats(
                collection_name=config.MILVUS_COLLECTION
            )
            logger.info(f"Collection stats: {stats}")
            
            # Query to get count
            result = client.query(
                collection_name=config.MILVUS_COLLECTION,
                filter="",
                output_fields=["count(*)"],
                limit=1,
            )
            logger.info(f"✓ Milvus collection exists with data")
        else:
            logger.warning(
                f"⚠ Collection '{config.MILVUS_COLLECTION}' not found"
            )
            
    except Exception as e:
        logger.error(f"Error checking Milvus: {e}")


def main():
    """Run all tests."""
    logger.info("=== RAG Pipeline Debugger ===\n")
    
    # Initialize
    init_settings()
    
    # Check storage
    if not check_storage():
        logger.error(
            "\n❌ Storage check failed. Please run ingestion first:"
        )
        logger.error("   python ingest.py")
        return
    
    # Check Milvus
    check_milvus_collection()
    
    # Test direct index query
    test_direct_index_query()
    
    # Test full retrieval
    test_queries = [
        "How to fasten seatbelt on Ford Transit?",
        "What are the specifications of Kia Carnival?",
        "Tell me about Mitsubishi Lancer Evolution X",
    ]
    
    for query in test_queries:
        test_retrieval(query)
        logger.info("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()

