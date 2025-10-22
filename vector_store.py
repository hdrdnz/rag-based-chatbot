from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Dict, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_vector_store(documents: List[Document], embedding_model, collection_name: str = "medical"):
    """
    ChromaDB vector store oluşturur ve KALICI olarak kaydeder
    
    Args:
        documents (List[Document]): LangChain Document listesi
        embedding_model: Embedding modeli
        collection_name (str): Collection adı
        
    Returns:
        Chroma: Oluşturulan vector store
    """
    try:
        logger.info("Vector store oluşturuluyor")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory="./chroma_db"
        )
        vector_store.persist()
        logger.info("Vector store başarıyla oluşturuldu ve kaydedildi")
        return vector_store
    except Exception as e:
        logger.error(f"Vector store oluşturulurken hata oluştu: {e}")
        raise

def get_vector_store(embedding_model, collection_name: str = "medical"):
    """
    Mevcut vector store'u yükler (KALICI kayıttan)
    
    Args:
        embedding_model: Embedding modeli
        collection_name (str): Collection adı
        
    Returns:
        Chroma: Yüklenen vector store
    """
    try:
        logger.info(f"Mevcut vector store yükleniyor: {collection_name}")

        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory="./chroma_db"
        )

        logger.info("Vector store başarıyla yüklendi (kalıcı kayıttan)")
        return vector_store
        
    except Exception as e:
        logger.error(f"Vector store yüklenirken hata oluştu: {e}")
        raise

def check_vector_store_exists(collection_name: str = "medical"):
    """
    Vector store'un mevcut olup olmadığını kontrol eder
    """
    try:
        base_path = "./chroma_db"
        
        if not os.path.exists(base_path):
            logger.info(f"Vector store mevcut değil: {base_path}")
            return False
        
        required_files = ["chroma.sqlite3"]
        for file in required_files:
            file_path = os.path.join(base_path, file)
            if not os.path.exists(file_path):
                logger.info(f"Gerekli dosya eksik: {file_path}")
                return False
        
        logger.info("Vector store mevcut")
        return True
        
    except Exception as e:
        logger.error(f"Vector store kontrolü sırasında hata oluştu: {e}")
        return False

def search_similar_documents(vector_store, query: str, k: int = 5):
    """
    Benzer document'ları arar
    
    Args:
        vector_store: ChromaDB vector store
        query (str): Arama sorgusu
        k (int): Döndürülecek document sayısı
        
    Returns:
        List[Document]: Benzer document'lar
    """
    try:
        logger.info(f"Benzer document'lar aranıyor: '{query[:50]}...'")
        
        results = vector_store.similarity_search(query=query, k=k)
        
        logger.info(f"{len(results)} adet benzer document bulundu")
        return results
        
    except Exception as e:
        logger.error(f"Document arama sırasında hata oluştu: {e}")
        return []

def get_vector_store_size(collection_name: str = "medical"):
    """
    Vector store dosya boyutunu döndürür
    """
    persist_directory = "./chroma_db"
    try:
        if not os.path.exists(persist_directory):
            return "0 MB"
        
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(persist_directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        size_mb = total_size / (1024 * 1024)
        return f"{size_mb:.2f} MB"
        
    except Exception as e:
        logger.error(f"Dosya boyutu hesaplanırken hata oluştu: {e}")
        return "Bilinmiyor"