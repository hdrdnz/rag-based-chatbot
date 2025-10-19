from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Dict, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_vector_store(documents: List[Document], embedding_model, collection_name: str = "medical_qa"):
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

def get_vector_store(embedding_model, collection_name: str = "medical_qa"):
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

def check_vector_store_exists(collection_name: str = "medical_qa"):
    """
    Vector store'un mevcut olup olmadığını kontrol eder
    
    Args:
        collection_name (str): Collection adı
        
    Returns:
        bool: Vector store mevcut mu?
    """
    try:
        base_path = "./chroma_db"
        collection_path = os.path.join(base_path, "medical_qa")

        if not os.path.exists(base_path):
            logger.info(f"Vector store mevcut değil: {base_path}")
            return False
        required_files = ["chroma.sqlite3", "index.bin"]
        for file in required_files:
            file_path = os.path.join(collection_path, file)
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

def get_collection_info(vector_store):
    """
    Collection hakkında bilgi döndürür
    
    Args:
        vector_store: ChromaDB vector store
        
    Returns:
        Dict: Collection bilgileri
    """
    try:
        collection = vector_store._collection
        count = collection.count()
        
        info = {
            "collection_name": collection.name,
            "document_count": count,
            "persist_directory": "./chroma_db"
        }
        
        return info
    except Exception as e:
        logger.error(f"Collection bilgisi alınırken hata oluştu: {e}")
        return {"error": str(e)}

def get_vector_store_size(collection_name: str = "medical_qa"):
    """
    Vector store dosya boyutunu döndürür
    
    Args:
        collection_name (str): Collection adı
        
    Returns:
        str: Dosya boyutu (MB cinsinden)
    """
    persist_directory = "./chroma_db"
    try:
        collection_path = os.path.join(persist_directory, collection_name)
        
        if not os.path.exists(collection_path):
            return "0 MB"
        
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(collection_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        size_mb = total_size / (1024 * 1024)
        return f"{size_mb:.2f} MB"
        
    except Exception as e:
        logger.error(f"Dosya boyutu hesaplanırken hata oluştu: {e}")
        return "Bilinmiyor"

def backup_vector_store(backup_directory: str = "./chroma_db_backup"):
    """
    Vector store'u yedekler
    
    Args:
        backup_directory (str): Yedek dizin
    """
    try:
        import shutil
        source_directory = "./chroma_db"
        
        if os.path.exists(source_directory):
            shutil.copytree(source_directory, backup_directory)
            logger.info(f"Vector store yedeklendi: {backup_directory}")
        else:
            logger.warning(f"Kaynak dizin mevcut değil: {source_directory}")
            
    except Exception as e:
        logger.error(f"Yedekleme sırasında hata oluştu: {e}")