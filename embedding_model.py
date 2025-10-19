from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_embedding_model():
    """
    Sentence-Transformers embedding modelini yükler
        
    Returns:
        SentenceTransformersDocumentEmbedder: LangChain embedding modeli
    """
    try :
        logger.info("Embedding modeli yükleniyor...")
        model_name="trmteb/turkish-embedding-model"
       
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        logger.info("Embedding modeli başarıyla yüklendi.")
        return embeddings
    except Exception as e:
        logger.error(f"Embedding modeli yüklenirken hata oluştu: {e}")
        raise
        
       
def test_embedding_model(embeddings, test_texts: List[str]):
    """
    Embedding modelini test eder
    
    Args:
        embeddings: LangChain embedding modeli
        test_texts (List[str]): Test edilecek metinler
    """
    logger.info("Embedding modeli test ediliyor...")
    
    try:
        vectors = embeddings.embed_documents(test_texts)
        
        print(f"{len(test_texts)} metin başarıyla vektörlere dönüştürüldü")
        print(f"Vektör boyutu: {len(vectors[0])}")
        print(f"İlk vektörün ilk 5 değeri: {vectors[0][:5]}")
        
        return vectors
        
    except Exception as e:
        logger.error(f"Embedding test edilirken hata oluştu: {e}")
        raise


def get_embedding_dimension(embeddings):
    """
    Embedding boyutunu döndürür
    
    Args:
        embeddings: LangChain embedding modeli
        
    Returns:
        int: Embedding boyutu
    """
    try:
        test_vector = embeddings.embed_query("test")
        return len(test_vector)
    except Exception as e:
        logger.error(f"Embedding boyutu alınırken hata oluştu: {e}")
        return None
