from typing import List
import logging
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_embedding_model():
    """
    Google text-embedding-004 modelini yükler (payload limiti ile)
    """
    try:
        logger.info("Google text-embedding-004 modeli yükleniyor...")
        
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY bulunamadı!")
        
        genai.configure(api_key=api_key)
        
        def split_text(text, max_length=30000):
            """Metni parçalara böl"""
            if len(text.encode('utf-8')) <= max_length:
                return [text]
            sentences = text.split('. ')
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len((current_chunk + sentence).encode('utf-8')) <= max_length:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
        
        class GoogleEmbeddings:
            def embed_documents(self, texts):
                try:
                    embeddings = []
                    for text in texts:
                        text_chunks = split_text(text)
                        chunk_embeddings = []
                        
                        for chunk in text_chunks:
                            result = genai.embed_content(
                                model='text-embedding-004',
                                content=chunk
                            )
                            chunk_embeddings.append(result['embedding'])
                    
                        import numpy as np
                        avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()
                        embeddings.append(avg_embedding)
                    
                    return embeddings
                except Exception as e:
                    logger.error(f"Document embedding hatası: {e}")
                    raise
            
            def embed_query(self, text):
                try:
                    text_chunks = split_text(text)
                    chunk_embeddings = []
                    
                    for chunk in text_chunks:
                        result = genai.embed_content(
                            model='text-embedding-004',
                            content=chunk
                        )
                        chunk_embeddings.append(result['embedding'])
                    import numpy as np
                    avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()
                    return avg_embedding
                    
                except Exception as e:
                    logger.error(f"Query embedding hatası: {e}")
                    raise
        
        embeddings = GoogleEmbeddings()
        logger.info("Google embedding modeli başarıyla yüklendi.")
        return embeddings
        
    except Exception as e:
        logger.error(f"Google embedding modeli yüklenirken hata oluştu: {e}")
        raise