from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def search_similar_documents(vector_store, query: str, k: int = 5):
    """
    Daha spesifik arama yapar
    """
    try:
        logger.info("Benzer document'lar aranıyor...")
        
        enhanced_queries = [
            query, 
            f"{query} belirtileri tedavisi", 
            f"{query} nedenleri önlemleri", 
        ]
        
        all_results = []
        for enhanced_query in enhanced_queries:
            results = vector_store.similarity_search(enhanced_query, k=k)
            all_results.extend(results)
        
        unique_results = []
        seen_content = set()
        for doc in all_results:
            if doc.page_content not in seen_content:
                unique_results.append(doc)
                seen_content.add(doc.page_content)
        
        final_results = unique_results[:k]
        
        logger.info(f"Toplam {len(final_results)} benzersiz document bulundu")
        return final_results
        
    except Exception as e:
        logger.error(f"Document arama hatası: {e}")
        return []


def create_context(similar_docs: List[Document], max_length: int = 2000) -> str:
    """
    Benzer document'lardan context oluşturur
    """
    try:
        logger.info("Context oluşturuluyor...")
        
        if not similar_docs:
            logger.warning("Hiç document bulunamadı")
            return ""
        
        context_parts = []
        
        for i, doc in enumerate(similar_docs):
            content = doc.page_content
            title = doc.metadata.get('title', f'Document {i+1}')
            
            formatted = f"Kaynak {i+1}:\nBaşlık: {title}\n\nİçerik: {content}\n"
            context_parts.append(formatted)
        
        context = "\n".join(context_parts)
        
        if len(context) > max_length:
            context = context[:max_length] + "..."
        
        logger.info(f"Context oluşturuldu (uzunluk: {len(context)} karakter)")
        return context
        
    except Exception as e:
        logger.error(f"Context oluşturma hatası: {e}")
        try:
            simple_context = "\n".join([doc.page_content for doc in similar_docs])
            logger.info(f"Basit context döndürüldü (uzunluk: {len(simple_context)} karakter)")
            return simple_context
        except:
            return ""


def create_prompt(context: str, query: str) -> str:
    """
    LLM için prompt oluşturur
    
    Args:
        context (str): Oluşturulan context
        query (str): Kullanıcı sorusu
        
    Returns:
        str: Hazırlanmış prompt
    """
    prompt = f"""Sen bir tıbbi asistan AI'sın. Aşağıdaki tıbbi bilgileri kullanarak kullanıcının sorusunu yanıtla.

TIBBİ BİLGİLER:
{context}

KULLANICI SORUSU: {query}

YANIT KURALLARI:
1. Sadece context'teki bilgileri kullan
2. Profesyonel tıbbi dil kullan
3. "Kullanıcı" ifadeleri kullanma
4. Belirsiz referanslar yapma
5. Kısa ve öz cevap ver
7. Yanıtını Türkçe ver
8.Hastalık dışında sorular gelirse "Ben sadece tıbbi sorulara cevap verebilirim" diye cevap ver.
9. Eğer yeterli bilgi yoksa "Bu konuda yeterli bilgi bulamadım" cevabını ver.

YANIT:"""

    return prompt

def generate_answer_gemini(prompt: str):
    """
    GENERATION: Gemini ile cevap üretir
    
    Args:
        prompt (str): Hazırlanmış prompt
        
    Returns:
        str: Üretilen cevap
    """
    try:
        from gemini_integration import generate_answer_with_gemini
        
        logger.info("Gemini ile cevap üretiliyor...")
        answer = generate_answer_with_gemini(prompt)
        
        logger.info("Cevap üretildi")
        return answer
    except Exception as e:
        logger.error(f"Cevap üretilirken hata oluştu: {e}")
        return "Üzgünüm, bir hata oluştu. Lütfen daha sonra tekrar deneyin."

def rag_query(vector_store, query: str, k: int = 5):
    """
    RAG Pipeline'ın ana fonksiyonu
    
    Args:
        vector_store: ChromaDB vector store
        query (str): Kullanıcı sorusu
        k (int): Arama sayısı
        
    Returns:
        str: RAG cevabı
    """
    try:
        logger.info("RAG Pipeline başlatılıyor...")

        print("Benzer document'lar aranıyor...")
        similar_docs = search_similar_documents(vector_store, query, k=k)

        if not similar_docs:
            return "Üzgünüm, bu konuda yeterli bilgi bulamadım."

        print("Context oluşturuluyor...")
        context = create_context(similar_docs)

        if not context:
            return "Üzgünüm, bu konuda yeterli bilgi bulamadım."

        print("Cevap üretiliyor...")
        prompt = create_prompt(context, query)
        answer = generate_answer_gemini(prompt)
        
        logger.info("RAG Pipeline başarıyla tamamlandı.")
        return answer
    except Exception as e:
        logger.error(f"RAG Pipeline sırasında hata oluştu: {e}")
        return "Üzgünüm, bir hata oluştu. Lütfen daha sonra tekrar deneyin."