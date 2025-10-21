import streamlit as st
import logging
import os
import requests
import zipfile
from dotenv import load_dotenv
from data_processing import hf_load_dataset, create_documents
from embedding_model import load_embedding_model
from vector_store import create_vector_store, get_vector_store, check_vector_store_exists
from rag_pipeline import rag_query
from gemini_integration import setup_gemini

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Tıbbi Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def download_chroma_from_drive():
    """Google Drive'dan ChromaDB indir ve aç"""
    if not os.path.exists('./chroma_db'):
        st.info("ChromaDB Google Drive'dan indiriliyor...")
        file_id = os.getenv('GOOGLE_DRIVE_FILE_ID')
        if not file_id:
            st.error("GOOGLE_DRIVE_FILE_ID bulunamadı!")
            return False
        
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={file_id}"
            output = "temp_chroma.zip"
            
            st.info("Google Drive'dan indiriliyor...")
            gdown.download(url, output, quiet=False)
            
            st.info("Zip dosyası indirildi, açılıyor...")
            
            os.makedirs('./chroma_db', exist_ok=True)
            
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall('./chroma_db/')
            
            os.remove(output)
            st.success("ChromaDB başarıyla yüklendi!")
            return True
            
        except Exception as e:
            st.error(f"ChromaDB indirilemedi: {e}")
            return False
    
    return True

@st.cache_resource
def load_embedding_model_cached():
    """Embedding modelini cache'le"""
    return load_embedding_model()

@st.cache_resource
def load_vector_store():
    """Vector store'u yükle"""
    try:
        if check_vector_store_exists():
            logger.info("Mevcut vector store yükleniyor...")
            embedding_model = load_embedding_model_cached()
            return get_vector_store(embedding_model)
        else:
            st.error("Vector store bulunamadı! Lütfen önce vector store'u oluşturun.")
            return None
    except Exception as e:
        logger.error(f"Vector store yüklenirken hata: {e}")
        return None

def main():
    """Ana uygulama"""
    st.title("Türkçe Tıbbi Chatbot")
    st.markdown("Türk hastanelerinin tıbbi makalelerinden yararlanarak sorularınızı yanıtlar.")
    
    if not download_chroma_from_drive():
        st.error("ChromaDB yüklenemedi!")
        return
    
    with st.sidebar:
        st.header("Bilgi")
        st.info("""
        Bu chatbot Türk hastanelerinin tıbbi makalelerini analiz ederek 
        sorularınızı yanıtlar. Sadece bilgilendirme amaçlıdır.
        """)
        
        if st.button("Vector Store'u Yenile"):
            st.cache_resource.clear()
            st.rerun()
    
    with st.spinner("Sistem hazırlanıyor..."):
        vector_store = load_vector_store()
        
        if vector_store is None:
            st.error("Vector store yüklenemedi. Lütfen sayfayı yenileyin.")
            return
        
        if setup_gemini():
            st.success("Sistem hazır!")
        else:
            st.error("Gemini API kurulamadı. Lütfen API key'inizi kontrol edin.")
            return
    
    st.header("Soru Sorun")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Tıbbi sorunuzu yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Düşünüyorum..."):
                try:
                    answer = rag_query(vector_store, prompt)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Hata oluştu: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()