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
    page_title="Tıbbi Tarayıcı",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def download_chroma_from_drive():
    """Google Drive'dan ChromaDB indir ve aç"""
    if not os.path.exists('./chroma_db'):
        file_id = os.getenv('GOOGLE_DRIVE_FILE_ID')
        if not file_id:
            st.error("GOOGLE_DRIVE_FILE_ID bulunamadı!")
            return False
        
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={file_id}"
            output = "temp_chroma.zip"
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("ChromaDB indiriliyor...")
            progress_bar.progress(25)
            
            gdown.download(url, output, quiet=False)
            
            status_text.text("Zip açılıyor...")
            progress_bar.progress(50)
            
            os.makedirs('./chroma_db', exist_ok=True)
            
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall('./chroma_db/')
            
            os.remove(output)
            
            status_text.text("ChromaDB hazır!")
            progress_bar.progress(100)
            st.empty()
            
            return True
            
        except Exception as e:
            st.error(f"ChromaDB indirilemedi: {e}")
            return False
    
    return True

@st.cache_resource
def load_everything():
    """Tüm sistemi bir kez yükle - cache'li"""
    embedding_model = load_embedding_model()

    if check_vector_store_exists():
        vector_store = get_vector_store(embedding_model)
    else:
        st.error("Vector store bulunamadı!")
        return None, None
    
    if not setup_gemini():
        st.error("Gemini API kurulamadı!")
        return None, None
    
    return embedding_model, vector_store

def main():
    """Ana uygulama"""
    st.title("Tıbbi Tarayıcı")
    st.markdown("Türk hastanelerinin tıbbi makalelerinden yararlanarak sorularınızı yanıtlar.")
    
    with st.sidebar:
        st.header("Bilgi")
        st.info("""
        Bu chatbot Türk hastanelerinin tıbbi makalelerini analiz ederek 
        sorularınızı yanıtlar. Sadece bilgilendirme amaçlıdır.
        """)
        
        if st.button("Sistemi Yenile"):
            st.cache_resource.clear()
            st.rerun()
    
    with st.spinner("Sistem başlatılıyor..."):
        embedding_model, vector_store = load_everything()
        
        if vector_store is None:
            st.error("Sistem yüklenemedi. Lütfen sayfayı yenileyin.")
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

if not os.path.exists('./chroma_db'):
    if not download_chroma_from_drive():
        st.error("ChromaDB yüklenemedi!")
        st.stop()

if __name__ == "__main__":
    main()