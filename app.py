
import streamlit as st
import logging
from data_processing import hf_load_dataset, create_documents, explore_dataset
from embedding_model import load_embedding_model, test_embedding_model
from vector_store import create_vector_store, get_vector_store, check_vector_store_exists, get_collection_info
from rag_pipeline import rag_query, test_rag_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    st.set_page_config(page_title="RAG Tıbbi Chatbot", page_icon="🏥")
    
    st.title("RAG Tıbbi Chatbot")
    st.caption("Tıbbi sorularınızı sorun. (Veri Seti: `umutertugrul/turkish-hospital-medical-articles`)")

    if "system_ready" not in st.session_state:
        st.session_state.system_ready = False
        st.session_state.vector_store = None
        st.session_state.embedding_model = None

    if not st.session_state.system_ready:
        with st.spinner("Sistem hazırlanıyor..."):
            try:
                st.info("Veri seti yükleniyor...")
                dataset = hf_load_dataset()
                
                st.info("Document'lar oluşturuluyor...")
                documents = create_documents(dataset)
                
                if check_vector_store_exists():
                    st.success("Vector Store Hazır")
                    st.session_state.embedding_model = load_embedding_model()
                    st.session_state.vector_store = get_vector_store(st.session_state.embedding_model)
                    info = get_collection_info(st.session_state.vector_store)
                    st.info(f"Document Sayısı: {info.get('document_count', 'Bilinmiyor')}")
                else:
                    st.warning("Vector Store Oluşturuluyor...")
                    st.session_state.embedding_model = load_embedding_model()
                    st.session_state.vector_store = create_vector_store(documents, st.session_state.embedding_model)
                    st.success("Vector Store Oluşturuldu!")
                
                st.session_state.system_ready = True
                st.rerun()
                
            except Exception as e:
                st.error(f"Sistem yükleme hatası: {e}")
                st.stop()
    else:
        st.success("Sistem Hazır!")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []


        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Tıbbi sorunuzu yazın..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Tıbbi bilgiler taranıyor ve yanıt oluşturuluyor..."):
                try:
                    answer = rag_query(st.session_state.vector_store, prompt, k=5)
                    response = answer if answer else "Üzgünüm, bu konuda yeterli bilgi bulamadım."
                    
                except Exception as e:
                    response = f"Sorgu işlenirken bir hata oluştu: {e}"
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

        st.markdown("---")
        st.markdown("### Örnek Sorular")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Migren ağrısı nasıl geçer?"):
                st.session_state.messages.append({"role": "user", "content": "Migren ağrısı nasıl geçer?"})
                st.rerun()
        
        with col2:
            if st.button("Grip aşısı ne zaman yaptırmalıyım?"):
                st.session_state.messages.append({"role": "user", "content": "Grip aşısı ne zaman yaptırmalıyım?"})
                st.rerun()
        
        with col3:
            if st.button("Kalp çarpıntısı normal mi?"):
                st.session_state.messages.append({"role": "user", "content": "Kalp çarpıntısı normal mi?"})
                st.rerun()
        
        if st.button("Sohbeti Temizle"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()