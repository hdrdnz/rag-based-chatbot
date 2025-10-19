import logging
from data_processing import hf_load_dataset, create_documents, explore_dataset
from embedding_model import load_embedding_model, test_embedding_model
from vector_store import create_vector_store, get_vector_store, check_vector_store_exists, get_collection_info
from rag_pipeline import rag_query, test_rag_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Ana uygulama - Tüm işlemler burada
    """

    try:
        print("Veri seti yükleniyor...")
        dataset =hf_load_dataset()
        explore_dataset(dataset)


        documents =create_documents(dataset)
        print(f"Toplam {len(documents)} adet document oluşturuldu.")

       
        print ("Embedding modeli yükleniyor...")
        embedding_model =load_embedding_model()
        print ("Embedding modeli başarıyla yüklendi.")

        print("Vector store kontrol ediliyor...")

        if check_vector_store_exists():
            print("Vector store mevcut.")
            vector_store=get_vector_store(embedding_model)
        else:
            print("Vector store mevcut değil. Yeni vector store oluşturuluyor...")
            vector_store=create_vector_store(documents,embedding_model)
            print("Vector store başarıyla oluşturuldu.")


        print("RAG pipeline test ediliyor...")
        test_rag_pipeline(vector_store)

        print("\n 5. Etkileşimli chat modu")
        print("Çıkmak için 'quit' yazın")
        print("-" * 40)
        
        while True:
            query = input("\nSorunuz: ").strip()
            
            if query.lower() in ['quit', 'çık', 'exit']:
                print("Görüşürüz!")
                break
            
            if not query:
                continue
            
            print("Düşünüyorum...")
            answer = rag_query(vector_store, query, k=3)
            print(f"Cevap: {answer}")
    
    except Exception as e:
        logger.error(f"Uygulama hatası: {e}")
        print(f"Hata: {e}")

if __name__ == "__main__":
    main()
