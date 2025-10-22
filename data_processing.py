
from datasets import load_dataset
import logging
from typing import List
from langchain_core.documents import Document
from dotenv import load_dotenv
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def hf_load_dataset(dataset_name: str = "umutertugrul/turkish-hospital-medical-articles"):
    """
    HuggingFace'ten veri setini yükler
    
    Args:
        dataset_name (str): Yüklenecek veri setinin adı
    Returns:
        Dataset: HuggingFace dataset objesi
    """
    try:
        logger.info("Veri seti yükleniyor...")

        hftoken = os.getenv('HUGGINGFACE_TOKEN')
        dataset = load_dataset(dataset_name, token=hftoken)
        logger.info("Veri seti başarıyla yüklendi.")
        return dataset
    except Exception as e:
        logger.error(f"Veri seti yüklenirken hata oluştu: {e}")
        raise

def explore_dataset(dataset):
    """
    Yeni veri setini keşfeder ve analiz eder
    
    Args:
        dataset: HuggingFace dataset objesi (turkish-hospital-medical-articles)
    """
    try:
        print("=== YENİ VERİ SETİ BİLGİLERİ ===")
        print(f"Toplam hastane sayısı: {len(dataset.keys())}")
        
        total_articles = 0
        hospital_stats = {}
        
        for hospital_name in dataset.keys():
            hospital_data = dataset[hospital_name]
            hospital_count = len(hospital_data)
            total_articles += hospital_count
            hospital_stats[hospital_name] = hospital_count
            
            print(f"{hospital_name}: {hospital_count} makale")
        
        print(f"\nToplam makale sayısı: {total_articles}")
        
        sorted_hospitals = sorted(hospital_stats.items(), key=lambda x: x[1], reverse=True)
        print(f"\nEn çok makale olan hastaneler:")
        for i, (hospital, count) in enumerate(sorted_hospitals[:5]):
            print(f"  {i+1}. {hospital}: {count} makale")
        
        first_hospital = list(dataset.keys())[0]
        first_article = dataset[first_hospital][0]
        
        print(f"\n=== İLK MAKALE ÖRNEĞİ ({first_hospital.upper()}) ===")
        print(f"Başlık: {first_article['title']}")
        print(f"Yayın Tarihi: {first_article.get('publish_date', 'Bilinmiyor')}")
        print(f"Güncelleme Tarihi: {first_article.get('update_date', 'Bilinmiyor')}")
        print(f"URL: {first_article.get('url', 'Bilinmiyor')}")
        print(f"İçerik Uzunluğu: {len(first_article['text'])} karakter")
        print(f"İçerik Önizleme:")
        print(f"   {first_article['text'][:300]}...")
        
        print(f"\n=== RASTGELE MAKALE ÖRNEKLERİ ===")
        import random
        
        for i in range(3):
            random_hospital = random.choice(list(dataset.keys()))
            random_article = random.choice(dataset[random_hospital])
            
            print(f"\n--- Örnek {i+1} ({random_hospital}) ---")
            print(f"Başlık: {random_article['title']}")
            print(f"Uzunluk: {len(random_article['text'])} karakter")
            print(f"İçerik: {random_article['text'][:150]}...")
        
        print(f"\n=== VERİ KALİTESİ ANALİZİ ===")
        
        all_lengths = []
        for hospital_name in dataset.keys():
            for article in dataset[hospital_name]:
                all_lengths.append(len(article['text']))
        
        if all_lengths:
            avg_length = sum(all_lengths) / len(all_lengths)
            min_length = min(all_lengths)
            max_length = max(all_lengths)
            
            print(f"Ortalama makale uzunluğu: {avg_length:.0f} karakter")
            print(f"En kısa makale: {min_length} karakter")
            print(f"En uzun makale: {max_length} karakter")
            
            short_articles = len([l for l in all_lengths if l < 500])
            medium_articles = len([l for l in all_lengths if 500 <= l < 2000])
            long_articles = len([l for l in all_lengths if l >= 2000])
            
            print(f"Uzunluk dağılımı:")
            print(f"   Kısa (<500 karakter): {short_articles} makale")
            print(f"   Orta (500-2000 karakter): {medium_articles} makale")
            print(f"   Uzun (>2000 karakter): {long_articles} makale")
        
        print(f"\nVeri seti analizi tamamlandı!")
        
    except Exception as e:
        logger.error(f"Veri !seti keşfedilirken hata: {e}")
        print(f"Hata: {e}")

def create_documents(dataset, max_samples=2000):
    """
    Yeni veri setinden LangChain Document'ları oluşturur
    
    Args:
        dataset: HuggingFace dataset objesi (turkish-hospital-medical-articles)
        max_samples (int, optional): Maksimum işlenecek makale sayısı. None ise tüm veri işlenir
        
    Returns:
        List[Document]: LangChain Document listesi
    """
    try:
        if max_samples is None:
            logger.info("Document'lar oluşturuluyor (TÜM VERİ)...")
        else:
            logger.info(f"Document'lar oluşturuluyor (maksimum {max_samples} makale)...")
        
        documents = []
        total_articles = 0
        skipped_articles = 0
        
        for hospital_name in dataset.keys():
            hospital_data = dataset[hospital_name]
            
            if max_samples is None:
                hospital_articles = len(hospital_data)  # Tüm veri
            else:
                hospital_articles = min(len(hospital_data), max_samples // len(dataset.keys()))
            
            logger.info(f"{hospital_name}: {hospital_articles} makale işleniyor...")
            
            for i in range(hospital_articles):
                article = hospital_data[i]
                
                text_content = article.get('text', '')
                title = article.get('title', f'Başlıksız {i+1}')
                
                if not text_content or text_content is None or text_content.strip() == '':
                    logger.warning(f"Boş içerik atlandı: {title}")
                    skipped_articles += 1
                    continue
                
                if len(text_content.strip()) < 50:
                    logger.warning(f"Çok kısa içerik atlandı: {title} (Uzunluk: {len(text_content)})")
                    skipped_articles += 1
                    continue
                
                content = f"""
Başlık: {title}

İçerik: {text_content}

Kaynak: {hospital_name}
Yayın Tarihi: {article.get('publish_date', 'Bilinmiyor')}
"""
                
                doc = Document(
                    page_content=content.strip(),
                    metadata={
                        'source': hospital_name,
                        'title': title,
                        'publish_date': article.get('publish_date', 'Bilinmiyor'),
                        'url': article.get('url', ''),
                        'article_id': i,
                        'content_length': len(text_content)
                    }
                )
                documents.append(doc)
                total_articles += 1
                
                if max_samples is not None and total_articles >= max_samples:
                    logger.info(f"Max samples ({max_samples}) limitine ulaşıldı, işlem durduruluyor...")
                    break
            
            if max_samples is not None and total_articles >= max_samples:
                break
        
        logger.info(f"Toplam {len(documents)} document oluşturuldu")
        logger.info(f"{skipped_articles} makale atlandı (boş/kısa içerik)")
        if total_articles + skipped_articles > 0:
            logger.info(f"Başarı oranı: {len(documents)/(len(documents)+skipped_articles)*100:.1f}%")
        
        return documents
        
    except Exception as e:
        logger.error(f"Document oluşturulurken hata: {e}")
        raise
