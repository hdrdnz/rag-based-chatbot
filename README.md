# RAG Tabanlı Türkçe Tıbbi Chatbot

Bu proje, Türkçe tıbbi makaleler üzerinde **Retrieval-Augmented Generation (RAG)** teknolojisi kullanarak çalışan akıllı bir chatbot sistemidir. HuggingFace'ten yüklenen Türk hastanelerinin tıbbi makalelerini analiz ederek, kullanıcıların tıbbi sorularına profesyonel cevaplar verir.

## Özellikler

- **24,000+ Tıbbi Makale**: Türk hastanelerinden toplanmış güncel tıbbi içerik
- **Türkçe Embedding Modeli**: `trmteb/turkish-embedding-model` ile optimize edilmiş arama
- **Google Gemini 2.0 Flash**: Gelişmiş dil modeli ile doğal cevaplar
- **ChromaDB Vector Store**: Hızlı ve etkili vektör arama
- **Streamlit Web Arayüzü**: Kullanıcı dostu web arayüzü

## Proje Yapısı

```
rag-based-chatbot/
├── rag_env/                 # Python sanal ortamı
├── app.py                   # Ana uygulama (konsol)
├── data_processing.py       # Veri yükleme ve işleme
├── embedding_model.py       # Türkçe embedding modeli
├── vector_store.py          # ChromaDB vector store yönetimi
├── rag_pipeline.py          # RAG pipeline mantığı
├── gemini_integration.py    # Google Gemini API entegrasyonu
├── requirements.txt         # Python bağımlılıkları
├── README.md              
└── chroma_db/              # Vector store veritabanı (otomatik oluşur)
```

## Teknik Detaylar

### Veri İşleme Pipeline

1. **Veri Yükleme**: HuggingFace'ten `umutertugrul/turkish-hospital-medical-articles` veri seti
2. **Document Oluşturma**: Makaleler LangChain Document formatına dönüştürülür
3. **Veri Temizleme**: Boş ve kısa içerikler filtrelenir
4. **Kalite Kontrolü**: %98.6 başarı oranı ile işleme

### Kullanılan Teknolojiler

| Kategori | Teknoloji | Versiyon | Açıklama |
|----------|-----------|----------|----------|
| LLM | Google Gemini 2.0 Flash | Latest | Ana dil modeli |
| Embedding | trmteb/turkish-embedding-model | Latest | Türkçe vektörleştirme |
| Vector DB | ChromaDB | 0.4.0+ | Vektör veritabanı |
| Framework | LangChain | 0.1.0+ | RAG framework |
| Web UI | Streamlit | 1.28.0+ | Web arayüzü |
| Data | HuggingFace Datasets | 2.14.0+ | Veri yönetimi |

## Kurulum

### 1. Projeyi Klonlayın

```bash
git clone <repository-url>
cd rag-based-chatbot
```

### 2. Sanal Ortam Oluşturun

```bash
# Sanal ortam oluştur
python -m venv rag_env

# Sanal ortamı aktifleştir
# Windows:
rag_env\Scripts\activate
# macOS/Linux:
source rag_env/bin/activate
```

### 3. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

### 4. API Anahtarlarını Ayarlayın

`.env` dosyası oluşturun:

```bash
# .env dosyası oluştur
touch .env
```

`.env` dosyasına aşağıdaki anahtarları ekleyin:

```env
# Google Gemini API Anahtarı
GOOGLE_API_KEY=your_gemini_api_key_here

# HuggingFace Token (opsiyonel)
HUGGINGFACE_TOKEN=your_hf_token_here
```

#### API Anahtarlarını Nasıl Alırsınız?

**Google Gemini API:**
1. [Google AI Studio](https://makersuite.google.com/app/apikey) adresine gidin
2. Google hesabınızla giriş yapın
3. "Create API Key" butonuna tıklayın
4. Oluşturulan anahtarı kopyalayın

**HuggingFace Token:**
1. [HuggingFace](https://huggingface.co/settings/tokens) adresine gidin
2. "New token" butonuna tıklayın
3. "Read" yetkisi ile token oluşturun

## Kullanım

### 1. Konsol Uygulaması

```bash
python app.py
```

Bu komut:
- Veri setini yükler
- Document'ları oluşturur
- Embedding modelini yükler
- Vector store oluşturur/yükler
- RAG pipeline'ı test eder
- Etkileşimli chat modunu başlatır

### 2. Streamlit Web Arayüzü

```bash
streamlit run app.py
```

Web arayüzü `http://localhost:8501` adresinde açılır.

### 3. Google Colab ile Deneme

**Hızlı Başlangıç için Colab:**

1. **Colab Notebook'u Açın:**
   ```python
   # Colab'da çalıştırın
   !git clone https://github.com/your-username/rag-based-chatbot.git
   %cd rag-based-chatbot
   !pip install -r requirements.txt
   ```

2. **API Anahtarlarını Ayarlayın:**
   ```python
   # Colab'da çalıştırın
   import os
   from google.colab import userdata
   
   # Gemini API anahtarınızı girin
   os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
   
   # HuggingFace token (opsiyonel)
   os.environ['HUGGINGFACE_TOKEN'] = userdata.get('HUGGINGFACE_TOKEN')
   ```

3. **Veri Setini Analiz Edin:**
   ```python
   # Colab'da çalıştırın
   from data_processing import hf_load_dataset, explore_dataset
   
   # Veri setini yükle ve analiz et
   dataset = hf_load_dataset()
   explore_dataset(dataset)
   ```

4. **RAG Pipeline'ı Test Edin:**
   ```python
   # Colab'da çalıştırın
   from app import main
   
   # Ana uygulamayı çalıştır
   main()
   ```

### 4. Ngrok ile Web Arayüzünü Paylaşma

**Ngrok ile Dışarıya Açma:**

1. **Ngrok'u Yükleyin:**
   ```python
   # Colab'da çalıştırın
   !pip install pyngrok
   ```

2. **Streamlit Uygulamasını Başlatın:**
   ```python
   # Colab'da çalıştırın
   import subprocess
   import threading
   from pyngrok import ngrok
   
   def run_streamlit():
       subprocess.run([
           "streamlit", "run", "app.py", 
           "--server.port", "8501",
           "--server.address", "0.0.0.0",
           "--server.headless", "true"
       ])
   
   # Streamlit'i arka planda başlat
   streamlit_thread = threading.Thread(target=run_streamlit)
   streamlit_thread.daemon = True
   streamlit_thread.start()
   
   # Ngrok tüneli oluştur
   public_url = ngrok.connect(8501)
   print(f"Web arayüzü: {public_url}")
   ```

3. **Ngrok Token Ayarlayın (Opsiyonel):**
   ```python
   # Colab'da çalıştırın
   from pyngrok import ngrok
   
   # Ngrok token'ınızı ayarlayın (ücretsiz hesap için)
   ngrok.set_auth_token("your_ngrok_token_here")
   
   # Tünel oluştur
   public_url = ngrok.connect(8501)
   print(f"Paylaşılabilir link: {public_url}")
   ```

**Ngrok Avantajları:**
- **Global Erişim**: Dünyanın her yerinden erişilebilir
- **Mobil Uyumlu**: Telefon ve tabletten de kullanılabilir
- **HTTPS**: Güvenli bağlantı
- **Analytics**: Ziyaretçi istatistikleri
- **Ücretsiz**: Temel kullanım için ücretsiz

**Önemli Notlar:**
- Ngrok tüneli Colab oturumu kapandığında kapanır
- Ücretsiz versiyonda 8 saat süre sınırı vardır
- Her yeniden başlatmada yeni URL oluşur

## Örnek Kullanım

### Tıbbi Sorular

**Soru:** "Migren ağrısı nasıl geçer?"

**Cevap:** "Migren ağrısı için öncelikle karanlık ve sessiz bir ortamda dinlenmek önemlidir. Soğuk kompres uygulayabilir, bol su içebilirsiniz. Stres faktörlerinden kaçınmak ve düzenli uyku saatleri önemlidir. Şiddetli ağrılarda doktor kontrolü gerekebilir."

---

**Soru:** "Grip aşısı ne zaman yaptırmalıyım?"

**Cevap:** "Grip aşısı genellikle sonbahar aylarında (Eylül-Kasım) yaptırılması önerilir. Aşı, grip mevsimi başlamadan önce yapılmalıdır. Risk gruplarındaki kişiler için özellikle önemlidir."