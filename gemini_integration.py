import google.generativeai as genai
import logging
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_gemini():
    """
    Gemini API'sini yapılandırır
    
    Returns:
        bool: Başarılı mı?
    """
    try :
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return False
        
        genai.configure(api_key=api_key)
        logger.info("Gemini API'sı başarıyla yapılandırıldı.")
        return True
    except Exception as e:
        logger.error(f"Gemini API'sı yapılandırılırken hata oluştu: {e}")
        return False

def generate_answer_with_gemini(prompt:str):
    """
    Gemini ile cevap üretir
    
    Args:
        prompt (str): Hazırlanmış prompt
        model_name (str): Kullanılacak model adı
        
    Returns:
        str: Üretilen cevap
    """
    
    try:
        model_name="gemini-2.0-flash"

        model = genai.GenerativeModel(model_name)
        response=model.generate_content(prompt)
        if response.text:
            return response.text
        else:
            return "Üzgünüm, bir hata oluştu. Lütfen daha sonra tekrar deneyin."
    except Exception as e:
        logger.error(f"Gemini ile cevap üretirken hata oluştu: {e}")
        return "Üzgünüm, bir hata oluştu. Lütfen daha sonra tekrar deneyin."