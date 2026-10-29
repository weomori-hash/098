import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re

# Если Tesseract не добавлен в PATH вашей ОС, раскомментируйте строчку ниже и укажите путь:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="Распознавание Договоров", layout="centered")

st.title("📄 Анализ скан-копий договоров")
st.write("Загрузите изображение договора для извлечения текста и ключевых данных.")

uploaded_file = st.file_uploader("Выберите изображение договора (.jpg, .png)", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    """Предобработка картинки для повышения точности OCR"""
    img = np.array(image)
    # Переводим в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Применяем адаптивную бинаризацию (убирает шумы, тени и выравнивает свет)
    processed_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return processed_img

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженный документ", use_container_width=True)
    
    with st.spinner("Нейросеть обрабатывает документ..."):
        # Очистка изображения
        processed_img = preprocess_image(image)
        
        try:
            # Запуск распознавания (поддержка русского и английского языков)
            extracted_text = pytesseract.image_to_string(processed_img, lang="rus+eng")
            
            st.success("Текст успешно извлечен!")
            
            # Разделение интерфейса на вкладки
            tab1, tab2 = st.tabs(["📄 Весь текст", "🔍 Ключевые данные"])
            
            with tab1:
                st.text_area("Распознанный текст договора", extracted_text, height=400)
            
            with tab2:
                st.subheader("Извлеченные параметры")
                
                # Поиск даты договора (форматы 00.00.0000 или "00" месяца 202X)
                date_match = re.search(r'(\d{2}[.\/ ]\d{2}[.\/ ]\d{4}|"\d{2}"\s+[а-яА-Я]+\s+\d{4})', extracted_text)
                date_val = date_match.group(0) if date_match else "Не найдена"
                
                # Поиск номера договора
                num_match = re.search(r'(?:Договор|№)\s*(?:№)?\s*([A-Za-z0-9№/-]+)', extracted_text, re.IGNORECASE)
                num_val = num_match.group(1) if num_match else "Не найден"
                
                st.markdown(f"**Номер договора:** `{num_val}`")
                st.markdown(f"**Дата заключения:** `{date_val}`")
                
                st.info("💡 Для продвинутого извлечения именованных сущностей (ФИО, ООО, ИНН) рекомендуется интегрировать NLP-библиотеку Natasha.")
                
        except Exception as e:
            st.error(f"Ошибка OCR: {e}")
            st.info("Проверьте, установлен ли Tesseract OCR в вашей системе и добавлены ли языковые пакеты rus и eng.")
