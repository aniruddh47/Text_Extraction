import streamlit as st
from paddleocr import PaddleOCR
from PIL import Image
from pdf2image import convert_from_bytes
import numpy as np
import cv2
import os
import pytesseract
import re

st.set_page_config(page_title="📝 OCR Text Extractor", layout="centered")
st.title("📝 Image & PDF to Text Extractor")

uploaded_file = st.file_uploader("Upload an image (jpg/png) or PDF", type=["jpg", "jpeg", "png", "pdf"])
lang = st.selectbox("Choose OCR language(s)", ["en", "en+hi", "en+fr", "fr", "hi", "mr"])
is_handwritten = st.checkbox("Is this handwritten text?")

def clean_ocr_text(text):
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line or len(line) < 2:
            continue
        line = re.sub(r"\s+", " ", line)
        line = re.sub(r"[^\w\s\-\.\,]", "", line)
        cleaned.append(line)
    return "\n".join(cleaned)

def extract_handwritten_text_tesseract(pil_image):
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    processed_pil = Image.fromarray(gray)
    config = "--oem 1 --psm 6"
    raw_text = pytesseract.image_to_string(processed_pil, config=config, lang="eng")
    cleaned_text = clean_ocr_text(raw_text)
    return raw_text, cleaned_text

def extract_text_sorted_by_position(result):
    lines = []
    if result and isinstance(result[0], list):
        for line in result[0]:
            try:
                box = line[0]
                text = line[1][0]
                top_left = box[0]
                lines.append((top_left[1], top_left[0], text))
            except:
                continue
        lines.sort()
        return "\n".join(line[2] for line in lines)
    return ""

def show_detected_boxes(image_np, result):
    for line in result[0]:
        points = np.array(line[0], dtype=np.int32)
        cv2.polylines(image_np, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    st.image(image_np, caption="📌 Detected Text Regions", use_column_width=True)

def preprocess_and_ocr(pil_image, ocr_engine, temp_filename="temp.png"):
    img_np = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(temp_filename, gray)
    result = ocr_engine.ocr(temp_filename, cls=True)
    os.remove(temp_filename)
    show_detected_boxes(img_np.copy(), result)
    return extract_text_sorted_by_position(result)

if uploaded_file:
    with st.spinner("Extracting text..."):
        extracted_raw = ""
        extracted_clean = ""

        if not is_handwritten:
            ocr = PaddleOCR(use_angle_cls=True, lang=lang,)

        if uploaded_file.type == "application/pdf":
            st.info("📄 Processing PDF...")
            try:
                pdf_bytes = uploaded_file.read()
                pages = convert_from_bytes(pdf_bytes)
                for i, page in enumerate(pages):
                    st.image(page, caption=f"📄 Page {i+1}", use_column_width=True)
                    if is_handwritten:
                        raw, clean = extract_handwritten_text_tesseract(page)
                    else:
                        clean = preprocess_and_ocr(page, ocr, f"temp_page_{i}.png")
                        raw = clean
                    extracted_raw += raw + "\n"
                    extracted_clean += clean + "\n"
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
        else:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="🖼 Uploaded Image", use_column_width=True)
                if is_handwritten:
                    extracted_raw, extracted_clean = extract_handwritten_text_tesseract(image)
                else:
                    extracted_clean = preprocess_and_ocr(image, ocr, "temp_image.png")
                    extracted_raw = extracted_clean
            except Exception as e:
                st.error(f"Error processing image: {e}")

    if extracted_clean.strip():
        st.subheader("📋 Cleaned Text:")
        st.text_area("Cleaned Output", extracted_clean.strip(), height=300)

        st.subheader("🧪 Raw OCR Output:")
        st.text_area("Raw Output", extracted_raw.strip(), height=250)

        st.download_button(
            label="💾 Download Cleaned Text",
            data=extracted_clean.strip(),
            file_name="cleaned_text.txt",
            mime="text/plain"
        )
        st.download_button(
            label="📥 Download Raw OCR Output",
            data=extracted_raw.strip(),
            file_name="raw_ocr_output.txt",
            mime="text/plain"
        )
    else:
        st.error("❌ No text detected! Try improving image clarity.")

# ↓ For Render/Streamlit compatibility
if __name__ == "__main__":
    st.set_option('server.enableCORS', False)
    st.set_option('server.enableXsrfProtection', False)
