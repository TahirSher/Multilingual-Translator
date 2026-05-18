import os
import re
import fitz  # PyMuPDF
import torch
import streamlit as st

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.platypus.flowables import KeepTogether

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multilingual PDF Translator",
    layout="wide"
)

st.title("🌍 Multilingual PDF Translator")
st.write(
    "Upload a PDF document, select source and target languages, "
    "and download the translated PDF."
)

# ============================================================
# LANGUAGE CONFIG
# ============================================================

LANGUAGES = {
    "English": "eng_Latn",
    "Korean": "kor_Hang",
    "Urdu": "urd_Arab",
    "Hindi": "hin_Deva",
    "Arabic": "arb_Arab",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Spanish": "spa_Latn",
    "Chinese": "zho_Hans",
    "Japanese": "jpn_Jpan",
    "Russian": "rus_Cyrl",
    "Turkish": "tur_Latn",
    "Italian": "ita_Latn",
    "Portuguese": "por_Latn"
}

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model_name = "facebook/nllb-200-distilled-600M"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return tokenizer, model, device

tokenizer, model, device = load_model()

# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

    text = ""

    for page in doc:
        text += page.get_text()

    return text

# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text_into_chunks(text, max_chars=1500):
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# ============================================================
# TRANSLATION
# ============================================================

def translate_chunk(chunk, src_lang, tgt_lang):

    tokenizer.src_lang = src_lang

    encoded = tokenizer(
        chunk,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_length=1024,
        num_beams=4
    )

    translated_text = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )[0]

    return translated_text

# ============================================================
# FULL DOCUMENT TRANSLATION
# ============================================================

def translate_document(text, src_lang, tgt_lang):

    chunks = split_text_into_chunks(text)

    translated_chunks = []

    progress_bar = st.progress(0)

    for idx, chunk in enumerate(chunks):

        translated = translate_chunk(chunk, src_lang, tgt_lang)

        translated_chunks.append(translated)

        progress_bar.progress((idx + 1) / len(chunks))

    return "\n\n".join(translated_chunks)

# ============================================================
# PDF GENERATION
# ============================================================

def create_translated_pdf(text, output_path):

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    story = []

    paragraphs = text.split("\n")

    for para in paragraphs:

        para = para.strip()

        if para:
            story.append(Paragraph(para, styles['BodyText']))
            story.append(Spacer(1, 10))

    doc.build(story)

# ============================================================
# UI
# ============================================================

uploaded_pdf = st.file_uploader(
    "Upload PDF File",
    type=["pdf"]
)

col1, col2 = st.columns(2)

with col1:
    source_language = st.selectbox(
        "Select Source Language",
        list(LANGUAGES.keys()),
        index=0
    )

with col2:
    target_language = st.selectbox(
        "Select Target Language",
        list(LANGUAGES.keys()),
        index=1
    )

# ============================================================
# PROCESSING
# ============================================================

if uploaded_pdf is not None:

    if st.button("Translate PDF"):

        with st.spinner("Extracting text from PDF..."):

            extracted_text = extract_text_from_pdf(uploaded_pdf)

            extracted_text = clean_text(extracted_text)

        st.success("Text extraction completed.")

        st.subheader("Extracted Text Preview")

        st.text_area(
            "Preview",
            extracted_text[:3000],
            height=250
        )

        with st.spinner("Translating document..."):

            translated_text = translate_document(
                extracted_text,
                LANGUAGES[source_language],
                LANGUAGES[target_language]
            )

        st.success("Translation completed.")

        st.subheader("Translated Text Preview")

        st.text_area(
            "Translated Preview",
            translated_text[:3000],
            height=250
        )

        output_pdf = "translated_document.pdf"

        with st.spinner("Generating translated PDF..."):

            create_translated_pdf(
                translated_text,
                output_pdf
            )

        st.success("PDF generated successfully.")

        with open(output_pdf, "rb") as f:

            st.download_button(
                label="📥 Download Translated PDF",
                data=f,
                file_name="translated_document.pdf",
                mime="application/pdf"
            )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    "Built with Hugging Face Transformers + Streamlit"
)
