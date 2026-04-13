import streamlit as st
from pypdf import PdfReader

@st.cache_data
def read_file(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        return "".join([p.extract_text() for p in reader.pages])
    return file.read().decode("utf-8")