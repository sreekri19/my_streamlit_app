import streamlit as st
import fitz  # PyMuPDF for PDF processing
from transformers import BertTokenizer, BertModel
import torch
from rapidfuzz import fuzz

# Initialize BERT model and tokenizer
@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

tokenizer, model = load_bert_model()

# Utility Functions
def extract_text_from_pdf(pdf_file_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_file_path)
    extracted_text = []
    for page in doc:
        blocks = page.get_text("dict")['blocks']
        for block in blocks:
            if 'lines' not in block:
                continue
            for line in block['lines']:
                for span in line['spans']:
                    text = span['text']
                    font_size = span['size']
                    font_flags = span['flags']
                    if font_size > 12 and font_flags == 20:
                        extracted_text.append(("heading", text.strip()))
                    else:
                        extracted_text.append(("paragraph", text.strip()))
    return extracted_text

def normalize_text(text):
    """Normalize the extracted text."""
    return ''.join(text.split()).lower()

def find_exact_match(search_term, text_blocks):
    """Find an exact match for the search term in the extracted PDF text."""
    normalized_search_term = normalize_text(search_term)
    
    for block_type, block_content in text_blocks:
        normalized_block_content = normalize_text(block_content)
        if normalized_search_term in normalized_block_content:
            return block_type, block_content
    return None

def find_closest_match_fuzzy(search_term, text_blocks):
    """Find the closest match using fuzzy matching with rapidfuzz."""
    best_matches = []
    normalized_search_term = normalize_text(search_term)
    
    for block_type, block_content in text_blocks:
        normalized_block_content = normalize_text(block_content)
        similarity = fuzz.ratio(normalized_search_term, normalized_block_content)
        if similarity > 75:
            best_matches.append((block_type, block_content, similarity))
    
    return sorted(best_matches, key=lambda x: x[2], reverse=True)

def get_paragraphs_from_heading(text_blocks, matched_heading):
    """Retrieve paragraphs starting from the matched heading."""
    paragraphs = []
    is_in_section = False
    for block_type, block_content in text_blocks:
        if block_type == "heading" and block_content == matched_heading:
            is_in_section = True
            paragraphs.append(block_content)
        elif block_type == "heading" and is_in_section:
            break
        elif block_type == "paragraph" and is_in_section:
            paragraphs.append(block_content)
    return paragraphs

def get_paragraphs_from_paragraph(text_blocks, matched_paragraph, num_paragraphs=10):
    """Get surrounding paragraphs for contextual understanding."""
    paragraphs = []
    for i, (block_type, block_content) in enumerate(text_blocks):
        if block_content == matched_paragraph and block_type == "paragraph":
            start = i
            end = min(i + num_paragraphs, len(text_blocks))
            paragraphs = [content for _, content in text_blocks[start:end] if _ == "paragraph"]
            break
    return paragraphs

def match_text_in_pdfs(query_text, pdf_file_list):
    """Match the input text against text in multiple PDF files and return the best match."""
    best_match_pdf = None
    best_match_context = None

    for pdf_file in pdf_file_list:
        text_blocks = extract_text_from_pdf(pdf_file)

        exact_match = find_exact_match(query_text, text_blocks)
        if exact_match:
            block_type, block_content = exact_match
            if block_type == "heading":
                relevant_paragraphs = get_paragraphs_from_heading(text_blocks, block_content)
                return pdf_file, "\n".join(relevant_paragraphs)
            else:
                surrounding_paragraphs = get_paragraphs_from_paragraph(text_blocks, block_content)
                return pdf_file, "\n".join(surrounding_paragraphs)

        best_matches = find_closest_match_fuzzy(query_text, text_blocks)
        if best_matches:
            for block_type, block_content, similarity in best_matches:
                if block_type == "heading":
                    relevant_paragraphs = get_paragraphs_from_heading(text_blocks, block_content)
                    return pdf_file, "\n".join(relevant_paragraphs)
                else:
                    surrounding_paragraphs = get_paragraphs_from_paragraph(text_blocks, block_content)
                    return pdf_file, "\n".join(surrounding_paragraphs)

    return None, None

# Streamlit UI
st.title('PDF Text Matching with Transformer Models')

# User input text
query_text = st.text_input("Enter the text to search:")

# File paths of PDF files
pdf_files = [
    "data/2004-08-11 Davidson v. Burns_Depo Transcript of Thomas Zagurski.pdf",
        "data/2006-08-29 Basile v. Honda_Depo Transcript of Paul LeCour.pdf",
        "data/2006-08-29 Basile v. Honda_Exhibits to Depo of Paul LeCour.pdf",
        "data/2008-02-21 Bradford v. AW Chesterton_Depo Transcript of Paul LeCour.pdf",
        "data/2010-10-12 Bankhead v. Allied_Depo Transcript of Ludlow Earle Bretz 2.pdf",
        "data/2010-10-12 Bankhead v. Allied_Depo Transcript of Ludlow Earle Bretz.pdf"
        "data/2015-06-24 Reed v. 3M_Depo Transcript of Albert Indelicato.pdf"
]
if st.button("Search"):
    if query_text:
        matched_pdf, matched_context = match_text_in_pdfs(query_text, pdf_files)
        if matched_pdf:
            st.write(f"Best match found in: {matched_pdf}")
            st.write(f"Context:\n{matched_context}")
        else:
            st.write("No match found.")
    else:
        st.write("Please enter a search query.")

