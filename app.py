import re
import fitz  # PyMuPDF for PDF processing
from sentence_transformers import SentenceTransformer, util
import torch
import streamlit as st

# Load SBERT model for semantic search (caching for efficiency)
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Utility Functions
def extract_text_from_pdf(pdf_file_path):
    """Extract and clean text from a PDF file."""
    doc = fitz.open(pdf_file_path)
    extracted_text = []
    
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        cleaned_text = format_text(page_text)  # Clean up the text
        extracted_text.append(cleaned_text)
    
    return extracted_text

def format_text(raw_text):
    """
    Clean up and format the extracted text:
    - Remove unwanted line breaks that don’t end with punctuation.
    - Remove excessive spaces.
    """
    # Remove newlines that don't end with punctuation (join split sentences back together)
    formatted_text = re.sub(r'(?<![.!?])\n+', ' ', raw_text)

    # Remove excessive newlines and spaces
    formatted_text = re.sub(r'\n+', ' ', formatted_text)  # Replace remaining newlines with a space
    formatted_text = re.sub(r'\s+', ' ', formatted_text).strip()  # Remove excessive spaces

    return formatted_text

def split_into_sentences(text):
    """Split text into individual sentences for comparison."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return sentences

def get_embeddings(sentences):
    """Generate embeddings for a list of sentences using SBERT."""
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return embeddings

def exact_match(query_text, sentences):
    """Look for an exact match in the sentences."""
    normalized_query = normalize_text(query_text)
    for sentence in sentences:
        if normalized_query in normalize_text(sentence):
            return sentence
    return None

def normalize_text(text):
    """Normalize text by removing extra spaces and lowercasing."""
    return ' '.join(text.split()).lower()

def get_context(sentences, match_idx, context_size=2):
    """Retrieve context around the matched sentence using a sliding window approach."""
    start = max(0, match_idx - context_size)
    end = min(len(sentences), match_idx + context_size + 1)
    context = sentences[start:end]
    
    merged_context = ' '.join(context)
    
    if not merged_context or len(merged_context.strip()) == 0:
        return "No clear context found for the matched text."
    
    return merged_context.strip()

def match_text_in_pdfs(query_text, pdf_file_list, similarity_threshold=0.7, context_size=2):
    """Match the input text against text in multiple PDF files using SBERT embeddings and exact matching."""
    query_embedding = model.encode(query_text, convert_to_tensor=True)

    best_match_pdf = None
    best_match_sentence = None
    best_similarity_score = -1
    best_match_idx = -1

    for pdf_file in pdf_file_list:
        st.write(f"Processing {pdf_file}...")

        # Extract and clean text from the PDF
        text_blocks = extract_text_from_pdf(pdf_file)

        for page_text in text_blocks:
            sentences = split_into_sentences(page_text)

            # First, try exact matching
            exact_match_sentence = exact_match(query_text, sentences)
            if exact_match_sentence:
                match_idx = sentences.index(exact_match_sentence)
                context = get_context(sentences, match_idx, context_size)
                return pdf_file, context  # Early exit once a match is found

            # Generate embeddings for fuzzy matching
            sentence_embeddings = get_embeddings(sentences)
            cosine_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]

            # Collect the best match based on cosine similarity
            best_sentence_idx = torch.argmax(cosine_scores).item()
            best_similarity = cosine_scores[best_sentence_idx].item()

            if best_similarity >= similarity_threshold and best_similarity > best_similarity_score:
                best_similarity_score = best_similarity
                best_match_pdf = pdf_file
                best_match_sentence = sentences[best_sentence_idx]
                best_match_idx = best_sentence_idx

                if best_similarity >= 0.9:
                    context = get_context(sentences, best_match_idx, context_size)
                    return best_match_pdf, context  # Early exit on high confidence

    # If no exact match, return the best fuzzy match with context
    if best_match_pdf:
        context = get_context(sentences, best_match_idx, context_size)
        return best_match_pdf, context

    return None, None

# Streamlit UI
st.title('PDF Text Matching with SBERT')

# User input text
query_text = st.text_input("Enter the text to search:")

# The backend is holding the PDFs; use predefined PDF paths
pdf_files = [
    "/Users/krishnasrinivaschilkamarri/Documents/DeBlase_Hackathon_prob_data/Pneumo Abex - Deposition Database/2004-08-11 Davidson v. Burns_Depo Transcript of Thomas Zagurski.pdf",
    "/Users/krishnasrinivaschilkamarri/Documents/DeBlase_Hackathon_prob_data/Pneumo Abex - Deposition Database/2006-08-29 Basile v. Honda_Depo Transcript of Paul LeCour.pdf",
    "/Users/krishnasrinivaschilkamarri/Documents/DeBlase_Hackathon_prob_data/Pneumo Abex - Deposition Database/2006-08-29 Basile v. Honda_Exhibits to Depo of Paul LeCour.pdf",
    "/Users/krishnasrinivaschilkamarri/Documents/DeBlase_Hackathon_prob_data/Pneumo Abex - Deposition Database/2008-02-21 Bradford v. AW Chesterton_Depo Transcript of Paul LeCour.pdf",
    "/Users/krishnasrinivaschilkamarri/Documents/DeBlase_Hackathon_prob_data/Pneumo Abex - Deposition Database/2010-10-12 Bankhead v. Allied_Depo Transcript of Ludlow Earle Bretz 2.pdf",
    "/Users/krishnasrinivaschilkamarri/Documents/DeBlase_Hackathon_prob_data/Pneumo Abex - Deposition Database/2015-06-24 Reed v. 3M_Depo Transcript of Albert Indelicato.pdf"
]

# Search functionality
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
















































# import streamlit as st
# import fitz  # PyMuPDF's correct import name is 'fitz'
# from transformers import BertTokenizer, BertModel
# import torch
# from rapidfuzz import fuzz

# # Initialize BERT model and tokenizer
# @st.cache_resource
# def load_bert_model():
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertModel.from_pretrained('bert-base-uncased')
#     return tokenizer, model

# tokenizer, model = load_bert_model()

# # Utility Functions
# def extract_text_from_pdf(pdf_file_path):
#     """Extract text from a PDF file."""
#     doc = fitz.open(pdf_file_path)
#     extracted_text = []
#     for page in doc:
#         blocks = page.get_text("dict")['blocks']
#         for block in blocks:
#             if 'lines' not in block:
#                 continue
#             for line in block['lines']:
#                 for span in line['spans']:
#                     text = span['text']
#                     font_size = span['size']
#                     font_flags = span['flags']
#                     if font_size > 12 and font_flags == 20:
#                         extracted_text.append(("heading", text.strip()))
#                     else:
#                         extracted_text.append(("paragraph", text.strip()))
#     return extracted_text

# def normalize_text(text):
#     """Normalize the extracted text."""
#     return ''.join(text.split()).lower()

# def find_exact_match(search_term, text_blocks):
#     """Find an exact match for the search term in the extracted PDF text."""
#     normalized_search_term = normalize_text(search_term)
    
#     for block_type, block_content in text_blocks:
#         normalized_block_content = normalize_text(block_content)
#         if normalized_search_term in normalized_block_content:
#             return block_type, block_content
#     return None

# def find_closest_match_fuzzy(search_term, text_blocks):
#     """Find the closest match using fuzzy matching with rapidfuzz."""
#     best_matches = []
#     normalized_search_term = normalize_text(search_term)
    
#     for block_type, block_content in text_blocks:
#         normalized_block_content = normalize_text(block_content)
#         similarity = fuzz.ratio(normalized_search_term, normalized_block_content)
#         if similarity > 75:
#             best_matches.append((block_type, block_content, similarity))
    
#     return sorted(best_matches, key=lambda x: x[2], reverse=True)

# def get_paragraphs_from_heading(text_blocks, matched_heading):
#     """Retrieve paragraphs starting from the matched heading."""
#     paragraphs = []
#     is_in_section = False
#     for block_type, block_content in text_blocks:
#         if block_type == "heading" and block_content == matched_heading:
#             is_in_section = True
#             paragraphs.append(block_content)
#         elif block_type == "heading" and is_in_section:
#             break
#         elif block_type == "paragraph" and is_in_section:
#             paragraphs.append(block_content)
#     return paragraphs

# def get_paragraphs_from_paragraph(text_blocks, matched_paragraph, num_paragraphs=10):
#     """Get surrounding paragraphs for contextual understanding."""
#     paragraphs = []
#     for i, (block_type, block_content) in enumerate(text_blocks):
#         if block_content == matched_paragraph and block_type == "paragraph":
#             start = i
#             end = min(i + num_paragraphs, len(text_blocks))
#             paragraphs = [content for _, content in text_blocks[start:end] if _ == "paragraph"]
#             break
#     return paragraphs

# def match_text_in_pdfs(query_text, pdf_file_list):
#     """Match the input text against text in multiple PDF files and return the best match."""
#     best_match_pdf = None
#     best_match_context = None

#     for pdf_file in pdf_file_list:
#         text_blocks = extract_text_from_pdf(pdf_file)

#         exact_match = find_exact_match(query_text, text_blocks)
#         if exact_match:
#             block_type, block_content = exact_match
#             if block_type == "heading":
#                 relevant_paragraphs = get_paragraphs_from_heading(text_blocks, block_content)
#                 return pdf_file, "\n".join(relevant_paragraphs)
#             else:
#                 surrounding_paragraphs = get_paragraphs_from_paragraph(text_blocks, block_content)
#                 return pdf_file, "\n".join(surrounding_paragraphs)

#         best_matches = find_closest_match_fuzzy(query_text, text_blocks)
#         if best_matches:
#             for block_type, block_content, similarity in best_matches:
#                 if block_type == "heading":
#                     relevant_paragraphs = get_paragraphs_from_heading(text_blocks, block_content)
#                     return pdf_file, "\n".join(relevant_paragraphs)
#                 else:
#                     surrounding_paragraphs = get_paragraphs_from_paragraph(text_blocks, block_content)
#                     return pdf_file, "\n".join(surrounding_paragraphs)

#     return None, None

# # Streamlit UI
# st.title('PDF Text Matching with Transformer Models')

# # User input text
# query_text = st.text_input("Enter the text to search:")

# # File paths of PDF files
# pdf_files = [
#     "data/2004-08-11 Davidson v. Burns_Depo Transcript of Thomas Zagurski.pdf",
#         "data/2006-08-29 Basile v. Honda_Depo Transcript of Paul LeCour.pdf",
#         "data/2006-08-29 Basile v. Honda_Exhibits to Depo of Paul LeCour.pdf",
#         "data/2008-02-21 Bradford v. AW Chesterton_Depo Transcript of Paul LeCour.pdf",
#         "data/2010-10-12 Bankhead v. Allied_Depo Transcript of Ludlow Earle Bretz 2.pdf",
#         "data/2010-10-12 Bankhead v. Allied_Depo Transcript of Ludlow Earle Bretz.pdf",
#         "data/2015-06-24 Reed v. 3M_Depo Transcript of Albert Indelicato.pdf"
# ]
# if st.button("Search"):
#     if query_text:
#         matched_pdf, matched_context = match_text_in_pdfs(query_text, pdf_files)
#         if matched_pdf:
#             st.write(f"Best match found in: {matched_pdf}")
#             st.write(f"Context:\n{matched_context}")
#         else:
#             st.write("No match found.")
#     else:
#         st.write("Please enter a search query.")

