from transformers import DPRContextEncoder,DPRContextEncoderTokenizer
import faiss
import numpy as np
from PyPDF2 import PdfReader


encoder_tokenizer=DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
encoder_model=DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

def read_pdf(file_name):
    reader=PdfReader(file_name)
    text=""
    for page in reader.pages:
        text=text+page.extract_text()
    return text

def read_all(file_names):
    texts = []
    for file_name in file_names:
        text = read_pdf(file_name)
        texts.append(text)
    return texts

def encode_docs(documents):
    embeddings=[]
    for doc in documents:
        inputs=encoder_tokenizer(doc,return_tensors='pt',truncation=True,padding=True)
        outputs=encoder_model(**inputs)
        embedding=outputs.pooler_output.detach().numpy()
        embeddings.append(embedding)
    return np.vstack(embeddings)


def encode_query(query):
    inputs = encoder_tokenizer(query, return_tensors='pt', truncation=True, padding=True)
    outputs = encoder_model(**inputs)
    embedding = outputs.pooler_output.detach().numpy()
    return embedding

def create_faiss_index(embeddings):
    embedding_size = embeddings.shape[1]
    index=faiss.IndexFlatL2(embedding_size)
    index.add(embeddings)
    return index

def save_faiss_index(index,path):
    faiss.write_index(index,path)

def load_faiss_index(path):
    index=faiss.read_index(path)
    return index

def add_docs_to_index(index,new_docs):
    new_embeddings=encode_docs(new_docs)
    index.add(new_embeddings.cpu().numpy())
    return index
