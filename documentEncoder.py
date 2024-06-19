from transformers import DPRContextEncoder,DPRContextEncoderTokenizer
import faiss
import numpy as np

encoder_tokenizer=DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
encoder_model=DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

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