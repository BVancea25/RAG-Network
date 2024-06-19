from documentEncoder import encode_docs,encode_query
import faiss
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')

document_corpus = [
    "Green tea has many health benefits, one of the biggest is it contains antioxidants.",
    "Exercise is essential for a healthy lifestyle.",
    "Balanced diet is important for maintaining good health."
]

document_embeddings=encode_docs(document_corpus)
embedding_size = document_embeddings.shape[1]
index=faiss.IndexFlatL2(embedding_size)
index.add(document_embeddings)

print("Introduce query \n")
query=input()
query_embedding=encode_query(query)

k = 2
distances, indices = index.search(query_embedding, k)
retrieved_documents = [document_corpus[i] for i in indices[0]]


gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
# Combine the query with retrieved documents
combined_input = query + "\n" + "\n".join(retrieved_documents)

# Tokenize combined input
inputs = gpt_tokenizer(combined_input, return_tensors='pt', truncation=True, padding=True)

# Generate response
outputs = gpt_model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)
response = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)

