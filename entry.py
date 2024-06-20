from documentEncoder import encode_docs,encode_query,read_pdf,create_faiss_index,save_faiss_index
import faiss
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

project_directory = os.getcwd()



gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')

document_corpus=[]
absolute_file_path = os.path.abspath("Bogdan_Radu_Vancea_CV.pdf")
text=read_pdf(absolute_file_path)
embedding=encode_docs(text)
index=create_faiss_index(embedding)
save_faiss_index(index,project_directory)

document_corpus.append(text)

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

