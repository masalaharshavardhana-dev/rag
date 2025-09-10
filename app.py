import textwrap
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

def perform_task(document_text, question):
    def chunk_text(text, chunk_size=200):
        return textwrap.wrap(text, width=chunk_size)
    chunks=chunk_text(document_text)
    embed_model=SentenceTransformer('all-MiniLM-L6-v2')
    embeddings=embed_model.encode(chunks)
    index=faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    
    qa_pipeline=pipeline("text2text-generation", model="google/flan-t5-small")
    def retrieval_and_generation(query, top_k=1):
        query_embedding=embed_model.encode([query]) 
        _,indices=index.search(query_embedding, top_k)
        retrieved_texts=[chunks[i] for i in indices[0]]
        context=" ".join(retrieved_texts)
        prompt=f"context:{context} \nQuestion: {question} \nAnswer: "
        result=qa_pipeline(prompt, max_length=100, do_sample=False)
        return result[0]["generated_text"]
    return retrieval_and_generation(question)