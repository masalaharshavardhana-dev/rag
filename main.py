import streamlit as st
import fitz
from app import perform_task
import os


st.title("Document Retrieval with RAG")
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])
question = st.text_input("Ask a question about the document", placeholder="Type your question here...", disabled=not uploaded_file)


if st.button("Generate Answer", disabled=not (uploaded_file and question)):
    if uploaded_file and question:
        with st.spinner("Processing document and generating answer..."):
            try:
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                document_text = ""
                
                if file_extension == ".pdf":
                    document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                    for page in document:
                        document_text += page.get_text()
                    document.close()
                else: 
                    document_text = uploaded_file.read().decode("utf-8")
                
                result = perform_task(document_text, question)
                st.success("Answer generated successfully!")
                st.write(result)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("Please check your API key and try again.")
    else:
        st.warning("Please upload a file, enter a question, and provide your Hugging Face API key.")