import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# 1. Setup Environment
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

if google_api_key:
    genai.configure(api_key=google_api_key)

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# --- THE FIX: SWITCHING TO GROQ ---
def get_conversational_chain():
    # We added strict instructions to NOT generate extra questions.
    prompt_template = """
    You are a helpful assistant. Answer the user's question strictly using the provided context. 
    
    Rules:
    1. If the answer is in the context, give a concise and clear answer.
    2. If the answer is NOT in the context, say "The answer is not available in the context."
    3. DO NOT generate follow-up questions, quizzes, or additional lists.
    4. Provide the answer and then END your response.

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    
    model = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.1  # Lower temperature makes the AI less "creative" and more focused.
    )
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# 1. Logic function (No rerun inside here)
def process_question():
    user_question = st.session_state.user_query
    if user_question:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
        if os.path.exists("faiss_index"):
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            chain = get_conversational_chain()
            
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            
            # Add to history
            st.session_state.chat_history.append({"user": user_question, "bot": response["output_text"]})
            
            # CLEAR THE INPUT BOX after processing
            st.session_state.user_query = "" 
        else:
            st.error("Please process PDF first.")

def main():
    st.set_page_config(page_title="Smart Doc Bot", layout="wide")
    st.header("Chat with PDF (Hybrid AI) 💁")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display History
    for chat in st.session_state.chat_history:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**Bot:** {chat['bot']}")
        st.write("---")

    # The Input Box with a CALLBACK (on_change)
    # This calls 'process_question' ONLY when you hit enter
    st.text_input("Ask a Question from the PDF Files", 
                  key="user_query", 
                  on_change=process_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Success!")
if __name__ == "__main__":
    main()