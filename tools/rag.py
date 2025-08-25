import os, re
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are Aarshika's website assistant. Answer ONLY with information from context.\n"
        "If the answer is not clearly supported, respond politely:\n"
        "\"I'm not sure based on Aarshika's documents. If you point me to the right file or section, I can learn it.\"\n\n"
        "Priorities:\n"
        "- Use 'Aarshika_Singh_Portfolio_RAG.pdf' for profile, skills, contact.\n"
        "- Use project PDFs for project details.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    ),
)


ENDPOINT = "https://models.inference.ai.azure.com"
from dotenv import load_dotenv; load_dotenv()
import streamlit as st
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or st.secrets.get("GITHUB_TOKEN")
CHAT_MODEL = os.getenv("GHM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("GHM_EMBED", "text-embedding-3-small")

DOCS_DIR = "documents"
FAISS_DIR = "faiss_index"

def clean_text(text: str) -> str:
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    return text.strip()

def load_documents(folder: str):
    docs = []
    if not os.path.isdir(folder): return docs
    for fn in os.listdir(folder):
        path = os.path.join(folder, fn)
        if fn.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            for d in loader.load():
                d.page_content = clean_text(d.page_content)
                docs.append(d)
        elif fn.lower().endswith(".csv"):
            df = pd.read_csv(path, encoding="utf-8", errors="ignore")
            docs.append(Document(page_content=clean_text(df.to_csv(index=False)), metadata={"source": path}))
    return docs

def split_docs(docs, size=1000, overlap=200):
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_documents(docs)

def build_index():
    docs = load_documents(DOCS_DIR)
    chunks = split_docs(docs)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=GITHUB_TOKEN, base_url=ENDPOINT)
    vs = FAISS.from_documents(chunks, embeddings)
    os.makedirs(FAISS_DIR, exist_ok=True)
    vs.save_local(FAISS_DIR)
    return vs

def load_index():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=GITHUB_TOKEN, base_url=ENDPOINT)
    if not os.path.isdir(FAISS_DIR): return build_index()
    return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)

def answer(query: str, chat_history=None):
    retriever = load_index().as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2, api_key=GITHUB_TOKEN, base_url=ENDPOINT)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True,chain_type_kwargs={"prompt": PROMPT})

    history_str = "".join([f"{t['role']}: {t['content']}\n" for t in (chat_history or [])])
    custom_prompt = f"You are Aarshika's assistant.\n{history_str}\nUser query: {query}"

    result = qa.invoke({"query": custom_prompt})
    ans = result["result"]
    sources = []
    for doc in result.get("source_documents", []):
        fn = os.path.basename(doc.metadata.get("source", "Unknown"))
        pg = doc.metadata.get("page")
        sources.append(f"{fn}, page {pg+1}" if pg is not None else fn)
    
    if not sources or not ans.strip() or "I'm not sure" in ans:
        ans = "I'm not sure based on Aarshika's documents. If you point me to the right file or section, I can learn it."
        return ans, []

    return ans, sorted(set(sources))
