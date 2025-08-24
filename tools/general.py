import os, streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

ENDPOINT = "https://models.inference.ai.azure.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or st.secrets.get("GITHUB_TOKEN")
ROUTER_MODEL = os.getenv("GHM_ROUTER_MODEL", "gpt-4o-mini")
GENERAL_MODEL = os.getenv("GHM_MODEL", "gpt-4o-mini")

def _llm(model: str, temp=0.0) -> ChatOpenAI:
    return ChatOpenAI(model=model, api_key=GITHUB_TOKEN, base_url=ENDPOINT, temperature=temp)

def classify_route(user_query: str, chat_history=None) -> str:
    sys_prompt = ("Decide route: 'RAG' for Aarshika-related, 'Calculator' for math, 'General' otherwise.")
    msgs = [SystemMessage(content=sys_prompt), HumanMessage(content=user_query)]
    decision = _llm(ROUTER_MODEL).invoke(msgs).content.strip().upper()
    if decision.startswith("RAG"): return "RAG"
    if decision.startswith("CALCULATOR"): return "Calculator"
    return "General"

def answer(query: str, chat_history=None) -> str:
    base_sys = "You are a concise, helpful assistant."
    msgs = [SystemMessage(content=base_sys)]
    if chat_history: msgs += [HumanMessage(m["content"]) if m["role"]=="user" else AIMessage(m["content"]) for m in chat_history]
    msgs.append(HumanMessage(content=query))
    return _llm(GENERAL_MODEL, temp=0.2).invoke(msgs).content
