from langchain.chains.llm_math.base import LLMMathChain
from langchain_openai import ChatOpenAI
import os, streamlit as st

ENDPOINT = "https://models.inference.ai.azure.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or st.secrets.get("GITHUB_TOKEN")
MATH_MODEL = os.getenv("GHM_MATH_MODEL", "gpt-4o-mini")

llm = ChatOpenAI(model=MATH_MODEL, api_key=GITHUB_TOKEN, base_url=ENDPOINT, temperature=0.0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=False)

def answer(query: str) -> str:
    try: return llm_math_chain.run(query).strip()
    except Exception as e: return f"❌ Failed to evaluate: {query}\nError: {e}"
