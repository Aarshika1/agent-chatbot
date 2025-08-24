from typing import List, Dict
from tools.rag import answer as rag_answer
from tools.calculator import answer as calc_answer
from tools.general import answer as general_answer, classify_route

def agent_router(user_query: str, chat_history: List[Dict[str, str]] | None = None):
    route = classify_route(user_query, chat_history)

    if route == "RAG":
        answer, sources = rag_answer(user_query, chat_history)
        return {"answer": answer, "sources": sources, "type": "RAG"}
    elif route == "Calculator":
        return {"answer": calc_answer(user_query), "sources": [], "type": "Calculator"}
    else:
        return {"answer": general_answer(user_query, chat_history), "sources": [], "type": "General"}
