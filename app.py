import time, streamlit as st
from agent import agent_router

st.set_page_config(page_title="Aarshika's Chatbot", page_icon="🤖")

with st.sidebar:
    st.header("⚙️ Settings")
    if "show_tools" not in st.session_state: st.session_state.show_tools = True
    st.session_state.show_tools = st.checkbox("Show Tool Used", value=st.session_state.show_tools)
    st.caption("Agent routes between RAG, Calculator, and General.")
    with st.expander("ℹ️ About Agent Tools", expanded=False):
        st.markdown("""
**The agent automatically decides how to answer your query using three tools:**

**RAG (Retrieval-Augmented Generation):**  
Uses documents to answer questions about Aarshika and her work or hobbies.

**Calculator:**  
Handles straightforward mathematical queries such as arithmetic, multiplication, division, and percentages.

**General:**  
A general-purpose LLM response for all other queries not suited for RAG or Calculator.
""")

st.title("🧠 Aarshika's Chatbot Agent")

if "messages" not in st.session_state: st.session_state.messages = []
if "pending" not in st.session_state: st.session_state.pending = None

user_in = st.text_input(
    "Ask anything in general or specific to Aarshika…",
    placeholder="Ask anything in general or specific to Aarshika…",
    key="user_in",
    label_visibility="collapsed"
)
col1, col2 = st.columns([1,1])
with col1: send = st.button("📤 Send", use_container_width=True)
with col2: clear = st.button("🗑️ Clear", use_container_width=True)

if clear:
    st.session_state.messages.clear(); st.rerun()
if send and user_in:
    st.session_state.pending = user_in; st.rerun()

if st.session_state.pending:
    q = st.session_state.pending; st.session_state.pending=None
    st.session_state.messages.append({"role":"user","content":q})
    with st.spinner("Thinking..."):
        start=time.time()
        try:
            hist=[m for m in st.session_state.messages if m["role"] in ("user","assistant")]
            res=agent_router(q, chat_history=hist)
            st.session_state.messages.append({"role":"assistant","content":res["answer"]})
            if res.get("sources"): st.session_state.messages.append({"role":"sources","content":res["sources"]})
            st.session_state.messages.append({"role":"tool","content":res["type"]})
            st.session_state.messages.append({"role":"timer","content":f"🕒 {time.time()-start:.2f}s"})
        except Exception as e:
            st.session_state.messages.append({"role":"assistant","content":f"❌ {e}"})
    st.rerun()

# display (most recent first, always Q→A pairs)
msgs = st.session_state.messages

pairs = []
i = 0
while i < len(msgs):
    if msgs[i]["role"] == "user" and i+1 < len(msgs) and msgs[i+1]["role"] == "assistant":
        user_msg = msgs[i]
        assistant_msg = msgs[i+1]
        # collect optional meta that follow
        j = i+2
        meta = []
        while j < len(msgs) and msgs[j]["role"] in ("sources","tool","timer"):
            meta.append(msgs[j])
            j += 1
        pairs.append((user_msg, assistant_msg, meta))
        i = j
    else:
        i += 1

# now render newest first
for user_msg, assistant_msg, meta in reversed(pairs):
    with st.chat_message("user"):
        st.markdown(user_msg["content"])
    with st.chat_message("assistant"):
        st.markdown(assistant_msg["content"])
        tool = next((m for m in meta if m["role"]=="tool"), None)
        src  = next((m for m in meta if m["role"]=="sources"), None)
        tmr  = next((m for m in meta if m["role"]=="timer"), None)
        if st.session_state.show_tools and tool: st.markdown(f"**Tool:** {tool['content']}")
        if tool and tool["content"]=="RAG" and src: st.markdown("**Sources:** "+", ".join(src["content"]))
        if tmr: st.markdown(tmr["content"])
