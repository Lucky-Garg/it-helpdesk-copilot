import streamlit as st
import rag

st.set_page_config(page_title="IT Helpdesk Copilot", page_icon="🛠️")
st.title("🛠️ IT Helpdesk Copilot")
st.caption("Answers only from the company IT knowledge base, otherwise escalates.")

@st.cache_resource
def setup():
    rag.ensure_ingested()

setup()

q = st.chat_input("Ask an IT question...")
if q:
    st.chat_message("user").write(q)
    with st.chat_message("assistant"):
        r = rag.ask(q)
        st.write(r["answer"])
        if r["status"] == "answered":
            st.caption("Sources: " + ", ".join(r["sources"]))
        else:
            st.warning("Escalated to service desk")
        with st.expander("How I decided"):
            for line in r["trace"]:
                st.text(line)
