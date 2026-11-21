import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="Urban Tandoor Menu Assistant", page_icon="🍛")
st.title("🍛 Urban Tandoor Menu Assistant")
st.caption("Ask about dishes, prices, add-ons, ingredients or branches. Urdu, Roman Urdu and English are fine.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    st.header("Try asking")
    for ex in ["What biryani do you have and how much?",
               "Kuch meetha batao",
               "Spicy chicken dish under Rs 1000?",
               "What is in the Biryani Duo deal?"]:
        if st.button(ex, use_container_width=True):
            st.session_state["pending"] = ex
    if st.button("Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask about the menu...") or st.session_state.pop("pending", None)

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Checking the menu..."):
                payload = {"messages": [{"role": m["role"], "content": m["content"]}
                                        for m in st.session_state["messages"]]}
                r = requests.post(f"{API}/chat", json=payload, timeout=60)
                r.raise_for_status()
                data = r.json()
            answer = data["answer"]
            st.markdown(answer)
            if data["tools_used"]:
                with st.expander("What the bot looked up in the backend"):
                    for t in data["tools_used"]:
                        st.code(f'{t["tool"]}({t["input"]})')
        except requests.exceptions.ConnectionError:
            answer = "The backend is not running. Start it with: uvicorn main:app --reload"
            st.error(answer)
        except Exception as e:
            answer = f"Sorry, something went wrong: {e}"
            st.error(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})