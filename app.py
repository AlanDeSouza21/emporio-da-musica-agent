import streamlit as st

from src.agent import ask, create_chat

st.set_page_config(page_title="Empório da Música", page_icon="🎸")
st.title("🎸 Empório da Música")
st.caption("Virtual assistant for customer support")

# One agent chat per browser session; history is kept in session_state
if "chat" not in st.session_state:
    st.session_state.chat = create_chat()
    st.session_state.messages = []

with st.sidebar:
    if st.button("New conversation"):
        st.session_state.chat = create_chat()
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Digite sua mensagem..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                answer, _ = ask(st.session_state.chat, prompt)
            except Exception as exc:
                answer = f"Desculpe, tive um problema técnico: {exc}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})