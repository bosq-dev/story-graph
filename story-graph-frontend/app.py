import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="Story Graph Chat", layout="wide")
st.title("Story Graph Demo")
st.caption("Converse normalmente. Enquanto isso, o backend extrai triplas e popula um grafo no Neo4j.")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Chat")

    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            triplets = item.get("triplets", [])
            if triplets:
                st.caption("Triplas extraidas nesta mensagem:")
                for t in triplets:
                    st.code(
                        f"{t['subject']}, {t['relation']}, {t['object']}, {t['object_type']} (conf={t['confidence']:.2f})"
                    )

    prompt = st.chat_input("Digite sua mensagem")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        payload = {
            "message": prompt,
            "session_id": st.session_state.session_id,
            "user_id": "demo-user",
        }
        try:
            response = requests.post(f"{BACKEND_URL}/chat/message", json=payload, timeout=40)
            response.raise_for_status()
            data = response.json()

            st.session_state.session_id = data["session_id"]
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": data["assistant_message"],
                    "triplets": data.get("extracted_triplets", []),
                }
            )
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Falha ao chamar backend: {exc}")

with right_col:
    st.subheader("Grafo")

    if st.button("Atualizar entidades"):
        try:
            entities_resp = requests.get(f"{BACKEND_URL}/graph/entities", timeout=20)
            entities_resp.raise_for_status()
            st.session_state.entities = entities_resp.json()
        except requests.RequestException as exc:
            st.error(f"Erro ao buscar entidades: {exc}")

    if st.button("Atualizar relacoes"):
        try:
            rel_resp = requests.get(f"{BACKEND_URL}/graph/relations", timeout=20)
            rel_resp.raise_for_status()
            st.session_state.relations = rel_resp.json()
        except requests.RequestException as exc:
            st.error(f"Erro ao buscar relacoes: {exc}")

    entities = st.session_state.get("entities", [])
    relations = st.session_state.get("relations", [])

    st.markdown("**Entidades**")
    st.dataframe(entities, use_container_width=True, hide_index=True)

    st.markdown("**Relacoes**")
    st.dataframe(relations, use_container_width=True, hide_index=True)
