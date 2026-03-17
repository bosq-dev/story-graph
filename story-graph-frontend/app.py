import os
import json

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")


def fetch_sessions() -> list[dict]:
    response = requests.get(f"{BACKEND_URL}/chat/sessions", timeout=20)
    response.raise_for_status()
    return response.json()


def fetch_history(session_id: str) -> list[dict]:
    response = requests.get(f"{BACKEND_URL}/chat/history", params={"session_id": session_id}, timeout=20)
    response.raise_for_status()
    return response.json()


def create_session(user_name: str) -> dict:
    response = requests.post(f"{BACKEND_URL}/chat/sessions", json={"user_name": user_name}, timeout=20)
    response.raise_for_status()
    return response.json()

st.set_page_config(page_title="Story Graph Chat", layout="wide")
st.title("Story Graph Demo")
st.caption("Converse normalmente. Enquanto isso, o backend extrai triplas e popula um grafo no Neo4j.")

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "active_user_name" not in st.session_state:
    st.session_state.active_user_name = None
if "messages_by_session" not in st.session_state:
    st.session_state.messages_by_session = {}
if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "entities" not in st.session_state:
    st.session_state.entities = []
if "relations" not in st.session_state:
    st.session_state.relations = []
if "is_sending" not in st.session_state:
    st.session_state.is_sending = False

with st.sidebar:
    st.header("Chats")
    st.caption("Cada sessao representa um usuario diferente no grafo.")

    if st.button("Atualizar lista", use_container_width=True):
        try:
            st.session_state.sessions = fetch_sessions()
        except requests.RequestException as exc:
            st.error(f"Erro ao listar chats: {exc}")

    with st.form("new-chat-form", clear_on_submit=True):
        new_user_name = st.text_input("Nome do usuario", placeholder="Ex.: Fabio")
        create_clicked = st.form_submit_button("Novo chat", use_container_width=True)

    if create_clicked:
        if not new_user_name.strip():
            st.error("Informe o nome do usuario para criar a sessao.")
        else:
            try:
                new_session = create_session(new_user_name.strip())
                st.session_state.sessions = fetch_sessions()
                st.session_state.active_session_id = new_session["id"]
                st.session_state.active_user_name = new_session["user_name"]
                st.session_state.messages_by_session[new_session["id"]] = []
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Erro ao criar chat: {exc}")

    if not st.session_state.sessions:
        try:
            st.session_state.sessions = fetch_sessions()
        except requests.RequestException as exc:
            st.error(f"Erro ao listar chats: {exc}")

    for session in st.session_state.sessions:
        label = session["title"]
        if st.button(label, key=f"session-{session['id']}", use_container_width=True):
            st.session_state.active_session_id = session["id"]
            st.session_state.active_user_name = session["user_name"]
            try:
                history = fetch_history(session["id"])
                st.session_state.messages_by_session[session["id"]] = history
            except requests.RequestException as exc:
                st.error(f"Erro ao carregar historico: {exc}")
            st.rerun()

active_session_id = st.session_state.active_session_id
active_user_name = st.session_state.active_user_name
active_messages = st.session_state.messages_by_session.get(active_session_id, []) if active_session_id else []

left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Chat")

    if active_user_name:
        st.caption(f"Sessao ativa: {active_user_name}")
    else:
        st.info("Crie um novo chat na barra lateral para comecar.")

    for item in active_messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            triplets = item.get("triplets", [])
            if triplets:
                st.caption("Triplas extraidas nesta mensagem:")
                for t in triplets:
                    st.code(
                        f"{t['subject']}, {t['relation']}, {t['object']}, {t['object_type']} (conf={t['confidence']:.2f})"
                    )

    prompt = st.chat_input("Digite sua mensagem", disabled=active_session_id is None or st.session_state.is_sending)
    if prompt and active_session_id and active_user_name:
        st.session_state.is_sending = True
        session_messages = st.session_state.messages_by_session.setdefault(active_session_id, [])
        session_messages.append({"role": "user", "content": prompt})

        payload = {
            "message": prompt,
            "session_id": active_session_id,
            "user_id": "-".join(active_user_name.strip().lower().split()) or "user",
            "user_name": active_user_name,
        }
        status_placeholder = st.empty()
        try:
            status_placeholder.info("Gerando resposta...")
            response = requests.post(
                f"{BACKEND_URL}/chat/message/stream",
                json=payload,
                headers={"Accept": "text/event-stream"},
                stream=True,
                timeout=(5, 180),
            )
            response.raise_for_status()
            done_payload: dict | None = None
            streamed_text = ""
            streamed_triplets = []
            current_event = "message"
            with st.chat_message("assistant"):
                assistant_placeholder = st.empty()
                for line in response.iter_lines(decode_unicode=True):
                    if line is None:
                        continue
                    stripped = line.strip()
                    if not stripped:
                        continue

                    if stripped.startswith("event:"):
                        current_event = stripped.split(":", 1)[1].strip()
                        continue

                    if not stripped.startswith("data:"):
                        continue

                    data_text = stripped.split(":", 1)[1].strip()
                    try:
                        event_payload = json.loads(data_text)
                    except json.JSONDecodeError:
                        continue

                    if current_event == "token":
                        token = str(event_payload.get("text", ""))
                        if token:
                            streamed_text += token
                            assistant_placeholder.markdown(f"{streamed_text}▌")
                    elif current_event == "done":
                        done_payload = event_payload
                        streamed_text = str(done_payload.get("assistant_message", streamed_text))
                        streamed_triplets = done_payload.get("extracted_triplets", [])
                        assistant_placeholder.markdown(streamed_text)
                    elif current_event == "error":
                        detail = str(event_payload.get("detail", "Erro no streaming do backend"))
                        raise requests.RequestException(detail)

                if done_payload is None:
                    raise requests.RequestException("Fluxo SSE terminou sem evento final 'done'.")

                if streamed_triplets:
                    st.caption("Triplas extraidas nesta mensagem:")
                    for t in streamed_triplets:
                        st.code(
                            f"{t['subject']}, {t['relation']}, {t['object']}, {t['object_type']} (conf={t['confidence']:.2f})"
                        )

            st.session_state.active_session_id = done_payload["session_id"]
            st.session_state.active_user_name = done_payload["user_name"]

            session_messages.append(
                {
                    "role": "assistant",
                    "content": streamed_text,
                    "triplets": streamed_triplets,
                }
            )
            st.session_state.sessions = fetch_sessions()
            status_placeholder.empty()
            st.rerun()
        except requests.Timeout:
            status_placeholder.empty()
            st.warning(
                "O backend demorou para responder e a requisicao expirou. "
                "Tente novamente em alguns segundos."
            )
        except requests.RequestException as exc:
            status_placeholder.empty()
            st.error(f"Falha ao chamar backend: {exc}")
        finally:
            st.session_state.is_sending = False

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
