import json
import os

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


st.title("Admin Graph Chat")
st.caption("Interface separada para administracao e consultas no grafo via tools.")

if "admin_active_session_id" not in st.session_state:
    st.session_state.admin_active_session_id = None
if "admin_active_user_name" not in st.session_state:
    st.session_state.admin_active_user_name = None
if "admin_messages_by_session" not in st.session_state:
    st.session_state.admin_messages_by_session = {}
if "admin_sessions" not in st.session_state:
    st.session_state.admin_sessions = []
if "admin_is_sending" not in st.session_state:
    st.session_state.admin_is_sending = False
if "admin_tool_events" not in st.session_state:
    st.session_state.admin_tool_events = []

with st.sidebar:
    st.header("Admin Sessions")

    if st.button("Atualizar lista", use_container_width=True):
        try:
            st.session_state.admin_sessions = fetch_sessions()
        except requests.RequestException as exc:
            st.error(f"Erro ao listar sessoes: {exc}")

    with st.form("new-admin-session", clear_on_submit=True):
        new_user_name = st.text_input("Nome da sessao admin", value="Admin")
        create_clicked = st.form_submit_button("Nova sessao", use_container_width=True)

    if create_clicked:
        try:
            session = create_session(new_user_name.strip() or "Admin")
            st.session_state.admin_sessions = fetch_sessions()
            st.session_state.admin_active_session_id = session["id"]
            st.session_state.admin_active_user_name = session["user_name"]
            st.session_state.admin_messages_by_session[session["id"]] = []
            st.session_state.admin_tool_events = []
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Erro ao criar sessao: {exc}")

    if not st.session_state.admin_sessions:
        try:
            st.session_state.admin_sessions = fetch_sessions()
        except requests.RequestException as exc:
            st.error(f"Erro ao listar sessoes: {exc}")

    for session in st.session_state.admin_sessions:
        label = session["title"]
        if st.button(label, key=f"admin-session-{session['id']}", use_container_width=True):
            st.session_state.admin_active_session_id = session["id"]
            st.session_state.admin_active_user_name = session["user_name"]
            try:
                history = fetch_history(session["id"])
                st.session_state.admin_messages_by_session[session["id"]] = history
            except requests.RequestException as exc:
                st.error(f"Erro ao carregar historico: {exc}")
            st.session_state.admin_tool_events = []
            st.rerun()

active_session_id = st.session_state.admin_active_session_id
active_user_name = st.session_state.admin_active_user_name
messages = st.session_state.admin_messages_by_session.get(active_session_id, []) if active_session_id else []

left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Chat")
    if active_user_name:
        st.caption(f"Sessao ativa: {active_user_name}")
    else:
        st.info("Crie ou selecione uma sessao admin na barra lateral.")

    for item in messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    prompt = st.chat_input(
        "Pergunte sobre o grafo",
        disabled=active_session_id is None or st.session_state.admin_is_sending,
    )

    if prompt and active_session_id and active_user_name:
        st.session_state.admin_is_sending = True
        session_messages = st.session_state.admin_messages_by_session.setdefault(active_session_id, [])
        session_messages.append({"role": "user", "content": prompt})

        payload = {
            "message": prompt,
            "session_id": active_session_id,
            "user_id": "admin",
            "user_name": active_user_name,
        }

        status_placeholder = st.empty()
        try:
            status_placeholder.info("Executando tools de grafo...")
            response = requests.post(
                f"{BACKEND_URL}/admin/chat/message/stream",
                json=payload,
                headers={"Accept": "text/event-stream"},
                stream=True,
                timeout=(5, 180),
            )
            response.raise_for_status()

            current_event = "message"
            last_event = "message"
            done_payload = None
            streamed_text = ""
            tool_events: list[dict] = []

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
                        last_event = current_event
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
                    elif current_event == "tool_call":
                        tool_events.append({"type": "tool_call", "payload": event_payload})
                    elif current_event == "tool_result":
                        tool_events.append({"type": "tool_result", "payload": event_payload})
                    elif current_event == "done":
                        done_payload = event_payload
                        streamed_text = str(done_payload.get("assistant_message", streamed_text))
                        assistant_placeholder.markdown(streamed_text)
                    elif current_event == "error":
                        detail = str(event_payload.get("detail", "Erro no streaming admin"))
                        raise requests.RequestException(detail)

                if done_payload is None:
                    partial = (streamed_text[:160] + "...") if len(streamed_text) > 160 else streamed_text
                    raise requests.RequestException(
                        "Fluxo SSE terminou sem evento final 'done'. "
                        f"Ultimo evento: {last_event}. "
                        f"Tool events recebidos: {len(tool_events)}. "
                        f"Texto parcial: {partial or '<vazio>'}"
                    )

            session_messages.append({"role": "assistant", "content": streamed_text})
            st.session_state.admin_tool_events = tool_events
            st.session_state.admin_sessions = fetch_sessions()
            status_placeholder.empty()
            st.rerun()
        except requests.Timeout:
            status_placeholder.empty()
            st.warning("Timeout na resposta do backend admin. Tente novamente em alguns segundos.")
        except requests.RequestException as exc:
            status_placeholder.empty()
            st.error(
                "Falha ao chamar backend admin. "
                f"Detalhes: {exc}. "
                "Verifique logs do container backend para o request_id retornado em evento 'error'."
            )
        finally:
            st.session_state.admin_is_sending = False

with right_col:
    st.subheader("Observabilidade de Tools")
    tool_events = st.session_state.get("admin_tool_events", [])
    if not tool_events:
        st.caption("Sem eventos nesta sessao ainda.")
    else:
        for index, item in enumerate(tool_events[::-1], start=1):
            with st.expander(f"{index}. {item['type']}", expanded=index <= 2):
                st.json(item["payload"])
