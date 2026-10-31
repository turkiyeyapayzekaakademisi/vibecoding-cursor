"""Tarih ve kültür asistanı için Streamlit arayüzü."""

from __future__ import annotations

import html
import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

DEFAULT_API = os.getenv("API_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
TYPE_LABELS = {
    "kitap": "Kitap",
    "muze": "Müze",
    "belgesel": "Belgesel",
    "birincil_kaynak": "Birincil kaynak",
    "konu": "Konu",
    "yer": "Yer",
}

st.set_page_config(
    page_title="Tarih ve Kültür Asistanı",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #f4efe6; }
      [data-testid="stSidebar"] { background: #2c241c; }
      [data-testid="stSidebar"] * { color: #f4efe6 !important; }
      h1, h2, h3 { font-family: Georgia, serif; color: #2c241c; }
      .hero {
        background: linear-gradient(135deg, #7c2d12, #b45309);
        color: #fff7ed;
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        margin-bottom: 1rem;
      }
      .hero p { margin: 0.4rem 0 0; font-size: 1.05rem; line-height: 1.5; }
      .meta-card, .section-card, .rec-card, .warn-card {
        background: #fffdf8;
        border: 1px solid #e7d7c1;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
      }
      .meta-card { min-height: 96px; }
      .meta-label, .section-label {
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #9a3412;
        font-weight: 700;
        margin-bottom: 0.35rem;
      }
      .meta-value { color: #3f2e1f; font-size: 1.02rem; }
      .chip {
        display: inline-block;
        background: #ffedd5;
        color: #7c2d12;
        border-radius: 999px;
        padding: 0.2rem 0.7rem;
        margin: 0.15rem 0.2rem 0 0;
        font-size: 0.86rem;
      }
      .warn-card { background: #fff7ed; border-color: #fdba74; }
      .rec-card { border-left: 6px solid #b45309; }
      .badge {
        display: inline-block;
        background: #7c2d12;
        color: #fff7ed;
        border-radius: 999px;
        padding: 0.15rem 0.6rem;
        font-size: 0.75rem;
        margin-bottom: 0.4rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("api_base", DEFAULT_API)


def api_url(path: str) -> str:
    return f"{st.session_state.api_base.rstrip('/')}{path}"


def check_health() -> dict | None:
    try:
        response = requests.get(api_url("/health"), timeout=8)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def ask_api(message: str) -> dict:
    response = requests.post(
        api_url("/chat"),
        json={"message": message, "conversation_id": st.session_state.conversation_id},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def clear_conversation() -> None:
    conversation_id = st.session_state.conversation_id
    if conversation_id:
        try:
            requests.delete(api_url(f"/conversations/{conversation_id}"), timeout=8)
        except requests.RequestException:
            pass
    st.session_state.conversation_id = None
    st.session_state.messages = []


def escape(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def render_answer(answer: dict) -> None:
    st.markdown(
        f'<div class="hero"><div class="meta-label">Özet</div><p>{escape(answer.get("ozet"))}</p></div>',
        unsafe_allow_html=True,
    )

    col_period, col_place, col_people = st.columns(3)
    with col_period:
        st.markdown(
            f'<div class="meta-card"><div class="meta-label">Dönem</div>'
            f'<div class="meta-value">{escape(answer.get("donem"))}</div></div>',
            unsafe_allow_html=True,
        )
    with col_place:
        st.markdown(
            f'<div class="meta-card"><div class="meta-label">Yer</div>'
            f'<div class="meta-value">{escape(answer.get("yer"))}</div></div>',
            unsafe_allow_html=True,
        )
    with col_people:
        chips = "".join(f'<span class="chip">{escape(name)}</span>' for name in answer.get("kisiler", []))
        st.markdown(
            f'<div class="meta-card"><div class="meta-label">Kişiler ve aktörler</div>{chips}</div>',
            unsafe_allow_html=True,
        )

    tab_context, tab_culture, tab_uncertain, tab_json = st.tabs(
        ["Tarihi bağlam", "Kültürel anlam", "Belirsizlikler", "Ham JSON"]
    )
    with tab_context:
        st.markdown(
            f'<div class="section-card"><div class="section-label">Ne oldu, neden önemli</div>'
            f'<div class="meta-value">{escape(answer.get("baglam"))}</div></div>',
            unsafe_allow_html=True,
        )
    with tab_culture:
        st.markdown(
            f'<div class="section-card"><div class="section-label">Günlük yaşam ve kültür</div>'
            f'<div class="meta-value">{escape(answer.get("kulturel_anlam"))}</div></div>',
            unsafe_allow_html=True,
        )
    with tab_uncertain:
        uncertainties = answer.get("belirsizlikler") or []
        if not uncertainties:
            st.info("Bu konuda işaretlenen belirsizlik yok.")
        for item in uncertainties:
            st.markdown(
                f'<div class="warn-card"><div class="section-label">Tartışmalı / kesin değil</div>'
                f'<div class="meta-value">{escape(item)}</div></div>',
                unsafe_allow_html=True,
            )
    with tab_json:
        st.json(answer)

    st.markdown("### Öneriler")
    recommendations = answer.get("oneriler") or []
    rec_cols = st.columns(min(2, len(recommendations)) or 1)
    for index, rec in enumerate(recommendations):
        kind = TYPE_LABELS.get(rec.get("tur", ""), rec.get("tur", ""))
        with rec_cols[index % len(rec_cols)]:
            st.markdown(
                f'<div class="rec-card"><span class="badge">{escape(kind)}</span>'
                f'<h4 style="margin:0.2rem 0 0.5rem;">{escape(rec.get("baslik"))}</h4>'
                f'<p style="margin:0 0 0.4rem;">{escape(rec.get("neden"))}</p>'
                f'<div class="meta-label">{escape(rec.get("donem_veya_yer"))}</div></div>',
                unsafe_allow_html=True,
            )


def render_sidebar() -> None:
    st.sidebar.title("Asistan")
    st.sidebar.caption("Structured JSON sohbet arayüzü")
    st.session_state.api_base = st.sidebar.text_input("API adresi", value=st.session_state.api_base)

    health = check_health()
    if health and health.get("status") == "ok":
        st.sidebar.success(f"API hazır · {health.get('model')}")
    else:
        st.sidebar.error("API'ye bağlanılamadı. FastAPI'nin çalıştığını kontrol et.")

    if st.session_state.conversation_id:
        st.sidebar.caption(f"Sohbet: {st.session_state.conversation_id[:8]}…")
    else:
        st.sidebar.caption("Yeni sohbet")

    if st.sidebar.button("Sohbeti temizle", use_container_width=True):
        clear_conversation()
        st.rerun()


def main() -> None:
    init_state()
    render_sidebar()

    st.title("Tarih ve Kültür Asistanı")
    st.caption("Yanıtlar özet, bağlam, belirsizlik ve öneri bölümleriyle gelir.")

    if not st.session_state.messages:
        st.info("Bir dönem, kişi, gelenek veya şehir sor. Örneğin: *İstanbul’da gezilecek tarihi yerler*")

    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            if item["role"] == "user":
                st.write(item["text"])
            else:
                render_answer(item["answer"])

    prompt = st.chat_input("Tarih veya kültür hakkında sor...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Kaynaklar taranıyor..."):
            try:
                payload = ask_api(prompt)
            except requests.HTTPError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(f"API hatası: {detail}")
                st.session_state.messages.pop()
                return
            except requests.RequestException as exc:
                st.error(f"Bağlantı hatası: {exc}")
                st.session_state.messages.pop()
                return

        st.session_state.conversation_id = payload.get("conversation_id")
        answer = payload.get("answer") or {}
        st.session_state.messages.append({"role": "assistant", "answer": answer})
        render_answer(answer)


if __name__ == "__main__":
    main()
