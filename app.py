"""Sign Language Caption Bot - Streamlit front end.

voice note / video -> real captions (faster-whisper) -> payload for the Logisim RV32I processor
                   -> you read 3 numbers back from Logisim -> the app checks them.

    pip install streamlit faster-whisper
    streamlit run app.py
Keep app.py, rv32i_bridge.py and caption_rom.mem in the same folder.
"""
from __future__ import annotations

import html
import os
import re
import tempfile
from pathlib import Path

import streamlit as st

from rv32i_bridge import (
    MAX_CHARS, build_ram_image, estimate_cycles, expected_results, format_mem,
    normalize_caption, rom_image,
)

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="Sign Language Caption Bot", page_icon="🤟")


# ------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner=False)
def load_model(size: str):
    from faster_whisper import WhisperModel  # imported lazily so the page loads without it

    return WhisperModel(size, device="cpu", compute_type="int8")


def transcribe(data: bytes, filename: str, size: str, language: str | None) -> dict:
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        model = load_model(size)
        seg_iter, info = model.transcribe(path, language=language, beam_size=1)
        segments = [(s.start, s.end, s.text.strip()) for s in seg_iter if s.text.strip()]
    finally:
        os.unlink(path)
    return {
        "segments": segments,
        "text": " ".join(t for _, _, t in segments),
        "language": info.language,
    }


def _srt_time(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def to_srt(segments) -> str:
    return "\n".join(
        f"{i}\n{_srt_time(a)} --> {_srt_time(b)}\n{t}\n" for i, (a, b, t) in enumerate(segments, 1)
    )


_GLOSS_SKIP = {"a", "an", "the", "is", "are", "am", "was", "were", "be", "been", "to", "of", "do", "does", "did"}


def simple_gloss(text: str) -> str:
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    return " / ".join(w.upper() for w in words if w.lower() not in _GLOSS_SKIP)


def parse_number(raw: str, base: int):
    raw = raw.strip().lower().replace("_", "")
    if base == 16 and raw.startswith("0x"):
        raw = raw[2:]
    try:
        return int(raw, base)
    except ValueError:
        return None


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    model_size = st.selectbox(
        "Speech model", ["tiny", "base", "small"], index=1,
        help="Bigger = more accurate but slower. The first run downloads the model, so do one "
             "test run with internet BEFORE the demo; after that it works offline.",
    )
    language = st.text_input("Language code (blank = auto-detect)", "", help="e.g. en, hi, te")

# ------------------------------------------------------------------ upload + captions
st.title("Sign Language Caption Bot 🤟")
big = st.checkbox("Deaf user mode (large captions)", value=True)

uploaded = st.file_uploader(
    "Upload a voice note or video",
    type=["mp3", "wav", "m4a", "mp4", "mov", "opus", "ogg", "aac", "webm", "mkv"],
)

if uploaded is not None and st.button("Generate captions", type="primary"):
    result = None
    with st.spinner("Transcribing... (first run downloads the speech model)"):
        try:
            result = transcribe(uploaded.getvalue(), uploaded.name, model_size, language.strip() or None)
        except ImportError:
            st.error("Speech model not installed. Run:  pip install faster-whisper")
        except Exception as exc:  # noqa: BLE001 - show any decoding/model error to the user
            st.error(f"Transcription failed: {exc}")
    if result is not None:
        if not result["text"]:
            st.warning("No speech was detected in this file.")
        st.session_state["result"] = result
        st.session_state["caption_text"] = result["text"]
        for k in ("proc_words", "proc_characters", "proc_checksum"):
            st.session_state.pop(k, None)

result = st.session_state.get("result")
if result:
    st.subheader("Captions")
    st.caption(f"Detected language: {result['language']}")
    caption = st.text_area("Caption text (you can correct it)", key="caption_text", height=120)
    if big and caption.strip():
        st.markdown(
            f"<div style='font-size:1.8rem;line-height:1.4'>{html.escape(caption)}</div>",
            unsafe_allow_html=True,
        )
    if result["segments"]:
        with st.expander("Timed captions"):
            for a, b, t in result["segments"]:
                st.write(f"`{_srt_time(a)[:8]}`  {t}")
            st.download_button("Download .srt", to_srt(result["segments"]), file_name="captions.srt")

    gloss = simple_gloss(caption)
    if gloss:
        st.markdown("**Simplified gloss (rule-based, experimental)**")
        st.write(gloss)
        st.caption("Only drops articles and 'to be' verbs and upper-cases the rest. "
                   "This is NOT a real sign-language translation.")

# ------------------------------------------------------------------ processor bridge
caption = st.session_state.get("caption_text", "").strip()
if caption:
    st.divider()
    st.subheader("Processor check (RV32I in Logisim)")
    st.write(
        "The caption text is sent to your Logisim RISC-V processor as a RAM image. "
        "The processor runs a small program that counts the words and characters and adds up "
        "the character codes. You read its three results back from Logisim and the app checks them. "
        "The transfer is file-based (manual load), not live."
    )

    payload, ram_words = build_ram_image(caption)
    if payload != normalize_caption(caption):
        st.warning(f"Only the first {len(payload)} characters go to the processor (limit {MAX_CHARS}); "
                   "the caption above stays complete.")
    mem_text = format_mem(ram_words)
    (HERE / "processor_input_memory.mem").write_text(mem_text)

    c1, c2 = st.columns(2)
    c1.download_button("RAM image (processor_input_memory.mem)", mem_text, file_name="processor_input_memory.mem")
    c2.download_button("Program image (caption_rom.mem)", rom_image(), file_name="caption_rom.mem")

    cycles = estimate_cycles(payload)
    secs = max(1, round(2 * cycles / 1024 + 0.5))
    st.markdown(
        "1. In Logisim Evolution open `rv32i_microprocessor`, right-click the **ROM** -> *Load Image* "
        "-> `caption_rom.mem` (once).\n"
        "2. *Simulate -> Reset Simulation* (Ctrl+R), then right-click the **RAM** -> *Load Image* "
        "-> `processor_input_memory.mem`. Keep RESET at 0.\n"
        f"3. Set the tick frequency to 1024 Hz and enable auto-tick. The program needs about "
        f"**{cycles} clock cycles** ({2 * cycles} ticks, roughly {secs} s). It then spins in a halt loop.\n"
        "4. Right-click the RAM -> *Edit Contents* and type the values of words 0, 1 and 2 below."
    )

    base_label = st.radio("Number format", ["Hex (Logisim hex editor)", "Decimal"], horizontal=True)
    base = 16 if base_label.startswith("Hex") else 10
    fields = [
        ("words", "RAM word 0 - number of words"),
        ("characters", "RAM word 1 - number of characters"),
        ("checksum", "RAM word 2 - sum of character codes"),
    ]
    typed = {key: st.text_input(label, key=f"proc_{key}") for key, label in fields}

    if all(v.strip() for v in typed.values()):
        exp = expected_results(payload)
        rows, ok = [], True
        for key, label in fields:
            got = parse_number(typed[key], base)
            match = got == exp[key]
            ok &= match
            rows.append({
                "Value": key,
                "Python expects": exp[key],
                "Processor returned": got if got is not None else "not a number",
                "Match": "OK" if match else "MISMATCH",
            })
        st.table(rows)
        if ok:
            st.success("Verified: the RV32I processor's results match the caption.")
        else:
            st.error("Mismatch. Check: ROM image loaded, RAM image loaded AFTER Ctrl+R, enough ticks, "
                     "and the number format (hex vs decimal).")
