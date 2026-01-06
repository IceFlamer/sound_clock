import streamlit as st
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import io
from scipy.io.wavfile import write

# -------------------------------
# НАСТРОЙКА СТРАНИЦЫ
# -------------------------------
st.set_page_config(
    page_title="Звуковые часы",
    page_icon="🎵",
    layout="wide"
)

# -------------------------------
# ГЕНЕРАЦИЯ ЗВУКА
# -------------------------------
SAMPLE_RATE = 44100

def waveform(freq, duration=1.0, wave_type="sine"):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)

    if wave_type == "sine":
        wave = np.sin(2 * np.pi * freq * t)
    elif wave_type == "square":
        wave = np.sign(np.sin(2 * np.pi * freq * t))
    elif wave_type == "triangle":
        wave = 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
    elif wave_type == "sawtooth":
        wave = 2 * (t * freq - np.floor(t * freq + 0.5))
    else:
        wave = np.sin(2 * np.pi * freq * t)

    # envelope
    env = np.ones_like(wave)
    a = int(0.05 * SAMPLE_RATE)
    d = int(0.2 * SAMPLE_RATE)
    env[:a] = np.linspace(0, 1, a)
    env[-d:] = np.linspace(1, 0, d)

    return wave * env


def sound_for_time():
    now = datetime.now()
    hour = now.hour
    minute = now.minute

    base_freq = 110 * 2 ** (hour / 12)
    freq = base_freq * (1 + minute / 60)

    wave = waveform(freq, 1.2, "sine")

    # гармоники
    for i in range(2, 5):
        wave += 0.2 / i * waveform(freq * i, 1.2)

    wave /= np.max(np.abs(wave))
    return wave.astype(np.float32), freq, now.strftime("%H:%M:%S")


def wav_bytes(signal):
    buffer = io.BytesIO()
    write(buffer, SAMPLE_RATE, signal)
    return buffer.getvalue()

# -------------------------------
# UI
# -------------------------------
st.markdown(
    "<h1 style='text-align:center'>🎵 Звуковые часы</h1>",
    unsafe_allow_html=True
)

signal, freq, time_str = sound_for_time()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⏰ Текущее время")
    st.markdown(f"""
    <div style="
        font-size:3rem;
        background:#111;
        padding:1rem;
        border-radius:12px;
        text-align:center">
        {time_str}
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶️ Проиграть звук"):
        st.audio(wav_bytes(signal), format="audio/wav")

    st.caption(f"Частота: **{freq:.1f} Гц**")

with col2:
    t = np.linspace(0, 1.2, len(signal))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t[:600],
        y=signal[:600],
        mode="lines"
    ))
    fig.update_layout(
        title="Звуковая волна",
        template="plotly_dark",
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("🔊 Звук воспроизводится напрямую в браузере через WAV")
