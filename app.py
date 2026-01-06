import streamlit as st
import numpy as np
from datetime import datetime, date, time, timedelta
import plotly.graph_objects as go
import io
from scipy.io.wavfile import write

# ===============================
# НАСТРОЙКА
# ===============================
st.set_page_config("🎵 Звуковые часы", "🎵", layout="wide")

SAMPLE_RATE = 44100
BASE_DURATION = 0.8

# ===============================
# ИНСТРУМЕНТЫ ПО ЧАСАМ
# ===============================
HOUR_INSTRUMENTS = {
    range(0, 6):  ("sine",  55),   # ночь
    range(6, 12): ("triangle",110),# утро
    range(12,18): ("square", 220), # день
    range(18,24): ("sawtooth",110) # вечер
}

def instrument_for_hour(hour):
    for r, inst in HOUR_INSTRUMENTS.items():
        if hour in r:
            return inst
    return "sine", 110

# ===============================
# ВОЛНЫ
# ===============================
def waveform(freq, duration, wave_type):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)

    if wave_type == "sine":
        w = np.sin(2*np.pi*freq*t)
    elif wave_type == "square":
        w = np.sign(np.sin(2*np.pi*freq*t))
    elif wave_type == "triangle":
        w = 2*np.abs(2*(t*freq-np.floor(t*freq+0.5)))-1
    elif wave_type == "sawtooth":
        w = 2*(t*freq-np.floor(t*freq+0.5))
    else:
        w = np.sin(2*np.pi*freq*t)

    # envelope
    a = int(0.05*SAMPLE_RATE)
    d = int(0.25*SAMPLE_RATE)
    env = np.ones_like(w)
    env[:a] = np.linspace(0,1,a)
    env[-d:] = np.linspace(1,0,d)

    return w*env

# ===============================
# ЗВУК ДЛЯ ВРЕМЕНИ (ПОЛИФОНИЯ)
# ===============================
def sound_for_time(t: time):
    hour, minute, second = t.hour, t.minute, t.second

    wave_type, base = instrument_for_hour(hour)

    # Ноты
    f_hour = base * 2**(hour/12)
    f_min  = f_hour * (1 + minute/60)
    f_sec  = f_hour * 4

    main = waveform(f_hour, BASE_DURATION, wave_type)
    interval = waveform(f_min, BASE_DURATION, "sine") * 0.6

    # секундный тик
    pulse = 0.4 if second % 2 == 0 else 0.2
    tick = waveform(f_sec, 0.12, "square") * pulse
    tick = np.pad(tick, (0, len(main)-len(tick)))

    signal = main + interval + tick
    signal /= np.max(np.abs(signal))

    return signal.astype(np.float32)

# ===============================
# WAV В ПАМЯТИ
# ===============================
def wav_bytes(signal):
    buf = io.BytesIO()
    write(buf, SAMPLE_RATE, signal)
    return buf.getvalue()

# ===============================
# UI
# ===============================
st.title("🎵 Оркестр времени")
st.caption("Часы, минуты и секунды как музыка")

st.divider()

# -------- РЕЖИМ --------
mode = st.radio("Режим:", ["Одно время", "Запись диапазона"], horizontal=True)

# ===============================
# ОДНО ВРЕМЯ
# ===============================
if mode == "Одно время":
    selected = st.time_input("Выберите время", datetime.now().time())

    signal = sound_for_time(selected)

    col1, col2 = st.columns([1,2])

    with col1:
        st.markdown(
            f"<div style='font-size:2.5rem;text-align:center'>{selected.strftime('%H:%M:%S')}</div>",
            unsafe_allow_html=True
        )
        if st.button("▶️ Проиграть"):
            st.audio(wav_bytes(signal), format="audio/wav")

    with col2:
        t = np.linspace(0, BASE_DURATION, len(signal))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t[:500], y=signal[:500], mode="lines"))
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

# ===============================
# ЗАПИСЬ ДИАПАЗОНА
# ===============================
else:
    st.subheader("🎙 Запись временного диапазона")

    c1, c2, c3 = st.columns(3)
    with c1:
        t_start = st.time_input("Начало", time(12,0,0))
    with c2:
        t_end = st.time_input("Конец", time(12,1,0))
    with c3:
        step = st.number_input("Шаг (сек)", 1, 10, 1)

    if st.button("⏺ Создать запись"):
        cur = datetime.combine(date.today(), t_start)
        end = datetime.combine(date.today(), t_end)

        chunks = []
        while cur <= end:
            chunks.append(sound_for_time(cur.time()))
            cur += timedelta(seconds=step)

        full = np.concatenate(chunks)
        audio = wav_bytes(full)

        st.success("Запись готова")
        st.audio(audio, format="audio/wav")
        st.download_button(
            "⬇️ Скачать WAV",
            audio,
            file_name="time_recording.wav",
            mime="audio/wav"
        )

st.divider()
st.caption("🔊 Полифония: часы + минуты + секунды • Инструменты по времени суток")
