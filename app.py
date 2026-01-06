import streamlit as st
import numpy as np
from datetime import datetime, date, time, timedelta
import plotly.graph_objects as go
import io
from scipy.io.wavfile import write, read
from scipy.fft import rfft, rfftfreq

# ===============================
# НАСТРОЙКА
# ===============================
st.set_page_config("🎵 Оркестр времени", "🎵", layout="wide")

SAMPLE_RATE = 44100
BASE_DURATION = 0.8

# ===============================
# SESSION STATE (КЛЮЧЕВО!)
# ===============================
if "selected_time" not in st.session_state:
    st.session_state.selected_time = datetime.now().time()

# ===============================
# ИНСТРУМЕНТЫ ПО ЧАСАМ
# ===============================
HOUR_INSTRUMENTS = {
    range(0, 6):  ("sine", 55),
    range(6, 12): ("triangle", 110),
    range(12,18): ("square", 220),
    range(18,24): ("sawtooth", 110)
}

def instrument_for_hour(hour):
    for r, inst in HOUR_INSTRUMENTS.items():
        if hour in r:
            return inst
    return "sine", 110

# ===============================
# ВОЛНЫ (БЕЗОПАСНЫЙ ENVELOPE)
# ===============================
def waveform(freq, duration, wave_type):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    n = len(t)

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

    attack = min(int(0.05*SAMPLE_RATE), n//2)
    decay  = min(int(0.25*SAMPLE_RATE), n//2)

    env = np.ones(n)
    if attack > 0:
        env[:attack] = np.linspace(0,1,attack)
    if decay > 0:
        env[-decay:] = np.linspace(1,0,decay)

    return w * env

# ===============================
# ЗВУК ДЛЯ ВРЕМЕНИ
# ===============================
def sound_for_time(t: time):
    h, m, s = t.hour, t.minute, t.second
    wave_type, base = instrument_for_hour(h)

    f_hour = base * 2**(h/12)
    f_min  = f_hour * (1 + m/60)
    f_sec  = f_hour * 4

    main = waveform(f_hour, BASE_DURATION, wave_type)
    interval = waveform(f_min, BASE_DURATION, "sine") * 0.6

    pulse = 0.4 if s % 2 == 0 else 0.2
    tick = waveform(f_sec, 0.12, "square") * pulse
    tick = np.pad(tick, (0, len(main)-len(tick)))

    signal = main + interval + tick
    signal /= np.max(np.abs(signal))

    return signal.astype(np.float32)

# ===============================
# WAV
# ===============================
def wav_bytes(signal):
    buf = io.BytesIO()
    write(buf, SAMPLE_RATE, signal)
    return buf.getvalue()

# ===============================
# ОБРАТНЫЙ АНАЛИЗ ЗВУКА
# ===============================
def infer_time_from_audio(wav_bytes):
    sr, data = read(io.BytesIO(wav_bytes))
    if data.ndim > 1:
        data = data.mean(axis=1)

    window = int(BASE_DURATION * sr)
    base_candidates = [55, 110, 220]

    hour_candidates = []
    minute_candidates = []

    for i in range(0, len(data) - window, window):
        chunk = data[i:i + window]

        spectrum = np.abs(rfft(chunk))
        freqs = rfftfreq(len(chunk), 1 / sr)

        peak_freq = freqs[np.argmax(spectrum)]

        best_error = np.inf
        best_hour = None
        best_minute = None

        for base in base_candidates:
            # восстановление часа
            hour = round(12 * np.log2(peak_freq / base))
            if not (0 <= hour <= 23):
                continue

            f_hour = base * 2**(hour / 12)
            minute = round((peak_freq / f_hour - 1) * 60)

            if not (0 <= minute <= 59):
                continue

            recon_freq = f_hour * (1 + minute / 60)
            error = abs(recon_freq - peak_freq)

            if error < best_error:
                best_error = error
                best_hour = hour
                best_minute = minute

        if best_hour is not None:
            hour_candidates.append(best_hour)
            minute_candidates.append(best_minute)

    if not hour_candidates:
        return None

    hour = int(np.median(hour_candidates))
    minute = int(np.median(minute_candidates))

    return hour, minute


# ===============================
# UI
# ===============================
st.title("🎵 Оркестр времени")
st.caption("Прямой и обратный звуковой код времени")

st.divider()
mode = st.radio("Режим:", ["Одно время", "Запись диапазона", "Определить время по звуку"], horizontal=True)

# ===============================
# ОДНО ВРЕМЯ
# ===============================
if mode == "Одно время":
    st.session_state.selected_time = st.time_input(
        "Выберите время",
        value=st.session_state.selected_time
    )

    signal = sound_for_time(st.session_state.selected_time)

    if st.button("▶️ Проиграть"):
        st.audio(wav_bytes(signal), format="audio/wav")

# ===============================
# ЗАПИСЬ ДИАПАЗОНА
# ===============================
elif mode == "Запись диапазона":
    t1 = st.time_input("Начало", time(12,0,0))
    t2 = st.time_input("Конец", time(12,1,0))
    step = st.number_input("Шаг (сек)", 1, 10, 1)

    if st.button("⏺ Создать запись"):
        cur = datetime.combine(date.today(), t1)
        end = datetime.combine(date.today(), t2)

        chunks = []
        while cur <= end:
            chunks.append(sound_for_time(cur.time()))
            cur += timedelta(seconds=step)

        full = np.concatenate(chunks)
        audio = wav_bytes(full)

        st.audio(audio, format="audio/wav")
        st.download_button("⬇️ Скачать WAV", audio, "time_recording.wav")

# ===============================
# ОБРАТНЫЙ АНАЛИЗ
# ===============================
else:
    uploaded = st.file_uploader("Загрузите WAV файл", type=["wav"])

    if uploaded:
        hour, minute = infer_time_from_audio(uploaded.read())
        st.success(f"🕰 Предполагаемое время: **{hour:02d}:{minute:02d}**")

st.divider()
st.caption("⚠️ Обратное определение времени — приближённое (FFT-анализ)")

