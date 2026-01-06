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
        data = data.mean(axis=1).astype(np.float32)
    
    # Нормализуем, чтобы избежать переполнения
    if np.max(np.abs(data)) > 0:
        data = data / np.max(np.abs(data))
    
    window = int(BASE_DURATION * sr)
    base_candidates = [55, 110, 220]
    candidates = []  # список (ошибка, час, минута)

    # Анализируем первый фрагмент (обычно его достаточно)
    chunk = data[:window]
    if len(chunk) < window:
        return None

    spectrum = np.abs(rfft(chunk))
    freqs = rfftfreq(len(chunk), 1 / sr)
    peak_idx = np.argmax(spectrum)
    peak_freq = freqs[peak_idx]

    # Перебираем все возможные base и часы
    for base in base_candidates:
        for hour in range(24):
            f_hour = base * (2 ** (hour / 12))
            # Ожидаемая частота минутного тона
            for minute in range(60):
                f_min_expected = f_hour * (1 + minute / 60)
                error = abs(f_min_expected - peak_freq)
                # Добавляем кандидата с малой ошибкой
                if error < 10:  # допуск ±10 Гц — можно настроить
                    candidates.append((error, hour, minute))

    if not candidates:
        return None

    # Выбираем кандидата с минимальной ошибкой
    candidates.sort()
    _, best_hour, best_minute = candidates[0]
    return int(best_hour), int(best_minute)



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
        result = infer_time_from_audio(uploaded.read())
        if result is not None:
            hour, minute = result
            st.success(f"🕰 Предполагаемое время: **{hour:02d}:{minute:02d}**")
        else:
            st.error("❌ Не удалось распознать время. Убедитесь, что файл создан этим приложением.")
        st.divider()
        st.caption("⚠️ Обратное определение времени — приближённое (FFT-анализ)")



