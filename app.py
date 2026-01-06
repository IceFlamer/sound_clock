import streamlit as st
import numpy as np
from datetime import datetime, date, time, timedelta
import io
from scipy.io.wavfile import write, read
from scipy.fft import rfft, rfftfreq
from collections import Counter

# ===============================
# НАСТРОЙКА
# ===============================
st.set_page_config("🎵 Оркестр времени", "🎵", layout="wide")
SAMPLE_RATE = 44100
BASE_DURATION = 0.8
BASE_NOTE = 110.0  # A2

# ===============================
# SESSION STATE
# ===============================
if "selected_time" not in st.session_state:
    st.session_state.selected_time = datetime.now().time()

# ===============================
# ВОЛНЫ С ENVELOPE
# ===============================
def waveform(freq, duration, wave_type):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    n = len(t)
    if wave_type == "sine":
        w = np.sin(2 * np.pi * freq * t)
    elif wave_type == "square":
        w = np.sign(np.sin(2 * np.pi * freq * t))
    elif wave_type == "triangle":
        w = 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
    elif wave_type == "sawtooth":
        w = 2 * (t * freq - np.floor(t * freq + 0.5))
    else:
        w = np.sin(2 * np.pi * freq * t)
    # Envelope
    attack = min(int(0.05 * SAMPLE_RATE), n // 2)
    decay = min(int(0.25 * SAMPLE_RATE), n // 2)
    env = np.ones(n)
    if attack > 0:
        env[:attack] = np.linspace(0, 1, attack)
    if decay > 0:
        env[-decay:] = np.linspace(1, 0, decay)
    return w * env

# ===============================
# ГЕНЕРАЦИЯ ЗВУКА ПО ВРЕМЕНИ (НОВАЯ СХЕМА)
# ===============================
def sound_for_time(t: time):
    h, m, s = t.hour, t.minute, t.second

    # ЧАС: 0–23 → 0–23 полутона от A2
    f_hour = BASE_NOTE * (2 ** (h / 12))

    # МИНУТЫ: каждые 5 минут → 1 полутон (0–11)
    m_step = (m // 5) % 12
    f_min = BASE_NOTE * (2 ** (m_step / 12))

    # Основные тона
    tone_h = waveform(f_hour, BASE_DURATION, "sine") * 0.7
    tone_m = waveform(f_min, BASE_DURATION, "sine") * 0.5

    # СЕКУНДЫ: ритм — (s % 4) + 1 тиков
    num_ticks = (s % 4) + 1
    tick_signal = np.zeros_like(tone_h)
    for i in range(num_ticks):
        tick = waveform(880, 0.06, "square") * 0.3
        start = int(i * 0.15 * SAMPLE_RATE)
        end = start + len(tick)
        if end <= len(tick_signal):
            tick_signal[start:end] += tick

    signal = tone_h + tone_m + tick_signal
    signal = signal / (np.max(np.abs(signal)) + 1e-6)
    return signal.astype(np.float32)

# ===============================
# WAV УТИЛИТЫ
# ===============================
def wav_bytes(signal):
    buf = io.BytesIO()
    write(buf, SAMPLE_RATE, signal)
    return buf.getvalue()

# ===============================
# ОБРАТНЫЙ АНАЛИЗ (НАДЁЖНЫЙ)
# ===============================
def infer_time_from_audio(wav_bytes_data):
    sr, data = read(io.BytesIO(wav_bytes_data))
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.float32)
    
    window = int(BASE_DURATION * sr)
    if len(data) < window:
        return None
    chunk = data[:window]  # достаточно первого сегмента

    # Спектр
    spectrum = np.abs(rfft(chunk))
    freqs = rfftfreq(len(chunk), 1 / sr)

    # Найдём ТОП-2 пика (игнорируем очень низкие частоты)
    # Уберём всё ниже 50 Гц
    valid = freqs >= 50
    spectrum = spectrum[valid]
    freqs = freqs[valid]

    # Находим два самых сильных пика
    peak_indices = np.argsort(spectrum)[-2:][::-1]  # два самых больших
    peak_freqs = freqs[peak_indices]

    # Сортируем по частоте: нижняя — час, верхняя — минуты (обычно)
    f1, f2 = sorted(peak_freqs[:2])
    candidates = []

    # Перебираем все возможные времена
    for hour in range(24):
        for minute in range(60):
            f_h_expected = BASE_NOTE * (2 ** (hour / 12))
            f_m_expected = BASE_NOTE * (2 ** ((minute // 5) % 12 / 12))
            # Сравниваем с двумя пиками
            err1 = abs(f1 - f_h_expected) + abs(f2 - f_m_expected)
            err2 = abs(f1 - f_m_expected) + abs(f2 - f_h_expected)  # на случай перепутанных
            error = min(err1, err2)
            if error < 30:  # допуск ±15 Гц на каждый
                candidates.append((error, hour, minute))

    if not candidates:
        return None

    candidates.sort()
    _, best_hour, best_minute = candidates[0]
    return int(best_hour), int(best_minute)

# ===============================
# UI
# ===============================
st.title("🎵 Оркестр времени")
st.caption("Прямой и обратный звуковой код времени (музыкальная схема)")
st.divider()
mode = st.radio("Режим:", ["Одно время", "Запись диапазона", "Определить время по звуку"], horizontal=True)

# ОДНО ВРЕМЯ
if mode == "Одно время":
    st.session_state.selected_time = st.time_input("Выберите время", value=st.session_state.selected_time)
    signal = sound_for_time(st.session_state.selected_time)
    if st.button("▶️ Проиграть"):
        st.audio(wav_bytes(signal), format="audio/wav")

# ЗАПИСЬ ДИАПАЗОНА
elif mode == "Запись диапазона":
    t1 = st.time_input("Начало", time(12, 0, 0))
    t2 = st.time_input("Конец", time(12, 1, 0))
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

# ОБРАТНЫЙ АНАЛИЗ
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
        st.caption("⚠️ Обратное определение времени — на основе двух пиков и перебора")
