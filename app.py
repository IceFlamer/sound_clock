import streamlit as st
import numpy as np
import sounddevice as sd
import time
from datetime import datetime
import threading
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

# Настройка страницы
st.set_page_config(
    page_title="Звуковые Часы: Оркестр Времени",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #FF416C, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
    }
    .instrument-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .time-display {
        font-size: 4rem;
        font-family: 'Courier New', monospace;
        text-align: center;
        background: #1a1a2e;
        padding: 2rem;
        border-radius: 20px;
        border: 3px solid #e94560;
        margin: 2rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #2196F3, #21CBF3);
        color: white;
        font-size: 1.2rem;
        padding: 1rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(33, 203, 243, 0.4);
    }
</style>
""", unsafe_allow_html=True)

class HarmonicTimeClock:
    """Часы, где время кодируется в звуке"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.is_playing = False
        self.volume = 0.3
        self.playback_thread = None
        
        # Расширенная палитра инструментов для каждого часа
        self.hour_instruments = {
            0: ('🌙 Виолончель', 'sawtooth', 65.41, '#1a1a2e'),
            1: ('🌌 Контрабас', 'sine', 73.42, '#16213e'),
            2: ('🌟 Альт', 'triangle', 82.41, '#0f3460'),
            3: ('🎻 Виолончель', 'sine', 87.31, '#1a237e'),
            4: ('🌅 Арфа', 'sine', 98.00, '#283593'),
            5: ('🌄 Флейта', 'sine', 110.00, '#303f9f'),
            6: ('🌤 Гобой', 'sawtooth', 123.47, '#3949ab'),
            7: ('☀️ Кларнет', 'square', 130.81, '#3f51b5'),
            8: ('🔆 Маримба', 'sine', 146.83, '#5c6bc0'),
            9: ('🎐 Колокольчик', 'triangle', 164.81, '#7986cb'),
            10: ('🌈 Челеста', 'sine', 174.61, '#9fa8da'),
            11: ('🎵 Арфа', 'triangle', 196.00, '#c5cae9'),
            12: ('🎻 Скрипка', 'sine', 220.00, '#ff9800'),
            13: ('🎶 Флейта', 'sawtooth', 246.94, '#ffb74d'),
            14: ('🎺 Труба', 'square', 261.63, '#ffcc80'),
            15: ('🥁 Ксилофон', 'sine', 293.66, '#aed581'),
            16: ('🔔 Вибрафон', 'triangle', 329.63, '#81c784'),
            17: ('🎹 Арфа', 'sine', 349.23, '#4db6ac'),
            18: ('🎼 Фортепиано', 'sine', 392.00, '#4dd0e1'),
            19: ('🎛 Орган', 'square', 440.00, '#29b6f6'),
            20: ('🌙 Колокол', 'sawtooth', 493.88, '#0288d1'),
            21: ('🎻 Скрипка', 'sine', 523.25, '#0277bd'),
            22: ('✨ Флейта', 'triangle', 587.33, '#01579b'),
            23: ('🌠 Челеста', 'sine', 659.25, '#311b92'),
        }
        
        # Соответствие нот названиям
        self.note_names = {
            65.41: 'C2', 69.30: 'C#2', 73.42: 'D2', 77.78: 'D#2',
            82.41: 'E2', 87.31: 'F2', 92.50: 'F#2', 98.00: 'G2',
            103.83: 'G#2', 110.00: 'A2', 116.54: 'A#2', 123.47: 'B2',
            130.81: 'C3', 138.59: 'C#3', 146.83: 'D3', 155.56: 'D#3',
            164.81: 'E3', 174.61: 'F3', 185.00: 'F#3', 196.00: 'G3',
            207.65: 'G#3', 220.00: 'A3', 233.08: 'A#3', 246.94: 'B3',
            261.63: 'C4', 277.18: 'C#4', 293.66: 'D4', 311.13
Self
Self
self.is


: 'D#4',
            329.63: 'E4', 349.23: 'F4', 369.99: 'F#4', 392.00: 'G4',
            415.30: 'G#4', 440.00: 'A4', 466.16: 'A#4', 493.88: 'B4',
            523.25: 'C5', 554.37: 'C#5', 587.33: 'D5', 622.25: 'D#5',
            659.25: 'E5', 698.46: 'F5', 739.99: 'F#5', 783.99: 'G5',
            830.61: 'G#5', 880.00: 'A5', 932.33: 'A#5', 987.77: 'B5',
        }
    
    def get_note_name(self, freq):
        """Получить название ноты по частоте"""
        closest_note = min(self.note_names.keys(), key=lambda x: abs(x - freq))
        return self.note_names[closest_note]
    
    def get_waveform(self, freq, duration, wave_type='sine'):
        """Генерация волны определенного типа"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        if wave_type == 'sine':
            wave = np.sin(2 * np.pi * freq * t)
        elif wave_type == 'square':
            wave = np.sign(np.sin(2 * np.pi * freq * t))
        elif wave_type == 'sawtooth':
            wave = 2 * (t * freq - np.floor(0.5 + t * freq))
        elif wave_type == 'triangle':
            wave = 2 * np.abs(2 * (t * freq - np.floor(0.5 + t * freq))) - 1
        else:
            wave = np.sin(2 * np.pi * freq * t)
        
        # Атака и затухание
        envelope = np.ones_like(t)
        attack_samples = int(0.1 * self.sample_rate)
        decay_samples = int(0.2 * self.sample_rate)
        
        if len(t) > attack_samples + decay_samples:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
        
        return wave * envelope
    
    def get_time_notes(self):
        """Получить ноты для текущего времени"""
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        second = now.second
        
        hour_name, wave_type, base_freq, color = self.hour_instruments[hour]
        
        # Минуты определяют высоту ноты
        minute_factor = 1 + (minute / 60)
        minute_freq = base_freq * minute_factor
        
        # Секунды создают пульсацию
        second_pulse = 1.0 if second % 2 == 0 else 0.8
        
        return {
            'hour': hour,
            'minute': minute,
            'second': second,
            'hour_name': hour_name,
            'base_freq': base_freq,
            'current_freq': minute_freq,
            'wave_type': wave_type,
            'color': color,
            'pulse': second_pulse,
            'time_str': now.strftime("%H:%M:%S"),
            'date_str': now.strftime("%d %B %Y")
        }
    
    def generate_sound(self, time_info):
        """Генерация звука для текущего времени"""
        duration = 0.8
        main_wave = self.get_waveform(
            time_info['current_freq'], 
            duration, 
            time_info['wave_type']
        )
        
        # Добавляем обертоны
        harmonics = np.zeros_like(main_wave)
        for i in range(2, 6):
            harmonic = self.get_waveform(
                time_info['current_freq'] * i * 0.5,
                duration,
                'sine'
            )
            harmonics += harmonic * (0.3 / i)
        
        main_wave = 0.7 * main_wave + 0.3 * harmonics
        main_wave *= time_info['pulse']
        
        return main_wave * self.volume
    
    def play_sound_once(self):
        """Однократное воспроизведение текущего времени"""
        try:
            time_info = self.get_time_notes()
            sound = self.generate_sound(time_info)
            sd.play(sound, self.sample_rate)
            return time_info
        except Exception as e:
            st.error(f"Ошибка воспроизведения: {e}")
            return None
    
    def continuous_playback(self):
        """Непрерывное воспроизведение в фоне"""
        last_second = -1
        
        while self.is_playing:
            current_second = datetime.now().second
            
            if current_second != last_second:
Self
Self
self.is


try:
                    time_info = self.get_time_notes()
                    sound = self.generate_sound(time_info)
                    sd.play(sound, self.sample_rate)
                    last_second = current_second
                except:
                    pass
            
            time.sleep(0.1)
    
    def start_continuous(self):
        """Запустить непрерывное воспроизведение"""
        if not self.is_playing:
            self.is_playing = True
            self.playback_thread = threading.Thread(target=self.continuous_playback)
            self.playback_thread.daemon = True
            self.playback_thread.start()
    
    def stop_continuous(self):
        """Остановить непрерывное воспроизведение"""
        self.is_playing = False
        if self.playback_thread:
            self.playback_thread.join(timeout=1)

def create_waveform_plot(time_info, clock):
    """Создание визуализации звуковой волны"""
    sound = clock.generate_sound(time_info)
    t = np.linspace(0, 0.8, len(sound))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=t[:500],  # Показываем только первые 500 точек для наглядности
        y=sound[:500],
        mode='lines',
        name='Звуковая волна',
        line=dict(color=time_info['color'], width=3),
        fill='tozeroy',
        fillcolor=f'rgba{(int(time_info['color'.lstrip("#'][i:i+2], 16) for i in (0, 2, 4)), 0.2)}'
    ))
    
    fig.update_layout(
        title=f"Волна: {time_info['hour_name']}",
        xaxis_title="Время (сек)",
        yaxis_title="Амплитуда",
        template="plotly_dark",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_frequency_plot(time_info, clock):
    """Создание визуализации частот"""
    hours = list(range(24))
    freqs = [clock.hour_instruments[h][2] for h in hours]
    colors = [clock.hour_instruments[h][3] for h in hours]
    names = [clock.hour_instruments[h][0] for h in hours]
    
    current_hour = time_info['hour']
    
    fig = go.Figure()
    
    # Все частоты
    fig.add_trace(go.Scatter(
        x=hours,
        y=freqs,
        mode='markers+lines',
        name='Частоты часов',
        line=dict(color='rgba(100, 100, 100, 0.3)', width=1),
        marker=dict(
            size=[15 if h == current_hour else 8 for h in hours],
            color=colors,
            line=dict(width=2, color='white')
        ),
        text=names,
        hovertemplate='<b>%{text}</b>
Час: %{x}:00
Частота: %{y:.1f} Гц<extra></extra>'
    ))
    
    # Текущая частота
    fig.add_trace(go.Scatter(
        x=[current_hour],
        y=[time_info['current_freq']],
        mode='markers',
        name='Текущая высота',
        marker=dict(
            size=25,
            color='#FFD700',
            symbol='star',
            line=dict(width=3, color='white')
        ),
        text=f"{clock.get_note_name(time_info['current_freq'])} ({time_info['current_freq']:.1f} Гц)",
        hovertemplate='<b>Сейчас</b>
%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Частотная карта дня",
        xaxis_title="Час дня",
        yaxis_title="Частота (Гц)",
        template="plotly_dark",
        height=400,
        showlegend=False
    )
    
    return fig

# Инициализация состояния сессии
if 'clock' not in st.session_state:
    st.session_state.clock = HarmonicTimeClock()
    st.session_state.is_playing = False
    st.session_state.last_played = None

# Заголовок
st.markdown('<h1 class="main-header">🎵 ЗВУКОВЫЕ ЧАСЫ: ОРКЕСТР ВРЕМЕНИ</h1>', unsafe_allow_html=True)
st.markdown("### *Время, которое можно услышать*")

# Основной контент
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Получение текущего времени
    time_info = st.session_state.clock.get_time_notes()
    
    # Отображение времени
    st.markdown(f"""
    <div class="time-display">
        {time_info['time_str']}

        <small style="font-size: 1.5rem; color: #aaa;">{time_info['date_str']}
Self
Self
self.is


file.png
PNG · 3 KB
</small>
    </div>
    """, unsafe_allow_html=True)

# Боковая панель с управлением
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/music-robot.png", width=100)
    st.title("Управление часами")
    
    # Громкость
    volume = st.slider(
        "🔊 Громкость", 
        0.0, 1.0, 
        st.session_state.clock.volume, 
        0.1,
        help="Регулировка громкости звука"
    )
    st.session_state.clock.volume = volume
    
    # Режимы воспроизведения
    st.subheader("Режимы воспроизведения")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        if st.button("▶️ Воспроизвести сейчас", use_container_width=True):
            played_info = st.session_state.clock.play_sound_once()
            if played_info:
                st.session_state.last_played = played_info
                st.success(f"Воспроизведен {played_info['hour_name']}")
    
    with col_b:
        if st.session_state.is_playing:
            if st.button("⏹️ Остановить", use_container_width=True, type="primary"):
                st.session_state.clock.stop_continuous()
                st.session_state.is_playing = False
                st.rerun()
        else:
            if st.button("🔄 Постоянно", use_container_width=True):
                st.session_state.clock.start_continuous()
                st.session_state.is_playing = True
                st.success("Непрерывное воспроизведение запущено")
    
    # Демо-режимы
    st.subheader("Демонстрации")
    
    demo_mode = st.radio(
        "Выберите демо:",
        ["По часам дня", "Все инструменты", "Музыкальная гамма"],
        index=0
    )
    
    if st.button("🎶 Запустить демо", use_container_width=True):
        if demo_mode == "Все инструменты":
            st.info("Воспроизведение всех 24 инструментов...")
            # Здесь можно добавить код для демо всех инструментов
        elif demo_mode == "Музыкальная гамма":
            st.info("Воспроизведение музыкальной гаммы...")
    
    # Информация о текущем звуке
    st.divider()
    st.subheader("🎼 Текущий звук")
    
    note_name = st.session_state.clock.get_note_name(time_info['current_freq'])
    
    st.markdown(f"""
    <div style="background: {time_info['color']}; padding: 1rem; border-radius: 10px; color: white;">
        <h4>{time_info['hour_name']}</h4>
        <p>🎵 Нота: <b>{note_name}</b></p>
        <p>📊 Частота: <b>{time_info['current_freq']:.1f} Гц</b></p>
        <p>🌊 Тип волны: <b>{time_info['wave_type']}</b></p>
        <p>⏰ Час: <b>{time_info['hour']:02d}:00</b></p>
    </div>
    """, unsafe_allow_html=True)

# Основные колонки для визуализаций
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(create_waveform_plot(time_info, st.session_state.clock), 
                   use_container_width=True)

with col2:
    st.plotly_chart(create_frequency_plot(time_info, st.session_state.clock), 
                   use_container_width=True)

# Расширенная информация
st.subheader("🎵 Как это работает?")
tab1, tab2, tab3 = st.tabs(["Концепция", "Кодирование", "Интерпретация"])

with tab1:
    st.markdown("""
    ### Принцип «Оркестр времени»
    
    Каждый час дня имеет свой уникальный **инструмент** и **тональность**:
    
    - **🌙 Ночь (00:00-06:00)**: Низкие, тёплые тембры (виолончель, контрабас)
    - **🌅 Утро (06:00-12:00)**: Светлые, воздушные звуки (флейта, арфа)
    - **☀️ День (12:00-18:00)**: Ясные, определённые тембры (ксилофон, маримба)
    - **🌙 Вечер (18:00-00:00)**: Меланхоличные, глубокие звуки (орган, виолончель)
    """)

with tab2:
    st.markdown(f"""
    ### Текущее кодирование
    
    **{time_info['time_str']}** представлено как:
    
    ```
    ЧАСЫ:   {time_info['hour_name']}
            Базовый тон: {time_info['base_freq']:.1f} Гц ({st.session_state.clock.get_note_name(time_info['base_freq'])})
    
    МИНУТЫ: Множитель высоты: {1 + (time_info['minute'] / 60):.3f}
            Текущий тон: {time_info['current_freq']:.1f} Гц ({note_name})
    
    СЕКУНДЫ: Пу


льсация: {"четная" if time_info['second'] % 2 == 0 else "нечетная"}
             Амплитуда: {time_info['pulse'] * 100:.0f}%
    ```
    """)

with tab3:
    st.markdown("""
    ### Как слушать время?
    
    1. **Определите инструмент** - это укажет приблизительный час
    2. **Услышьте высоту тона** - более высокий звук означает больше минут прошло
    3. **Ощутите пульсацию** - ритм указывает на чётность секунд
    
    **Пример:** Звук флейты с высоким тоном и быстрым пульсом означает 
    утренний час (6-11) ближе к его завершению.
    """)

# Футер
st.divider()
st.caption("""
🎵 *Звуковые часы — концептуальный арт-проект. Время — это музыка, которую можно слышать.*  
🔊 **Рекомендуется использовать наушники** для лучшего восприятия тонких звуковых деталей.
""")

# Авто-обновление каждую секунду
if st.session_state.is_playing:
    time.sleep(0.1)
    st.rerun()

# Предупреждение о совместимости звука
try:
    import sounddevice as sd
    devices = sd.query_devices()
    st.sidebar.info(f"Аудиоустройство: {sd.default.device[0]}")
except:
    st.sidebar.warning("⚠️ Аудиоустройство не найдено. Проверьте настройки звука.")