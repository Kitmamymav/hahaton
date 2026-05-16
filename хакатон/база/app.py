import streamlit as st
import pandas as pd
import sqlite3
import json
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import hashlib
import time  
from datetime import datetime
from sklearn.model_selection import train_test_split
from api_client import OilAPIClient

# ==========================================
# 🎨 НАСТРОЙКИ ВИЗУАЛА (СТРОГИЙ ТЕМНЫЙ СТИЛЬ)
# ==========================================
st.set_page_config(
    page_title="AI Платформа контроля качества", 
    page_icon="🛢️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS: Строгий темный фон, белый текст.
custom_css = """
<style>
/* Прячем дефолтное меню Streamlit и футер, но ОСТАВЛЯЕМ header для кнопки меню! */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Фон (приятный темно-серый) */
[data-testid="stAppViewContainer"] {
    background: #1E2129;
}

/* Box боковой панели */
[data-testid="stSidebar"] {
    background: #252932 !important;
    border-right: 1px solid rgba(0, 255, 170, 0.1);
}

/* Уменьшаем отступы в левом баре и гарантируем скролл */
[data-testid="stSidebar"] > div:first-child {
    overflow-y: auto !important;
}
[data-testid="stSidebarUserContent"] {
    padding-top: 2rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    padding-bottom: 2rem !important;
}

/* Убираем второй системный "глаз" браузера в поле пароля */
input[type="password"]::-ms-reveal,
input[type="password"]::-ms-clear {
    display: none;
}

/* Основные заголовки белые */
h1, h2, h3, h4, h5, h6, label { 
    color: #FAFAFA !important; 
}
p, span {
    color: #E2E8F0;
}

/* Делаем зону загрузки файлов темной с белым текстом */
[data-testid="stFileUploadDropzone"] {
    background-color: #1E2129 !important;
    border: 1px dashed rgba(0, 255, 170, 0.5) !important;
}
[data-testid="stFileUploadDropzone"] * {
    color: #FAFAFA !important;
}

/* Исправляем обычные кнопки (Выйти, Browse files) */
button:not([data-testid="baseButton-primary"]) {
    background-color: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}
button:not([data-testid="baseButton-primary"]) p,
button:not([data-testid="baseButton-primary"]) span {
    color: #FAFAFA !important;
}

/* Метрики и формы */
[data-testid="stMetric"], [data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(0, 255, 170, 0.2) !important;
    border-radius: 12px !important;
    padding: 15px !important;
}

/* Главные кнопки */
[data-testid="baseButton-primary"] {
    background: linear-gradient(90deg, #00FFAA, #0088FF) !important;
    color: #000 !important;
    border: none !important;
    font-weight: 800 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    box-shadow: 0 0 10px rgba(0, 255, 170, 0.3) !important;
    transition: all 0.3s ease !important;
}
[data-testid="baseButton-primary"]:hover {
    box-shadow: 0 0 20px rgba(0, 255, 170, 0.6) !important;
    transform: scale(1.02) !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Словарь для красивого перевода колонок
COLUMN_NAMES_RU = {
    "temperature": "Температура (°C)",
    "pressure": "Давление (МПа)",
    "density": "Плотность (кг/м³)",
    "viscosity": "Вязкость (сСт)",
    "sulfur_content": "Содержание серы (%)",
    "flash_point": "Температура вспышки (°C)",
    "water_cut": "Обводненность (%)",
    "salt_content": "Содержание солей (мг/л)",
    "quality_index": "Индекс качества"
}

# ==========================================
# 🛠 БАЗА ДАННЫХ И БЕЗОПАСНОСТЬ
# ==========================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    conn = sqlite3.connect("logs.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS requests_log (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, model_id TEXT, features_count INTEGER, predictions_sample TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password_hash TEXT)")
    cursor.execute("SELECT * FROM users WHERE username='admin'")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", ("admin", hash_password("superpass")))
    conn.commit()
    conn.close()

def verify_login(username, password):
    conn = sqlite3.connect("logs.db")
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE username=?", (username,))
    result = cursor.fetchone()
    conn.close()
    return result and result[0] == hash_password(password)

def log_request(model_id: str, features_count: int, preds: list):
    conn = sqlite3.connect("logs.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO requests_log (timestamp, model_id, features_count, predictions_sample) VALUES (?, ?, ?, ?)", 
                   (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model_id, features_count, json.dumps(preds[:5])))
    conn.commit()
    conn.close()

init_db()

# Инициализация сессии
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if "df" not in st.session_state: st.session_state.df = None
if "model_id" not in st.session_state: st.session_state.model_id = None
if "train_df" not in st.session_state: st.session_state.train_df = None
if "test_df" not in st.session_state: st.session_state.test_df = None
if "predictions" not in st.session_state: st.session_state.predictions = None
if "metrics" not in st.session_state: st.session_state.metrics = None
if "welcome_shown" not in st.session_state: st.session_state.welcome_shown = False 

# ==========================================
# 🔐 АВТОРИЗАЦИЯ И РЕГИСТРАЦИЯ
# ==========================================
if not st.session_state.authenticated:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.title("🔐 Доступ к системе")
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        auth_tab, reg_tab = st.tabs(["🔑 Вход", "📝 Регистрация"])
        
        with auth_tab:
            with st.form("login_form"):
                username = st.text_input("Имя пользователя", key="login_user")
                password = st.text_input("Пароль", type="password", key="login_pass")
                if st.form_submit_button("Войти", use_container_width=True):
                    with st.spinner("Проверка учетных данных..."):
                        time.sleep(0.5)
                        if verify_login(username, password):
                            st.session_state.authenticated = True
                            st.rerun()
                        else: 
                            st.error("Ошибка доступа: неверные данные")
                            
        with reg_tab:
            with st.form("register_form"):
                reg_username = st.text_input("Новое имя пользователя", key="reg_user")
                reg_password = st.text_input("Пароль", type="password", key="reg_pass")
                confirm_password = st.text_input("Подтвердите пароль", type="password", key="reg_confirm")
                if st.form_submit_button("Зарегистрироваться", use_container_width=True):
                    if not reg_username or not reg_password:
                        st.error("❌ Заполните все обязательные поля")
                    elif reg_password != confirm_password:
                        st.error("❌ Введенные пароли не совпадают")
                    else:
                        with st.spinner("Создание учетной записи..."):
                            try:
                                conn = sqlite3.connect("logs.db")
                                cursor = conn.cursor()
                                cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                                               (reg_username, hash_password(reg_password)))
                                conn.commit()
                                conn.close()
                                st.success("🎉 Регистрация успешна! Теперь вы можете войти во вкладке 'Вход'.")
                            except sqlite3.IntegrityError:
                                st.error("❌ Пользователь с таким именем пользователя уже зарегистрирован")
    st.stop()

# ==========================================
# 🎛️ ПУЛЬТ УПРАВЛЕНИЯ
# ==========================================
if st.session_state.authenticated and not st.session_state.welcome_shown:
    st.toast("✅ Авторизация успешна! Добро пожаловать в систему.", icon="👋")
    st.session_state.welcome_shown = True

with st.sidebar:
    st.header("Управление")
    if st.button("Выйти из системы"):
        st.session_state.authenticated = False
        st.session_state.welcome_shown = False 
        st.rerun()
    
    st.divider()
    st.subheader("1. Подготовка данных")
    uploaded_file = st.file_uploader("Загрузить CSV файл", type=["csv"], label_visibility="collapsed")
    if uploaded_file:
        with st.spinner("⏳ Обработка и загрузка датасета..."):
            df = pd.read_csv(uploaded_file)
            clip = st.slider("Отсечение аномалий (%)", 0.0, 5.0, 1.0)
            if clip > 0:
                num = df.select_dtypes(include=['number']).columns
                low, high = df[num].quantile(clip/100), df[num].quantile(1-clip/100)
                df = df[~((df[num] < low) | (df[num] > high)).any(axis=1)]
            st.session_state.df = df
            split = st.slider("Размер тестовой выборки (%)", 5, 50, 20)
            st.session_state.train_df, st.session_state.test_df = train_test_split(df, test_size=split/100, random_state=42)

    st.divider()
    st.subheader("2. Параметры API")
    api_url = st.text_input("Адрес сервера", value="http://5.129.248.80:8000")
    client = OilAPIClient(api_url)
    
    selected_model = st.selectbox("Алгоритм обучения:", ["random_forest", "linear_regression"])
    
    if st.button("Обучить модель"):
        if st.session_state.train_df is not None:
            with st.spinner(f"⚙️ Идет обучение алгоритма {selected_model}..."):
                clean = st.session_state.train_df.where(pd.notnull(st.session_state.train_df), None)
                try:
                    res = requests.post(f"{api_url.strip().rstrip('/')}/train", json={"model": selected_model, "data": clean.to_dict(orient="records")})
                    
                    if res.status_code == 200:
                        bytes_res = res.json()
                        try:
                            st.session_state.model_id = bytes_res.get("model_id")
                            st.success("Обучение завершено!")
                        except json.JSONDecodeError:
                            st.error("Сервер ответил статусом 200, но прислал сломанный ответ вместо ID.")
                    else:
                        st.error(f"Ошибка сервера {res.status_code}. Данная модель не поддерживается.")
                except Exception as e:
                    st.error(f"Сбой сети: {e}")

    if st.session_state.model_id:
        st.divider()
        st.subheader("3. Аналитика")
        if st.button("Рассчитать прогнозы"):
            with st.spinner("📊 Вычисление прогнозов на тестовой выборке..."):
                target = st.session_state.test_df.columns[-1]
                preds = client.predict(st.session_state.model_id, st.session_state.test_df.drop(columns=[target]).to_dict(orient="records"))
                if preds:
                    st.session_state.predictions = preds
                    raw_metrics = client.get_metrics(st.session_state.test_df[target].tolist(), preds)
                    st.session_state.metrics = {k.lower(): v for k, v in raw_metrics.items()}
                    log_request(st.session_state.model_id, len(st.session_state.test_df.columns)-1, preds)

# ==========================================
# 🖥️ ГЛАВНЫЙ ЭКРАН
# ==========================================
st.title("Платформа интеллектуального контроля качества")

if st.session_state.df is None:
    st.info("👋 **Добро пожаловать в систему!**\n\nДля начала работы загрузите технологические данные (CSV файл) в меню слева.")
else:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Всего записей", len(st.session_state.df))
    c2.metric("Кол-во датчиков", len(st.session_state.df.columns)-1)
    c3.metric("Строк обучения", len(st.session_state.train_df))
    c4.metric("Строк теста", len(st.session_state.test_df))
    
    with st.expander("📊 Разведочный анализ (Матрица корреляций)"):
        with st.spinner("Отрисовка корреляционной матрицы..."):
            fig_corr, ax_corr = plt.subplots(figsize=(8, 4)) 
            corr_df = st.session_state.df.rename(columns=COLUMN_NAMES_RU)
            sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr, annot_kws={"size": 8})
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.yticks(fontsize=9)
            st.pyplot(fig_corr, use_container_width=True)

    if st.session_state.predictions and st.session_state.metrics:
        # === ФИКС 6: Защита от краша при сдвиге ползунков ===
        if len(st.session_state.predictions) != len(st.session_state.test_df):
            st.warning("⚠️ Параметры датасета были изменены. Пожалуйста, пересчитайте прогнозы в меню слева.")
        else:
            m = st.session_state.metrics
            target = st.session_state.test_df.columns[-1]
            
            st.divider()
            st.subheader("📊 Оценка точности предсказаний")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("MAE (Ср. ошибка)", f"{m.get('mae', 0.0):.4f}")
            mc2.metric("RMSE (Кв. ошибка)", f"{m.get('rmse', 0.0):.4f}")
            mc3.metric("R² (Точность)", f"{m.get('r2', 0.0):.4f}")
            
            plot_df = st.session_state.test_df.copy().reset_index(drop=True)
            plot_df['Предсказание'] = st.session_state.predictions
            plot_df['Ошибка'] = (plot_df[target] - plot_df['Предсказание']).abs()
            
            g1, g2 = st.columns(2)
            with g1:
                st.write("**Интерактивная аппроксимация (Факт vs Прогноз)**")
                
                scatter = alt.Chart(plot_df).mark_circle(size=70, opacity=0.7).encode(
                    x=alt.X(f'{target}:Q', title='Fact', scale=alt.Scale(zero=False)),
                    y=alt.Y('Предсказание:Q', title='Predict', scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip(f'{target}:Q', title='Факт', format='.2f'), alt.Tooltip('Предсказание:Q', title='Прогноз', format='.2f'), alt.Tooltip('Ошибка:Q', title='Абс. ошибка', format='.2f')]
                ).interactive()
                
                min_val = min(plot_df[target].min(), plot_df['Предсказание'].min())
                max_val = max(plot_df[target].max(), plot_df['Предсказание'].max())
                ideal_df = pd.DataFrame({target: [min_val, max_val], 'Предсказание': [min_val, max_val]})
                
                ideal_line = alt.Chart(ideal_df).mark_line(color='red', strokeDash=[5, 5]).encode(
                    x=f'{target}:Q',
                    y='Предсказание:Q'
                )
                
                st.altair_chart(scatter + ideal_line, use_container_width=True)
                
            with g2:
                st.write("**Динамика показателя качества (Распределение)**")
                melted_df = plot_df[[target, 'Предсказание']].rename(columns={target: 'Факт'}).melt(var_name='Тип', value_name='Значение')
                
                hist = alt.Chart(melted_df).mark_bar(opacity=0.5).encode(
                    x=alt.X('Значение:Q', bin=alt.Bin(maxbins=30), title='Значение показателя'),
                    y=alt.Y('count()', title='Частота', stack=None),
                    color=alt.Color('Тип:N', scale=alt.Scale(domain=['Факт', 'Предсказание'], range=['green', 'blue']))
                ).interactive()
                st.altair_chart(hist, use_container_width=True)
                
            csv_data = plot_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Скачать результаты анализа (CSV)",
                data=csv_data,
                file_name="prognoz_kachestva.csv",
                mime="text/csv"
            )

    if st.session_state.model_id:
        st.divider()
        st.subheader("🎛️ Интерактивный калькулятор «Что-Если?»")
        st.write("Смоделируйте ситуацию: измените параметры датчиков вручную, чтобы узнать, каким будет качество.")
        
        feature_cols = st.session_state.df.columns[:-1]
        user_inputs = {}
        
        calc_cols = st.columns(3)
        for i, col_name in enumerate(feature_cols):
            label_ru = COLUMN_NAMES_RU.get(col_name, col_name)
            
            min_v = float(st.session_state.df[col_name].min())
            max_v = float(st.session_state.df[col_name].max())
            mean_v = float(st.session_state.df[col_name].mean())
            
            with calc_cols[i % 3]:
                user_inputs[col_name] = st.slider(label_ru, min_v, max_v, mean_v)
                
        if st.button("🔮 Получить мгновенный прогноз", type="primary"):
            with st.spinner("⏳ Выполняется расчет..."):
                single_pred = client.predict(st.session_state.model_id, [user_inputs])
                if single_pred:
                    st.success("Расчет выполнен!")
                    st.metric(label="Прогнозируемый индекс качества", value=f"{single_pred[0]:.4f}")
                    log_request(st.session_state.model_id, len(user_inputs), single_pred)

    with st.expander("📝 Системный журнал"):
        try:
            conn = sqlite3.connect("logs.db")
            logs_df = pd.read_sql_query("SELECT * FROM requests_log ORDER BY id DESC LIMIT 10", conn)
            st.dataframe(logs_df, use_container_width=True)
            conn.close()
        except Exception as e:
            st.info("Журнал пуст.")
