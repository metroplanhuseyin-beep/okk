import streamlit as st
import pandas as pd
from datetime import date, time
from prediction import MatchPredictor
from feature_engineering import FeatureEngineering

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Futbol Maç Tahmin Sistemi",
    page_icon="⚽",
    layout="centered"
)

# Özel CSS Stilleri
st.markdown("""
    <style>
    .main {
        background-color: #1e272e;
        color: #ecf0f1;
    }
    .stApp {
        background-color: #1e272e;
    }
    h1, h2, h3, h4, h5, h6, label, p {
        color: #ecf0f1 !important;
        font-family: Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        border: none;
        border-radius: 3px;
        padding: 10px;
        font-family: Verdana, sans-serif;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #27ae60;
    }
    .stSelectbox, .stDateInput, .stTimeInput {
        background-color: #2c3e50;
        color: #ecf0f1;
    }
    div[data-baseweb="select"] > div {
        background-color: #2c3e50;
        color: #ecf0f1;
    }
    </style>
""", unsafe_allow_html=True)

# Sınıfları başlat
@st.cache_resource
def init_services():
    fe = FeatureEngineering()
    predictor = MatchPredictor()
    return fe, predictor

fe, predictor = init_services()

st.title("⚽ Futbol Maç Tahmin Sistemi")
st.markdown("---")

# --- VERİ YÜKLEME FONKSİYONLARI ---
@st.cache_data
def load_leagues_data():
    try:
        leagues = pd.read_sql_query("""
            SELECT DISTINCT l.league_key, l.league_name, c.country_name
            FROM leagues l
            JOIN countries c ON l.country_key = c.country_key
            ORDER BY c.country_name, l.league_name
        """, fe.conn)
        return leagues
    except Exception as e:
        return None

leagues_df = load_leagues_data()

if leagues_df is None or leagues_df.empty:
    st.error("Ligler yüklenirken hata oluştu veya veri tabanı boş!")
    st.stop()

# Lig Seçimi Sözlüğü Oluşturma
league_options = {
    f"{row['league_name']} ({row['country_name']})": row['league_key']
    for _, row in leagues_df.iterrows()
}

# --- FORM BİLEŞENLERİ (Maç Bilgileri) ---
st.subheader("📋 Maç Bilgileri")

selected_league_name = st.selectbox("Lig:", list(league_options.keys()))
league_key = league_options[selected_league_name]

# Seçilen lige ait takımları çek
@st.cache_data
def load_teams_data(l_key):
    query = """
    SELECT team_key, team_name
    FROM teams
    WHERE league_key = ?
    ORDER BY team_name
    """
    return pd.read_sql_query(query, fe.conn, params=[l_key])

teams_df = load_teams_data(league_key)

team_options = {
    row['team_name']: row['team_key']
    for _, row in teams_df.iterrows()
}

col1, col2 = st.columns(2)
with col1:
    home_team_name = st.selectbox("Ev Sahibi:", list(team_options.keys()) if team_options else ["Takım Bulunamadı"])
with col2:
    away_default_idx = 1 if len(team_options) > 1 else 0
    away_team_name = st.selectbox("Deplasman:", list(team_options.keys()) if team_options else ["Takım Bulunamadı"], index=away_default_idx)

home_team_key = team_options.get(home_team_name)
away_team_key = team_options.get(away_team_name)

col3, col4, col5 = st.columns(3)
with col3:
    match_date = st.date_input("Tarih", value=date.today())
with col4:
    match_time = st.time_input("Saat", value=time(20, 0))
with col5:
    seasons = ['2024/2025', '2023/2024', '2022/2023', '2021/2022', '2020/2021', '2019/2020']
    season = st.selectbox("Sezon:", seasons)

st.markdown("---")

# --- YARDIMCI GÖSTERİM FONKSİYONLARI ---
def get_top_predictions(probabilities, top_n=15):
    """En yüksek olasılıklı tahminleri döndür"""
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    main_pred = sorted_probs[0][0]
    other_preds = [(pred, prob) for pred, prob in sorted_probs[1:] if pred != main_pred]
    return other_preds[:min(len(other_preds), top_n-1)]

def display_prediction_group(title, predictions):
    st.markdown(f"### {title}")
    for key, value in predictions.items():
        text = ""
        if isinstance(value, dict):
            if 'Tahmin' in value:
                if key in ['Skor', 'Gol Aralığı'] and 'Tüm Olasılıklar' in value:
                    text_parts = [f"**{value['Tahmin']}** ({value['Olasılık']})"]
                    other_preds = get_top_predictions(value['Tüm Olasılıklar'])
                    text_parts.extend([f"{pred} (%{prob*100:.1f})" for pred, prob in other_preds])
                    text = ", ".join(text_parts)
                elif key == 'Gol Farkı' and 'Olasılıklar' in value:
                    text_parts = [f"**{value['Tahmin']}** ({value['Olasılık']})"]
                    tahmin = value['Tahmin']
                    olasılıklar = {k: v for k, v in value['Olasılıklar'].items() if k != tahmin}
                    text_parts.extend([f"{k} ({v})" for k, v in olasılıklar.items()])
                    text = ", ".join(text_parts)
                elif 'Olasılıklar' in value:
                    tahmin = value['Tahmin']
                    text_parts = [f"**{tahmin}** ({value['Olasılık']})"]
                    olasılıklar = {k: v for k, v in value['Olasılıklar'].items() if k != tahmin}
                    text_parts.extend([f"{k} ({v})" for k, v in olasılıklar.items()])
                    text = ", ".join(text_parts)
                else:
                    text = f"**{value['Tahmin']}** ({value['Olasılık']})"
        else:
            text = str(value)
        
        st.markdown(f"- **{key}:** {text}")
    st.markdown("")

# --- TAHMİN BUTONU VE İŞLEMİ ---
if st.button("🔮 Tahmin Yap"):
    if home_team_key == away_team_key:
        st.warning("Ev sahibi ve deplasman takımı aynı olamaz!")
    else:
        try:
            with st.spinner("Modeller yükleniyor ve maç tahmin ediliyor..."):
                if 'current_league' not in st.session_state or st.session_state.current_league != league_key:
                    predictor.load_models(league_key)
                    st.session_state.current_league = league_key

                formatted_date = match_date.strftime("%Y-%m-%d")
                formatted_time = match_time.strftime("%H:%M")

                predictions = predictor.predict_match(
                    league_key=league_key,
                    match_date=formatted_date,
                    match_time=formatted_time,
                    home_team_key=home_team_key,
                    away_team_key=away_team_key,
                    season=season,
                )

            if predictions:
                st.success("Tahminler başarıyla oluşturuldu!")
                display_prediction_group("⏱️ İlk Yarı Tahminleri", predictions['İlk Yarı Tahminleri'])
                st.markdown("---")
                display_prediction_group("⏱️ İkinci Yarı Tahminleri", predictions['İkinci Yarı Tahminleri'])
                st.markdown("---")
                display_prediction_group("🏁 Maç Sonu Tahminleri", predictions['Maç Sonu Tahminleri'])
            else:
                st.error("Tahmin yapılamadı! Lütfen girdi verilerini kontrol edin.")
                
        except Exception as e:
            st.error(f"Hata oluştu: {str(e)}")
