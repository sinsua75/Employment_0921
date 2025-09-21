import os
import io
import time
import datetime
from typing import Optional, Dict, Any
import json

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_lottie import st_lottie
import openpyxl  # Ensure this library is installed for Excel file handling

# ==============================================================================
# 0. CONFIGURATION & INITIAL SETUP
# ==============================================================================
st.set_page_config(
    page_title="기후 위기는 환경을 넘어 취업까지 흔든다",
    page_icon="🌍",
    layout="wide"
)

# --- App constants & Data URLs ---
TODAY = datetime.datetime.now().date()
CONFIG = {
    "nasa_gistemp_url": "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
    "worldbank_api_url": "https://api.worldbank.org/v2/country/all/indicator/SL.IND.EMPL.ZS",
    "noaa_co2_url": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt",
    "font_path": "/fonts/Pretendard-Bold.ttf", # Note: Local font path might need to be adjusted
}
MEMO_FILE = "memos.json"

# ==============================================================================
# 1. UTILITY & DATA LOADING FUNCTIONS
# ==============================================================================
def retry_get(url: str, params: Optional[Dict] = None, **kwargs: Any) -> Optional[requests.Response]:
    """Robust GET request with retries and user-agent."""
    final_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    for attempt in range(kwargs.get('max_retries', 2) + 1):
        try:
            resp = requests.get(url, params=params, headers=final_headers, timeout=kwargs.get('timeout', 15))
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            if attempt < kwargs.get('max_retries', 2):
                time.sleep(kwargs.get('backoff', 1.0) * (attempt + 1))
                continue
            st.sidebar.warning(f"API 요청 실패: {url.split('?')[0]} ({e})")
            return None

@st.cache_data(ttl=3600)
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize and preprocess a dataframe."""
    if df is None or df.empty: return pd.DataFrame()
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'], errors='coerce')
    d = d.dropna(subset=['date'])
    d = d[d['date'].dt.date <= TODAY]
    d['value'] = pd.to_numeric(d['value'], errors='coerce')
    subset_cols = ['date', 'group'] if 'group' in d.columns else ['date']
    d = d.drop_duplicates(subset=subset_cols)
    sort_cols = ['group', 'date'] if 'group' in d.columns else ['date']
    d = d.sort_values(sort_cols).reset_index(drop=True)
    if 'group' in d.columns:
        d['value'] = d.groupby('group')['value'].transform(lambda s: s.interpolate(method='linear', limit_direction='both', limit_area='inside'))
    else:
        d['value'] = d['value'].interpolate(method='linear', limit_direction='both', limit_area='inside')
    return d.dropna(subset=['value']).reset_index(drop=True)

def normalize_series(s: pd.Series) -> pd.Series:
    """Normalize a pandas Series to a 0-1 scale."""
    if s.max() == s.min(): return pd.Series(0.5, index=s.index)
    return (s - s.min()) / (s.max() - s.min())

@st.cache_data(ttl=3600)
def fetch_gistemp_csv() -> Optional[pd.DataFrame]:
    """Fetch and parse NASA GISTEMP global monthly anomalies."""
    resp = retry_get(CONFIG["nasa_gistemp_url"], max_retries=1)
    if resp is None: return None
    try:
        content = resp.content.decode('utf-8', errors='replace')
        lines = content.split('\n')
        data_start_index = next((i for i, line in enumerate(lines) if line.strip().startswith('Year,')), -1)
        if data_start_index == -1: return None
        df = pd.read_csv(io.StringIO("\n".join(lines[data_start_index:])))
        df.columns = [c.strip() for c in df.columns]
        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        present_months = [m for m in months if m in df.columns]
        df_long = df.melt(id_vars=['Year'], value_vars=present_months, var_name='Month', value_name='Anomaly')
        month_map = {name: num for num, name in enumerate(months, 1)}
        df_long['date'] = pd.to_datetime(df_long['Year'].astype(str) + '-' + df_long['Month'].map(month_map).astype(str), errors='coerce')
        df_final = df_long[['date']].copy()
        df_final['value'] = pd.to_numeric(df_long['Anomaly'], errors='coerce')
        df_final['group'] = '지구 평균 온도 이상치(℃)'
        return df_final.dropna(subset=['date', 'value'])
    except Exception as e:
        st.sidebar.error(f"GISTEMP 데이터 파싱 중 오류: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_noaa_co2_data() -> Optional[pd.DataFrame]:
    """Fetch and parse NOAA Mauna Loa CO2 data."""
    resp = retry_get(CONFIG["noaa_co2_url"], max_retries=1)
    if resp is None: return None
    try:
        content = resp.content.decode('utf-8')
        lines = [line for line in content.split('\n') if not line.strip().startswith('#')]
        df = pd.read_csv(io.StringIO('\n'.join(lines)), delim_whitespace=True, header=None,
                         names=['year', 'month', 'decimal_date', 'value', 'value_deseasonalized', 'num_days', 'stdev', 'unc'])
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df_final = df[['date', 'value']].copy()
        df_final['group'] = '대기 중 CO₂ 농도 (ppm)'
        return df_final[df_final['value'] > 0].reset_index(drop=True)
    except Exception as e:
        st.sidebar.error(f"NOAA CO2 데이터 파싱 중 오류: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_worldbank_employment() -> Optional[pd.DataFrame]:
    """Fetch World Bank API for Employment in industry, including ISO codes."""
    params = {'format': 'json', 'per_page': '20000'}
    resp = retry_get(CONFIG["worldbank_api_url"], params=params, max_retries=1)
    if resp is None: return None
    try:
        data = resp.json()
        if not isinstance(data, list) or len(data) < 2 or not data[1]: return None
        df = pd.json_normalize(data[1])
        df = df[['country.value', 'countryiso3code', 'date', 'value']]
        df.columns = ['group', 'iso_code', 'year', 'value']
        df['date'] = pd.to_datetime(df['year'] + '-01-01', errors='coerce')
        return df[['date', 'group', 'iso_code', 'value']].dropna()
    except Exception as e:
        st.sidebar.error(f"World Bank 데이터 처리 중 오류: {e}")
        return None

def get_sample_climate_data() -> pd.DataFrame:
    """Generate sample climate data as a fallback."""
    dates = pd.date_range(end=TODAY, periods=14*12, freq='MS')
    values = np.round(np.linspace(0.4, 1.2, len(dates)) + np.random.normal(0, 0.05, len(dates)), 3)
    return pd.DataFrame({'date': dates, 'value': values, 'group': '지구 평균 온도 이상치(℃)'})

def get_sample_co2_data() -> pd.DataFrame:
    """Generate sample CO2 data as a fallback."""
    dates = pd.date_range(end=TODAY, periods=14*12, freq='MS')
    values = np.round(np.linspace(380, 420, len(dates)) + np.random.normal(0, 0.5, len(dates)), 2)
    return pd.DataFrame({'date': dates, 'value': values, 'group': '대기 중 CO₂ 농도 (ppm)'})

def get_sample_employment_data() -> pd.DataFrame:
    """Generate sample employment data as a fallback."""
    years = pd.date_range(start=f"{TODAY.year-9}-01-01", end=f"{TODAY.year}-01-01", freq='AS')
    data = []
    countries = {'한국(예시)': 'KOR', 'OECD 평균(예시)': 'OED'}
    for country, code in countries.items():
        base_value = 24.0 if '한국' in country else 22.0
        for year in years:
            data.append({'date': year, 'group': country, 'iso_code': code, 'value': float(base_value + np.random.normal(0, 0.8))})
    return pd.DataFrame(data)

@st.cache_data
def get_sample_renewable_data() -> pd.DataFrame:
    """Generate sample renewable energy data as a fallback."""
    years = list(range(2010, 2024))
    data = {
        '연도': years,
        '태양광 (TWh)': [30 + i**2.1 for i in range(len(years))],
        '풍력 (TWh)': [80 + i**2.3 for i in range(len(years))],
        '수력 (TWh)': [3400 + i*30 for i in range(len(years))],
    }
    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars='연도', var_name='에너지원', value_name='발전량 (TWh)')
    return df_melted

@st.cache_data
def process_uploaded_unemployment_data(uploaded_file):
    """사용자가 업로드한 e-나라지표 엑셀 파일을 처리합니다."""
    if not uploaded_file:
        return pd.DataFrame()
    try:
        df = pd.read_excel(uploaded_file, skiprows=28, header=1)
        df = df.iloc[:, [0, 1, 3]].copy()
        df.columns = ["연도", "취업자 수 (만 명)", "실업률 (%)"]
        
        df = df.dropna(subset=["연도"])
        df = df[pd.to_numeric(df['연도'], errors='coerce').notna()]
        for col in df.columns:
            if col != '연도':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df["연도"] = df["연도"].astype(int)
        today = datetime.datetime.now()
        df = df[df["연도"] < today.year]
        return df
    except Exception as e:
        st.error(f"파일 처리 중 오류: {e}. 'openpyxl' 라이브러리가 설치되었는지 확인해주세요.")
        return pd.DataFrame()

@st.cache_data
def load_user_employment_data():
    """예시용 기후 산업 일자리 데이터를 생성합니다."""
    data = {"년도": [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
            "녹색산업 일자리": [10.5, 11.5, 13.0, 15.0, 17.0, 19.0, 21.5, 24.0],
            "화석연료 산업 일자리": [22.0, 21.0, 20.0, 18.0, 16.0, 14.0, 12.5, 11.0]}
    return pd.DataFrame(data)

@st.cache_data
def load_lottieurl(url: str):
    """Lottie 애니메이션 데이터 로딩 함수 (재시도 기능 포함)"""
    for _ in range(3):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                return r.json()
        except requests.exceptions.RequestException:
            time.sleep(1)
    return None

def load_memos():
    """memos.json 파일에서 모든 메모를 불러옵니다."""
    try:
        if not os.path.exists(MEMO_FILE):
            with open(MEMO_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
        with open(MEMO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_memos(memos):
    """memos.json 파일에 모든 메모를 저장합니다."""
    with open(MEMO_FILE, "w", encoding="utf-8") as f:
        json.dump(memos, f, ensure_ascii=False, indent=4)

# ==============================================================================
# 2. TAB CONTENT FUNCTIONS
# ==============================================================================
def run_survey_page():
    st.subheader("설문 ✍️")
    st.markdown("아래 **15문항** 설문에 답해주세요!")

    q1 = st.radio("1️⃣ 기후변화가 내 직업에 영향을 줄 것 같나요?", ["전혀 아니다", "조금 그렇다", "매우 그렇다"])
    q2 = st.selectbox("2️⃣ 가장 관심 있는 녹색 일자리 분야는?", ["신재생에너지", "ESG 컨설팅", "탄소 배출권", "기후 데이터 분석"])
    q3 = st.slider("3️⃣ 기후변화 대응 역량을 키우고 싶은 정도 (0~10)", 0, 10, 5)
    q4 = st.radio("4️⃣ 기후위기를 얼마나 심각하게 느끼시나요?", ["전혀 심각하지 않다", "보통이다", "매우 심각하다"])
    q5 = st.checkbox("5️⃣ 기후변화 대응 관련 교육을 받은 적이 있다")
    q6 = st.multiselect("6️⃣ 평소 실천하는 친환경 생활 습관을 선택해주세요", ["재활용", "대중교통 이용", "에너지 절약", "친환경 제품 구매", "채식 실천"])
    q7 = st.radio("7️⃣ 기후변화 대응에서 더 중요한 역할을 해야 할 주체는 누구라고 생각하시나요?", ["정부", "기업", "개인", "모두"])
    q8 = st.select_slider("8️⃣ 녹색 전환 과정에서 내 직업 안정성에 대한 우려 정도", options=["없음", "낮음", "보통", "높음", "매우 높음"])
    q9 = st.text_area("9️⃣ 녹색 일자리 확대를 위해 필요한 정책이나 제안이 있다면 적어주세요.")
    q10 = st.radio("🔟 기후변화 대응을 위한 세금(탄소세 등) 부과에 동의하시나요?", ["찬성", "반대", "잘 모르겠다"])
    q11 = st.slider("1️⃣1️⃣ 기업의 친환경 경영이 중요하다고 생각하는 정도 (0~10)", 0, 10, 7)
    q12 = st.radio("1️⃣2️⃣ 기후변화로 인한 직무 재교육이 필요하다고 생각하시나요?", ["필요 없다", "어느 정도 필요하다", "매우 필요하다"])
    q13 = st.multiselect("1️⃣3️⃣ 녹색 일자리 전환 시 가장 필요한 지원은?", ["재교육", "재정지원", "멘토링/상담", "일자리 매칭"])
    q14 = st.radio("1️⃣4️⃣ 해외보다 국내 기후정책이 더 중요하다고 생각하시나요?", ["국내 정책이 우선", "해외 협력이 더 중요", "둘 다 중요"])
    q15 = st.text_area("1️⃣5️⃣ 자유롭게 기후변화와 미래 일자리에 대한 의견을 남겨주세요.")

    if st.button("설문 제출"):
        st.success("✅ 설문이 제출되었습니다. 감사합니다!")

def run_intro_page():
    st.title("📝 기후 위기는 환경을 넘어 취업까지 흔든다")
    st.markdown("""
    ### 🌍 기후 위기와 취업, 더 이상 남의 이야기가 아닙니다

    기후변화는 이제 '환경운동가들의 이야기'나 '지구 차원의 막연한 위협'이 아닙니다. 세계 기상기구(WMO)에 따르면 지난 10년은 기온 상승 폭이 가장 큰 시기로 기록되었으며, 우리나라 역시 평균기온이 꾸준히 오르고 폭염·폭우·한파 같은 기후재난이 빈번해지고 있습니다. 중요한 것은 이러한 변화가 단순히 날씨에만 영향을 미치는 것이 아니라, 미래의 산업 구조와 청소년 세대의 진로, 그리고 **취업**까지 직접적으로 흔들고 있다는 점입니다.

    고용노동부가 발표한 자료에 따르면 **'녹색 일자리'**는 최근 5년간 꾸준히 증가하고 있으며, 정부는 2050 탄소중립 목표 달성을 위해 신재생에너지, 전기차, 탄소저감 기술 같은 분야에 수십만 개의 신규 일자리를 창출할 계획입니다. 반대로, 화석연료 중심의 전통 산업은 기후 규제 강화로 인해 일자리가 줄어들고 있습니다.

    결국, 기후 위기는 미래 사회의 취업 환경을 결정짓는 핵심 변수이며, 청소년 세대가 반드시 주목해야 할 중요한 문제임을 알 수 있습니다.
    """)

def run_main_analysis_page(climate_df, co2_df, employment_df, renewable_df):
    st.title("📈 숫자가 말하는 기후와 일자리 변화")
    st.markdown("---")

    st.markdown("""
    ### 📊 기후 위기, 실제 데이터로 증명됩니다
    기후 위기는 실제로 산업 구조와 취업률의 변화를 불러오고 있습니다.
    
    먼저 **환경 관련 산업 종사자 수**는 꾸준히 늘어나는 추세입니다. 고용노동부는 ‘2050 탄소중립 시나리오’를 발표하며 친환경 인프라 구축, 태양광·풍력 발전 확대, 수소 에너지 산업 육성을 통해 수십만 개의 새로운 녹색 일자리를 창출할 것이라고 밝혔습니다. 예를 들어, 태양광 패널 설치·관리, 전기차 배터리 개발·재활용, 탄소배출 감축을 위한 AI 기반 시스템 운영 등은 불과 10년 전만 해도 없던 새로운 직업군입니다.
    
    반대로 **전통 산업**은 위기에 직면해 있습니다. 화력발전과 석유화학 같은 산업은 탄소세와 환경 규제 부담으로 인해 점차 축소되고 있으며, 실제로 일부 석탄발전소는 조기 폐쇄가 결정되었습니다. 이 과정에서 해당 업종에 종사하던 근로자들은 실업이나 재교육의 필요에 직면하고 있습니다.
    
    아래 대시보드에서 실제로 기후 관련 데이터와 산업 데이터를 비교해 보세요.
    """)
    st.markdown("---")

    # 대시보드 코드 삽입
    st.header("📈 공식 공개 데이터 기반 분석")
    st.markdown("NASA (기온), NOAA (CO₂), World Bank (고용)의 공개 데이터를 분석합니다. API 호출 실패 시 예시 데이터로 자동 대체됩니다.")

    with st.container(border=True):
        st.subheader("📊 핵심 지표 요약", divider='rainbow')
        try:
            latest_climate = climate_df.sort_values('date', ascending=False).iloc[0]
            latest_co2 = co2_df.sort_values('date', ascending=False).iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric(f"최신 온도 이상치 ({latest_climate['date']:%Y-%m})", f"{latest_climate['value']:.2f} ℃")
            col2.metric(f"최신 CO₂ 농도 ({latest_co2['date']:%Y-%m})", f"{latest_co2['value']:.2f} ppm")
            col3.metric("고용 데이터 국가 수", f"{employment_df['group'].nunique()} 개")
        except (IndexError, ValueError):
            st.info("핵심 지표를 계산할 데이터가 부족합니다.")
    
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.subheader("🌡️ 지구 평균 온도 이상치")
            show_trendline = st.checkbox("5년 이동평균 추세선", value=True, key="trend_cb")
            if not climate_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=climate_df['date'], y=climate_df['value'], mode='lines', name='월별 이상치', line=dict(width=1, color='lightblue')))
                if show_trendline:
                    climate_df['trend'] = climate_df['value'].rolling(window=60, min_periods=12).mean()
                    fig.add_trace(go.Scatter(x=climate_df['date'], y=climate_df['trend'], mode='lines', name='5년 이동평균', line=dict(width=3, color='royalblue')))
                st.plotly_chart(fig, use_container_width=True)
                st.download_button("온도 데이터 다운로드", climate_df.to_csv(index=False, encoding='utf-8-sig'), "climate_data.csv", "text/csv", key="dl_climate")

    with c2:
        with st.container(border=True):
            st.subheader("💨 대기 중 CO₂ 농도")
            st.markdown("<p style='font-size: smaller;'>하와이 마우나로아 관측소 기준</p>", unsafe_allow_html=True)
            if not co2_df.empty:
                fig = px.line(co2_df, x='date', y='value', labels={'date': '날짜', 'value': 'CO₂ (ppm)'})
                st.plotly_chart(fig, use_container_width=True)
                st.download_button("CO₂ 데이터 다운로드", co2_df.to_csv(index=False, encoding='utf-8-sig'), "co2_data.csv", "text/csv", key="dl_co2")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("🏭 산업별 고용 비율 변화")
        if not employment_df.empty:
            employment_df['year'] = employment_df['date'].dt.year
            min_year = int(employment_df['year'].min())
            max_year = int(employment_df['year'].max())
            selected_year = st.slider("연도를 선택하여 지도를 변경하세요:", min_year, max_year, max_year)
            st.markdown(f"**{selected_year}년 기준 전 세계 산업 고용 비율 (Choropleth Map)**")
            map_df = employment_df[employment_df['year'] == selected_year]
            if not map_df.empty:
                fig_map = px.choropleth(map_df, locations="iso_code", color="value", hover_name="group", color_continuous_scale=px.colors.sequential.Plasma, labels={'value': '고용 비율 (%)'})
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.warning(f"{selected_year}년에는 표시할 고용 데이터가 없습니다.")

            st.markdown("**국가별 산업 고용 비율 추이 비교**")
            all_countries = sorted(employment_df['group'].unique())
            default_countries = [c for c in ['World', 'Korea, Rep.', 'China', 'United States', 'Germany'] if c in all_countries] or all_countries[:3]
            selected_countries = st.multiselect("비교할 국가를 선택하세요:", all_countries, default=default_countries)
            if selected_countries:
                comp_df = employment_df[employment_df['group'].isin(selected_countries)]
                fig_comp = px.line(comp_df, x='year', y='value', color='group', labels={'year':'연도', 'value':'산업 고용 비율(%)', 'group':'국가'})
                st.plotly_chart(fig_comp, use_container_width=True)
                st.download_button("선택 국가 고용 데이터 다운로드", comp_df.to_csv(index=False, encoding='utf-8-sig'), "employment_selected.csv", "text/csv", key="dl_emp")

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    이러한 데이터는 기후변화가 단순히 환경 문제가 아니라, 청년 세대의 일자리 지형도를 근본적으로 바꾸고 있음을 보여줍니다. 미래 진로를 고민하는 학생들에게 이는 중요한 인사이트가 될 것입니다.
    """)

def run_green_transition_page():
    st.title("🔎 녹색 전환: 위험과 기회의 두 얼굴")
    st.markdown("---")

    st.markdown("""
    ### 🌿 녹색 전환, 왜 취업과 직결되는가?
    핵심 원인은 바로 **'녹색 전환(Green Transition)'**입니다. 기후 대응을 위해 기업과 사회 전반이 친환경 기술을 도입하고, 이에 맞는 새로운 직무를 만들어내고 있기 때문입니다.
    
    예를 들어, 대기업들은 **ESG 경영(환경·사회·지배구조)**을 강화하면서 환경 관련 직무 채용을 확대하고 있습니다. 실제로 삼성·LG·현대차 같은 대기업은 탄소배출 감축을 위한 전담 부서를 운영하고 있으며, 에너지·환경 관련 전공자를 적극적으로 채용한다는 사실이 언론을 통해 보도된 바 있습니다. 이뿐만 아니라, 기후·환경 스타트업이 급격히 성장하면서 청년들의 새로운 진입 기회가 넓어지고 있습니다.
    
    그러나 기후 위기의 또 다른 얼굴은 **위험**입니다. 전통 산업의 축소와 일자리 감소는 청년 세대에게 직접적인 위협이 됩니다. 예를 들어, 석탄발전소 노동자들이 직장을 잃거나, 자동차 산업 내 내연기관 중심 부서가 축소되는 현상이 이미 나타나고 있습니다. 따라서 기후 위기는 청년들에게 '위험'과 '기회'를 동시에 던져주고 있으며, 변화에 어떻게 대응하고 준비하느냐가 취업 성패를 좌우하게 됩니다.
    
    이러한 점에서 기후 위기는 단순히 환경 운동 차원의 문제가 아니라, **청소년 진로교육의 핵심 주제**가 되어야 합니다. 학교와 사회가 함께 기후-취업의 연결성을 교육하지 않는다면, 학생들은 급격히 변화하는 산업 환경에 뒤처질 수밖에 없습니다.
    """)

def run_solution_page(unemployment_df, climate_raw, employment_sample_df):
    st.title("💡 청소년이 준비해야 할 미래 전략")
    st.markdown("---")
    
    st.markdown("""
    ### 🚀 기후 위기를 기회로 바꾸는 세 가지 제언
    지금까지 살펴본 바와 같이, 기후 위기는 지구 환경 문제를 넘어 우리의 진로와 취업 환경을 직접적으로 바꾸고 있습니다. 기후 위기에 대한 이해와 대응은 단순한 선택이 아니라, 미래 취업 경쟁력을 위한 필수 조건입니다.
    
    우리는 다음 세 가지 실천을 제안합니다.
    
    **제언 1: 기후 데이터 탐사대 – 미래 일자리 탐구하기**
    청소년 스스로 기후 데이터와 산업 통계를 찾아 분석하며, 변화하는 취업 환경을 직접 탐구한다.
    
    **제언 2: 그린 IT 프로젝트 – 전공과 기후 위기 연결하기**
    소프트웨어과 학생이라면 기후 데이터를 분석하는 프로그램을 만들어보거나, 에너지 절약을 위한 앱 아이디어를 기획할 수 있다.
    
    **제언 3: 청소년의 목소리 – 기후와 취업을 연결해 제안하기**
    기후 위기가 곧 청년 고용 창출과 직결된다는 사실을 어른들에게 알리고, 정책 제안 활동에 참여할 수 있다.
    """)
    st.markdown("---")

    # 해결방안 게임 섹션
    st.header("🚀 해결방안: 나의 미래 직업 만들기 (게임)")
    st.info("당신의 선택이 미래의 커리어와 환경에 어떤 영향을 미치는지 시뮬레이션 해보세요!")

    lottie_study = load_lottieurl("https://lottie.host/175b5a27-63f5-4220-8374-e32a13f789e9/5N7sBfSbB6.json")
    lottie_activity = load_lottieurl("https://lottie.host/97217a14-a957-41a4-9e1e-2879685a21e0/p3T5exs27n.json")
    lottie_career = load_lottieurl("https://lottie.host/7e05e830-7456-4c31-b844-93b5a1b55909/Rk4yQO6fS3.json")

    st.subheader("1️⃣ 당신의 선택은?")
    col1, col2, col3 = st.columns(3)
    with col1:
        if lottie_study: st_lottie(lottie_study, height=150, key="study")
        st.markdown("#### 학업 활동")
        edu_choice = st.radio("어떤 과목에 더 집중할까요?", ('탄소 배출량 분석 AI 모델링', '전통 내연기관 효율성 연구'), key="edu")
    with col2:
        if lottie_activity: st_lottie(lottie_activity, height=150, key="activity")
        st.markdown("#### 대외 활동")
        activity_choice = st.radio("어떤 동아리에 가입할까요?", ('신재생에너지 정책 토론 동아리', '고전 문학 비평 동아리'), key="activity")
    with col3:
        if lottie_career: st_lottie(lottie_career, height=150, key="career")
        st.markdown("#### 진로 탐색")
        career_choice = st.radio("어떤 기업의 인턴십에 지원할까요?", ('에너지 IT 스타트업', '안정적인 정유회사'), key="career")

    base_score = 50; base_co2 = 0; skills = []
    if edu_choice == '탄소 배출량 분석 AI 모델링':
        score_edu = 25; co2_edu = -15; skills.extend(["AI/머신러닝", "데이터 분석"])
    else:
        score_edu = 5; co2_edu = 5; skills.append("기계 공학")
    if activity_choice == '신재생에너지 정책 토론 동아리':
        score_activity = 15; co2_activity = -10; skills.extend(["정책 이해", "토론 및 설득"])
    else:
        score_activity = 5; co2_activity = 0; skills.append("인문학적 소양")
    if career_choice == '에너지 IT 스타트업':
        score_career = 20; co2_career = -10; skills.extend(["실무 경험", "문제 해결 능력"])
    else:
        score_career = 10; co2_career = 5; skills.append("대기업 프로세스 이해")
    final_score = round(base_score + score_edu + score_activity + score_career)
    final_co2 = round(base_co2 + co2_edu + co2_activity + co2_career)
    final_skills = list(set(skills))

    st.subheader("2️⃣ 10년 후, 당신의 모습은?")
    st.markdown(f"""
    <div style="background-color: #F0F2F6; border-radius: 10px; padding: 20px; display: flex; justify-content: space-around; align-items: center; color: black;">
        <div style="text-align: center;">
            <span style="font-size: 1.2em;">🎓 미래 커리어 경쟁력</span><br>
            <span style="font-size: 2.5em; font-weight: bold;">{final_score}</span><span style="font-size: 1.5em;"> 점</span>
        </div>
        <div style="text-align: center;">
            <span style="font-size: 1.2em;">🌱 환경 기여도 (CO₂)</span><br>
            <span style="font-size: 2.5em; font-weight: bold;">{-final_co2}</span><span style="font-size: 1.5em;"> 감축</span>
        </div>
        <div style="text-align: center;">
            <span style="font-size: 1.2em;">🔑 획득한 핵심 역량</span><br>
            <span style="font-size: 1.2em;">{', '.join(final_skills) if final_skills else '선택 대기중...'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if final_score >= 100:
        st.success("🎉 **완벽한 미래 인재!** 당신은 기후 위기를 기회로 만드는 시대의 리더가 될 것입니다.", icon="🚀")
    elif final_score >= 75:
        st.info("👍 **유망한 인재!** 녹색 전환 시대에 성공적으로 적응할 수 있는 뛰어난 잠재력을 갖췄습니다.", icon="🌟")
    else:
        st.warning("🤔 **성장 가능성!** 변화하는 산업 트렌드에 조금 더 관심을 가진다면 당신의 미래는 더욱 밝아질 거예요.", icon="💡")

    st.markdown("---")
    
    # 나의 실천 다짐 남기기 (공유 방명록)
    st.header("✍️ 나의 실천 다짐 남기기 (공유 방명록)")
    st.write("여러분의 다짐은 이 웹사이트에 영구적으로 저장되어 모든 방문자에게 공유됩니다!")
    
    cols = st.columns([0.7, 0.3])
    with cols[0]:
        name = st.text_input("닉네임", placeholder="자신을 표현하는 멋진 닉네임을 적어주세요!", key="memo_name")
        memo = st.text_area("실천 다짐", placeholder="예) 텀블러 사용하기, 가까운 거리는 걸어다니기 등", key="memo_text")
    with cols[1]:
        color = st.color_picker("메모지 색상 선택", "#FFFACD", key="memo_color")
        if st.button("다짐 남기기!", use_container_width=True):
            if name and memo:
                all_memos = load_memos()
                all_memos.insert(0, {"name": name, "memo": memo, "color": color, "timestamp": str(datetime.datetime.now())})
                save_memos(all_memos)
                st.balloons()
                st.success("소중한 다짐이 모두에게 공유되었습니다!")
                st.rerun()
            else:
                st.warning("닉네임과 다짐을 모두 입력해주세요!")
    
    st.divider()
    st.subheader("💬 우리의 다짐들")
    memos_list = load_memos()
    
    if not memos_list:
        st.info("아직 작성된 다짐이 없어요. 첫 번째 다짐을 남겨주세요!")
    else:
        memo_cols = st.columns(3)
        for i, m in enumerate(memos_list):
            with memo_cols[i % 3]:
                st.markdown(f"""
                <div style="background-color:{m.get('color', '#FFFACD')}; border-left: 5px solid #FF6347; border-radius: 8px; padding: 15px; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); height: 150px;">
                    <p style="font-size: 1.1em; color: black; margin-bottom: 10px;">"{m.get('memo', '')}"</p>
                    <strong style="font-size: 0.9em; color: #555;">- {m.get('name', '')} -</strong>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("🌐 참고 자료")
    st.markdown("""
    * 대학진학률 및 취업률 그래프, 여성가족부, [https://www.ypec.re.kr/mps/youthStat/education/collegeEmployRate?menuId=MENU00757](https://www.ypec.re.kr/mps/youthStat/education/collegeEmployRate?menuId=MENU00757)
    * 기후변화 4대지표, 탄소중립 정책포털, [https://www.gihoo.or.kr/statistics.es?mid=a30401000000](https://www.gihoo.or.kr/statistics.es?mid=a30401000000)
    * 향후 10년 사라질 직업 1위는?, 포켓뉴스 다음채널, [https://v.daum.net/v/4z6QWe3IKx](https://v.daum.net/v/4z6QWe3IKx)
    * 주요 업종 일자리 그래프, 고용노동부, [https://www.moel.go.kr/news/enews/report/enewsView.do?news_seq=17516](https://www.moel.go.kr/news/enews/report/enewsView.do?news_seq=17516)
    * **추가 출처**: NASA GISTEMP, NOAA GML, World Bank
    """)

# ==============================================================================
# 3. MAIN APPLICATION LOGIC
# ==============================================================================
def main():
    """Main function to run the Streamlit app."""
    # --- Data Loading and Session State Management ---
    if 'data_loaded' not in st.session_state:
        # **[수정]** fetch_gistemp_csv()가 빈 DataFrame을 반환할 경우를 명시적으로 처리
        climate_data = fetch_gistemp_csv()
        if climate_data.empty:
            st.session_state.climate_df = preprocess_dataframe(get_sample_climate_data())
        else:
            st.session_state.climate_df = preprocess_dataframe(climate_data)
            
        # **[수정]** co2 데이터 처리
        co2_data = fetch_noaa_co2_data()
        if co2_data.empty:
            st.session_state.co2_df = preprocess_dataframe(get_sample_co2_data())
        else:
            st.session_state.co2_df = preprocess_dataframe(co2_data)

        # **[수정]** 고용 데이터 처리
        employment_data = fetch_worldbank_employment()
        if employment_data.empty:
            st.session_state.employment_df = preprocess_dataframe(get_sample_employment_data())
        else:
            st.session_state.employment_df = preprocess_dataframe(employment_data)
            
        st.session_state.renewable_df = get_sample_renewable_data()
        st.session_state.uploaded_unemployment_df = pd.DataFrame()
        st.session_state.data_loaded = True

    # --- Sidebar for Navigation and Upload ---
    with st.sidebar:
        st.header("📊 옵션 설정")
        uploaded_file = st.file_uploader("취업률 엑셀 파일 업로드", type=["xlsx", "xls"])
        if uploaded_file:
            st.session_state.uploaded_unemployment_df = process_uploaded_unemployment_data(uploaded_file)
        
        all_years = pd.concat([
            st.session_state.uploaded_unemployment_df['연도'] if not st.session_state.uploaded_unemployment_df.empty else pd.Series(dtype='int'),
            pd.Series(st.session_state.climate_df['date'].dt.year.unique(), dtype='int')
        ]).dropna().unique()

        min_year, max_year = (int(all_years.min()), int(all_years.max())) if len(all_years) > 0 else (2017, datetime.datetime.now().year - 1)
        year_range = st.slider("표시할 연도 범위", min_year, max_year, (min_year, max_year))
        
        st.header("📖 목차")
        st.markdown("""
        - [📝 서론](#-기후-위기와-취업-더-이상-남의-이야기가-아닙니다)
        - [📈 본론 1](#-기후-위기-실제-데이터로-증명됩니다)
        - [🔎 본론 2](#-녹색-전환-왜-취업과-직결되는가)
        - [💡 결론 및 제언](#-기후-위기를-기회로-바꾸는-세-가지-제언)
        - [🚀 게임](#-해결방안-나의-미래-직업-만들기-게임)
        - [✍️ 방명록](#-나의-실천-다짐-남기기-공유-방명록)
        """)

    # --- Main Page Content ---
    st.markdown("<h1 style='text-align: center;'>🌍 기후 위기는 환경을 넘어 취업까지 흔든다</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2em; font-weight: bold;'>1403 권초현, 1405 김동현, 1410 신수아, 1416 조정모</p>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 설문조사", "📝 서론", "📈 본론 1", "🔎 본론 2", "💡 결론 및 제언"])

    with tab1:
        run_survey_page()

    with tab2:
        run_intro_page()

    with tab3:
        run_main_analysis_page(st.session_state.climate_df, st.session_state.co2_df, st.session_state.employment_df, st.session_state.renewable_df)

    with tab4:
        run_green_transition_page()

    with tab5:
        run_solution_page(st.session_state.uploaded_unemployment_df, st.session_state.climate_df, load_user_employment_data())

if __name__ == "__main__":
    main()