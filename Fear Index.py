import re, datetime, zoneinfo, requests, pandas as pd, yfinance as yf, streamlit as st
import FinanceDataReader as fdr
import numpy as np
from datetime import timedelta

KST=zoneinfo.ZoneInfo("Asia/Seoul")
ROOT="https://feargreedmeter.com"; PATH="/fear-and-greed-index"
UA={"User-Agent":"Mozilla/5.0"}

def fgi_label(v:int)->str:
    if v<=24:return "Extreme Fear"
    if v<=44:return "Fear"
    if v<=55:return "Neutral"
    if v<=75:return "Greed"
    return "Extreme Greed"

@st.cache_data(ttl=3600)
def _get_build_id()->str:
    r=requests.get(ROOT+PATH,headers=UA,timeout=10);r.raise_for_status()
    m=re.search(r'"buildId"\s*:\s*"([A-Za-z0-9\-\_]+)"',r.text)
    if not m:m=re.search(r'/_next/data/([A-Za-z0-9\-\_]+)/fear-and-greed-index\.json',r.text)
    if not m:raise RuntimeError("buildId not found")
    return m.group(1)

@st.cache_data(ttl=3600)
def fetch_fgi_history()->pd.DataFrame:
    try:
        url=f"{ROOT}/_next/data/{_get_build_id()}{PATH}.json"
        j=requests.get(url,headers=UA,timeout=10).json()
        rows=j["pageProps"]["data"]["fgiData"]["fgi"]
        out=[{"날짜":str(r["date"])[:10],"FGI":int(r["now"])} for r in rows if isinstance(r.get("now"),(int,float))]
        df=pd.DataFrame(out).sort_values("날짜")
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_daily_price(ticker: str):
    """일봉 종가 데이터 가져오기 (최신가, 전일 대비 변동률)"""
    try:
        df = yf.download(tickers=ticker, period="5d", interval="1d", progress=False, threads=False, auto_adjust=False)
        if df is None or df.empty:
            return None, None
        
        # Close 단일 Series 확보
        if isinstance(df.columns, pd.MultiIndex):
            if ("Close", ticker) in df.columns:
                s = df[("Close", ticker)].astype(float)
            else:
                s = df.xs("Close", axis=1, level=0).iloc[:, 0].astype(float)
        else:
            s = df["Close"].astype(float)
        
        s = s.dropna()
        if len(s) == 0:
            return None, None
            
        last = s.iloc[-1].item()
        prev = s.iloc[-2].item() if len(s) >= 2 else None
        chg = None if prev is None else (last/prev-1)*100
        return last, chg
    except Exception as e:
        return None, None

@st.cache_data(ttl=3600)
def fetch_last20_daily(ticker: str) -> pd.DataFrame:
    """최근 20일 일봉 데이터"""
    try:
        df = yf.download(tickers=ticker, period="1y", interval="1d", progress=False, threads=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Close 단일 Series 확보
        if isinstance(df.columns, pd.MultiIndex):
            if ("Close", ticker) in df.columns:
                s = df[("Close", ticker)].astype(float)
            else:
                s = df.xs("Close", axis=1, level=0).iloc[:, 0].astype(float)
        else:
            s = df["Close"].astype(float)

        s = s.dropna()
        if len(s) == 0:
            return pd.DataFrame()
            
        date_idx = pd.to_datetime(s.index.date)
        dod_pct = s.pct_change() * 100
        ath = s.cummax()
        mdd_pct = (s / ath - 1) * 100

        out = pd.DataFrame({
            "Date": date_idx,
            "Close": s.values,
            "DoD_%": dod_pct.values,
            "MDD_%": mdd_pct.values
        }, index=s.index)
        
        return out.tail(20).reset_index(drop=True)[["Date", "Close", "DoD_%", "MDD_%"]]
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_last20_vix() -> pd.DataFrame:
    """VIX 최근 20일 데이터"""
    try:
        df = yf.download(tickers="^VIX", period="2y", interval="1d", progress=False, threads=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Close 단일 Series 확보
        if isinstance(df.columns, pd.MultiIndex):
            s = df.xs("Close", axis=1, level=0).iloc[:, 0].astype(float)
        else:
            s = df["Close"].astype(float)

        s = s.dropna()
        if len(s) == 0:
            return pd.DataFrame()
            
        date_idx = pd.to_datetime(s.index.date)
        dod_pct = s.pct_change() * 100
        wow_pct = (s / s.shift(5) - 1) * 100

        out = pd.DataFrame({
            "Date": date_idx,
            "Close": s.values,
            "DoD": dod_pct.values,
            "WoW": wow_pct.values
        }, index=s.index)
        
        return out.tail(20).reset_index(drop=True)[["Date", "Close", "DoD", "WoW"]]
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_ma_daily(ticker: str, w5: int = 5, w20: int = 20):
    """이동평균 계산"""
    try:
        df = yf.download(tickers=ticker, period="6mo", interval="1d",
                         progress=False, threads=False, auto_adjust=False)
        if df is None or df.empty:
            return None, None
        
        # Close 단일 Series 확보
        if isinstance(df.columns, pd.MultiIndex):
            if ("Close", ticker) in df.columns:
                s = df[("Close", ticker)].astype(float)
            else:
                s = df.xs("Close", axis=1, level=0).iloc[:, 0].astype(float)
        else:
            s = df["Close"].astype(float)
            
        s = s.dropna()
        if len(s) == 0:
            return None, None
            
        ma5 = s.rolling(w5).mean().iloc[-1].item() if len(s) >= w5 else None
        ma20 = s.rolling(w20).mean().iloc[-1].item() if len(s) >= w20 else None
        return ma5, ma20
    except Exception as e:
        return None, None

@st.cache_data(ttl=3600)
def fetch_etf_data(ticker: str, n: int = 20) -> tuple[pd.DataFrame, str, float, str]:
    """ETF 데이터, ATH 날짜, 최근 1달 최저가 및 날짜 반환"""
    try:
        df = fdr.DataReader(ticker, start='2023-01-01')  # 1년 이상 데이터
        if df is None or df.empty:
            return pd.DataFrame(), None, None, None

        s = df['Close'].astype(float).dropna()
        if len(s) == 0:
            return pd.DataFrame(), None, None, None
            
        ath_value = s.max()
        
        # ATH 날짜 찾기 (전체 1년 데이터에서 가장 최근)
        ath_dates = s[s == ath_value]
        ath_date_str = pd.to_datetime(ath_dates.index[-1]).strftime('%m/%d') if not ath_dates.empty else None
        
        # 최근 1달(30일) 최저가 및 날짜 찾기
        recent_30d = s.tail(30)
        if len(recent_30d) > 0:
            low_1m_value = recent_30d.min()
            low_1m_dates = recent_30d[recent_30d == low_1m_value]
            low_1m_date_str = pd.to_datetime(low_1m_dates.index[-1]).strftime('%m/%d') if not low_1m_dates.empty else None
        else:
            low_1m_value = None
            low_1m_date_str = None
        
        # 최근 n일 데이터만 준비
        out = pd.DataFrame(index=s.index)
        out["Date"] = pd.to_datetime(s.index.date)
        out["Close"] = s.values
        out["DoD_%"] = s.pct_change() * 100
        out["ATH"] = s.cummax()
        out["MDD_%"] = (s / out["ATH"] - 1) * 100
        out = out.tail(n).reset_index(drop=True)
        
        return out[["Date", "Close", "DoD_%", "ATH", "MDD_%"]], ath_date_str, low_1m_value, low_1m_date_str
    except Exception as e:
        return pd.DataFrame(), None, None, None

def pct_str(x):
    return "—" if x is None or pd.isna(x) else f"{x:+.2f}%"

def num_str(x):
    return "—" if x is None or pd.isna(x) else f"{x:.2f}"

def color_span(val,is_pct=False,reverse=False):
    if val is None or pd.isna(val):return "<span>—</span>"
    c="green" if ((val>=0 and not reverse) or (val<0 and reverse)) else "red"
    s=f"{val:+.2f}%" if is_pct else f"{val:+.2f}"
    return f"<span style='color:{c}'>{s}</span>"

def span_pct(val):
    if val is None or pd.isna(val):return "<span>—</span>"
    if abs(val)<1e-12:return "<span>0.00%</span>"
    c="green" if val>0 else "red"
    return f"<span style='color:{c}'>{val:+.2f}%</span>"

def span_mdd(val):
    if val is None or pd.isna(val):return "<span>—</span>"
    if abs(val)<1e-12:return "<span>0.00%</span>"
    return f"<span style='color:red'>{val:.2f}%</span>" if val<0 else f"<span>{val:.2f}%</span>"

def render_table(title, columns, rows):
    html = f"<div><div class='card-title'>{title}</div><table style='width:100%;border-collapse:collapse;font-size:16px;border:1px solid #e5e7eb;table-layout:fixed'><thead><tr>"
    # 헤더는 무조건 가운데 정렬
    for col in columns:
        html += f"<th style='border-bottom:1px solid #e5e7eb;text-align:center;padding:6px 8px;width:{100/len(columns)}%'>{col}</th>"
    html += "</tr></thead><tbody>"
    for r in rows:
        html += "<tr>" + "".join(r) + "</tr>"
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

st.set_page_config(page_title="공포 지표 대시보드",layout="wide")
st.markdown(
    "<style>.grid{display:grid;grid-template-columns:1fr;gap:12px}"
    "@media (min-width: 768px) {.grid{grid-template-columns:repeat(3,1fr)}}"
    ".card{background:#eee;border:1px solid #e5e7eb;border-radius:16px;padding:10px 20px}"
    ".card-title{font-weight:600;font-size:20px;color:#444}"
    ".card-value{font-weight:700;font-size:32px}"
    ".desktop-inline{display:inline}"
    ".mobile-block{display:none}"
    ".badge-inline{display:inline}"
    ".badge-block{display:none}"
    "@media (max-width: 767px) {"
    ".desktop-inline{display:none}"
    ".mobile-block{display:block}"
    ".badge-inline{display:none}"
    ".badge-block{display:block;margin-top:2px}"
    "}</style>",
    unsafe_allow_html=True
)
st.markdown("<div style='font-weight:600;font-size:24px'>하락장 공포 지표 대시보드</div>",unsafe_allow_html=True)
st.caption(datetime.datetime.now(KST).strftime("기준: %y-%m-%d %H:%M:%S KST"))

# 데이터 가져오기
fgi_df = fetch_fgi_history()
fgi_now = int(fgi_df["FGI"].iloc[-1]) if not fgi_df.empty else None
fgi_label_now = fgi_label(fgi_now) if fgi_now is not None else "—"

# QQQ 데이터
qqq_now, qqq_chg = fetch_daily_price("QQQM")
qqq_df = fetch_last20_daily("QQQM")

# VIX 데이터
vix_now, vix_chg = fetch_daily_price("^VIX")
vix_df = fetch_last20_vix()

# QQQ 이동평균
ma5, ma20 = fetch_ma_daily("QQQM")

tab1, tab2, tab3 = st.tabs(["Fear", "Target", "AI전력"])

with tab1:
    # 3개 지표 카드
    indicators = [
        ("FGI", fgi_now, fgi_label_now),
        ("QQQM", qqq_now, None),
        ("VIX", vix_now, None)
    ]
    
    cols = None
    for i, (name, value, label) in enumerate(indicators):
        if i % 3 == 0:
            cols = st.columns(3)
        
        with cols[i % 3]:
            if name == "FGI":
                # FGI 카드 + 배지
                color_map = {
                    "Extreme Greed": ("#4CB43C", "#fff"),
                    "Greed": ("#AEB335", "#fff"),
                    "Neutral": ("#FDB737", "#fff"),
                    "Fear": ("#FF9E9E", "#fff"),
                    "Extreme Fear": ("#FF8A65", "#fff")
                }
                bg, fg = color_map.get(fgi_label_now, ("#EEE", "#444"))
                fgi_badge = f"<span style='background:{bg};color:{fg};padding:2px 10px;border-radius:8px;font-size:20px;margin-left:6px'>{fgi_label_now}</span>"
                
                html = (
                    f"<div class='card'><div class='card-title'>F&G Index</div>"
                    f"<div class='card-value'>{(fgi_now if fgi_now is not None else '—')}{fgi_badge}</div></div>"
                )
                st.markdown(html, unsafe_allow_html=True)
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

                # FGI 테이블
                if not fgi_df.empty:
                    fgi_last20 = fgi_df.tail(20).iloc[::-1].copy()
                    rows = []
                    for d, v in zip(fgi_last20["날짜"], fgi_last20["FGI"]):
                        label = fgi_label(int(v))
                        bg, fg = {
                            "Extreme Greed": ("#4CB43C", "#fff"),
                            "Greed":         ("#AEB335", "#fff"),
                            "Neutral":       ("#FDB737", "#fff"),
                            "Fear":          ("#FF9E9E", "#fff"),
                            "Extreme Fear":  ("#FF8A65", "#fff"),
                        }.get(label, ("#EEE", "#000"))
                        badge = f"<span style='background:{bg};color:{fg};padding:2px 8px;border-radius:999px;font-size:14px;font-weight:600'>{label}</span>"
                        rows.append([
                            f"<td style='padding:6px 8px;text-align:center'>{d}</td>",
                            f"<td style='padding:6px 8px;text-align:center'>{int(v)}</td>",
                            f"<td style='padding:6px 8px;text-align:center'>{badge}</td>"
                        ])
                    render_table("FGI", ["날짜", "FGI", "지표"], rows)
                
            elif name == "QQQM":
                # QQQ 카드 + MA 정보
                qqq_val = f"{qqq_now:.2f}" if qqq_now is not None else "—"
                qqq_ma_info = ""
                if ma5 is not None or ma20 is not None:
                    m5 = f"{ma5:.2f}" if ma5 is not None else "—"
                    m20 = f"{ma20:.2f}" if ma20 is not None else "—"
                    m5_dis = f"{((qqq_now/ma5)*100):.1f}" if (qqq_now is not None and ma5 not in [None,0]) else "—"
                    m20_dis = f"{((qqq_now/ma20)*100):.1f}" if (qqq_now is not None and ma20 not in [None,0]) else "—"
                    qqq_ma_info = f"""
                    <span class='desktop-inline' style='font-size:16px; font-weight:600; color:#666; margin-left:6px;'>5MA : {m5} ({m5_dis}) / 20MA : {m20} ({m20_dis})</span>
                    <div class='mobile-block' style='font-size:16px; font-weight:600; color:#666; margin-top:8px;'>
                        <div>5MA : {m5} ({m5_dis})</div>
                        <div>20MA : {m20} ({m20_dis})</div>
                    </div>"""

                html = (
                    f"<div class='card'><div class='card-title'>QQQM</div>"
                    f"<div class='card-value'>{qqq_val}{qqq_ma_info}</div></div>"
                )
                st.markdown(html, unsafe_allow_html=True)
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

                # QQQ 테이블
                if not qqq_df.empty:
                    qqq_reversed = qqq_df.iloc[::-1]

                    mdd_latest_zero_date = None
                    mdd_zero_rows = qqq_reversed[qqq_reversed["MDD_%"].abs() < 1e-12]
                    if not mdd_zero_rows.empty:
                        mdd_latest_zero_date = mdd_zero_rows["Date"].max()

                    rows = []
                    for _, r in qqq_reversed.iterrows():
                        if (mdd_latest_zero_date is not None) and (r["Date"] == mdd_latest_zero_date):
                            date = f"<td style='padding:6px 8px;text-align:center;background:#FFF3C4;border-radius:4px'>{r['Date'].strftime('%y-%m-%d')}</td>"
                        else:
                            date = f"<td style='padding:6px 8px;text-align:center'>{r['Date'].strftime('%y-%m-%d')}</td>"

                        price = f"<td style='padding:6px 8px;text-align:right'>{r['Close']:.2f}</td>"
                        dod_html = span_pct(r["DoD_%"])
                        mdd_html = span_mdd(r["MDD_%"])
                        rows.append([date, price,
                                    f"<td style='padding:6px 8px;text-align:right'>{dod_html}</td>",
                                    f"<td style='padding:6px 8px;text-align:right'>{mdd_html}</td>"])
                    render_table("QQQM", ["날짜","가격","전일대비","고점대비"], rows)
                
            elif name == "VIX":
                # VIX 카드
                vix_val = f"{vix_now:.2f}" if vix_now is not None else "—"
                html = (
                    f"<div class='card'><div class='card-title'>VIX</div>"
                    f"<div class='card-value'>{vix_val}</div></div>"
                )
                st.markdown(html, unsafe_allow_html=True)
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

                # VIX 테이블
                if not vix_df.empty:
                    vix_reversed = vix_df.iloc[::-1]
                    rows = []
                    for _, r in vix_reversed.iterrows():
                        date = r["Date"].strftime("%y-%m-%d")
                        price = f"{r['Close']:.2f}"
                        dod_html = (
                            f"<span style='color:{'green' if r['DoD']>0 else ('red' if r['DoD']<0 else 'inherit')}'>{r['DoD']:+.2f}%</span>"
                            if pd.notna(r["DoD"]) else "<span>—</span>"
                        )
                        wow_html = (
                            f"<span style='color:{'green' if r['WoW']>0 else ('red' if r['WoW']<0 else 'inherit')}'>{r['WoW']:+.2f}%</span>"
                            if pd.notna(r["WoW"]) else "<span>—</span>"
                        )
                        rows.append([
                            f"<td style='padding:6px 8px;text-align:center'>{date}</td>",
                            f"<td style='padding:6px 8px;text-align:right'>{price}</td>",
                            f"<td style='padding:6px 8px;text-align:right'>{dod_html}</td>",
                            f"<td style='padding:6px 8px;text-align:right'>{wow_html}</td>"
                        ])
                    render_table("VIX", ["날짜", "가격", "전일대비", "전주대비"], rows)

    st.caption("FGI: feargreedmeter.com · QQQ/VIX: Yahoo Finance(일봉 종가)")

with tab2:
    etfs=[
        ("379810","KODEX 미국나스닥100"),
        ("487230","KODEX 미국AI전력핵심인프라"),
        ("486450","SOL 미국AI전력인프라"),
    ]
    
    cols=None
    for i,(etf_ticker,etf_name) in enumerate(etfs):
        if i%3==0:
            cols=st.columns(3)

        with cols[i%3]:
            try:
                etf_df, ath_date_str, low_1m_value, low_1m_date_str = fetch_etf_data(etf_ticker, n=20)
            except Exception as e:
                etf_df = pd.DataFrame()
                ath_date_str = None
                low_1m_value = None
                low_1m_date_str = None
                
            # MDD에 따른 카드 배경색 결정
            card_bg = "#eee"  # 기본 배경색
            if not etf_df.empty:
                latest_mdd = float(etf_df.iloc[-1]["MDD_%"])
                
                if latest_mdd <= -5:
                    mdd_abs = abs(latest_mdd)
                    if mdd_abs < 10:
                        card_bg = "#F8B6AB"
                    elif mdd_abs < 15:
                        card_bg = "#E76E62"
                    else:
                        card_bg = "#DE5143"
                
            ath_info = ""
            if not etf_df.empty:
                # ATH 정보 처리
                ath_price = float(etf_df.iloc[-1]["ATH"])
                ath_str = f"ATH: {ath_price:,.0f}" + (f" ({ath_date_str})" if ath_date_str else "")
                
                # 최근 1달 최저가 정보 처리
                low_1m_str = ""
                if low_1m_value is not None:
                    low_1m_str = f" / 1M Low: {low_1m_value:,.0f}" + (f" ({low_1m_date_str})" if low_1m_date_str else "")
                
                ath_info = f"<span class='desktop-inline' style='font-size:14px; color:#666; margin-left:6px;'>{ath_str}{low_1m_str}</span><div class='mobile-block' style='font-size:14px; color:#666; margin-top:4px;'><div>{ath_str}</div><div>{low_1m_str.replace(' / ', '')}</div></div>"

            # 실제 전고점(ATH) 날짜 찾기 (테이블 음영처리용)
            ath_latest_date = None
            if not etf_df.empty:
                ath_price = float(etf_df.iloc[-1]["ATH"])
                ath_date_rows = etf_df[etf_df["Close"] == ath_price]
                if not ath_date_rows.empty:
                    ath_latest_date = ath_date_rows["Date"].max()

            # 카드 가격
            etf_now = float(etf_df["Close"].iloc[-1]) if not etf_df.empty else None
            html = (
                f"<div style='background:{card_bg};border:1px solid #e5e7eb;border-radius:16px;padding:10px 20px'>"
                f"<div class='card-title'>{etf_name}</div>"
                f"<div class='card-value'>{(f'{etf_now:,.0f}' if etf_now is not None else '—')}{ath_info}</div></div>"
            )
            st.markdown(html, unsafe_allow_html=True)
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            if not etf_df.empty:
                etf_reversed = etf_df.iloc[::-1]
                rows = []
                for _, r in etf_reversed.iterrows():
                    # 실제 ATH 날짜에 음영처리
                    if (ath_latest_date is not None) and (r["Date"] == ath_latest_date):
                        date = f"<td style='padding:6px 8px;text-align:center;background:#FFF3C4;border-radius:4px'>{r['Date'].strftime('%y-%m-%d')}</td>"
                    else:
                        date = f"<td style='padding:6px 8px;text-align:center'>{r['Date'].strftime('%y-%m-%d')}</td>"

                    price = f"<td style='padding:6px 8px;text-align:right'>{r['Close']:,.0f}</td>"
                    dod = f"<td style='padding:6px 8px;text-align:right'>{span_pct(r['DoD_%'])}</td>"
                    mdd = f"<td style='padding:6px 8px;text-align:right'>{span_mdd(r['MDD_%'])}</td>"
                    
                    rows.append([date, price, dod, mdd])
                render_table(f"{etf_ticker}", ["날짜","가격","전일대비","고점대비"], rows)
    
    st.caption("FinanceDataReader(일봉 종가)")

with tab3:
    st.markdown("<div style='font-weight:600;font-size:20px;margin-bottom:12px'>미국 주식 매매 트래킹</div>", unsafe_allow_html=True)
    
    # RSI 계산 함수
    def calculate_rsi(data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # 주식 데이터 가져오기
    @st.cache_data(ttl=300)
    def get_stock_data(ticker):
        try:
            stock = yf.Ticker(ticker)
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=120)
            
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            
            ninety_days_ago = pd.Timestamp(end_date - datetime.timedelta(days=90)).tz_localize('America/New_York')
            recent_data = hist[hist.index >= ninety_days_ago]
            ath_90d = recent_data['High'].max()
            
            drawdown = ((current_price - ath_90d) / ath_90d) * 100
            rsi = calculate_rsi(hist['Close'], 14).iloc[-1]
            
            return {
                'ticker': ticker,
                'current_price': current_price,
                'ath_90d': ath_90d,
                'drawdown': drawdown,
                'rsi': rsi
            }
        except Exception as e:
            return None
    
    # 물타기 기준 설정
    dca_rules = {
        'GEV': ('-10%', '-15%'),
        'CEG': ('-8%', '-12%'),
        'ANET': ('-12%', '-18%'),
        'ETN': ('-8%', '-12%'),
        'OKLO': ('-15%', '-25%'),
        'TT': ('-10%', '-15%'),
        'VST': ('-8%', '-12%'),
        'VRT': ('-10%', '-15%'),
        'PWR': ('-8%', '-12%'),
        'SMR': ('-20%', '-30%'),
        'CCJ': ('-10%', '-15%')
    }
    
    # 티커 목록
    tickers = ['GEV', 'CEG', 'ANET', 'ETN', 'OKLO', 'TT', 'VST', 'VRT', 'PWR', 'SMR', 'CCJ']
    
    # 데이터 수집
    stock_data = []
    for ticker in tickers:
        data = get_stock_data(ticker)
        if data:
            stock_data.append(data)
    
    if stock_data:
        df = pd.DataFrame(stock_data)
        
        # render_table 함수 사용
        rows = []
        for _, row in df.iterrows():
            ticker = row['ticker']
            dd = row['drawdown']
            
            # 하락률 폰트 색상 결정
            dd_color = "inherit"
            if dd <= -5:
                mdd_abs = abs(dd)
                if mdd_abs < 10:
                    dd_color = "#F8B6AB"
                elif mdd_abs < 15:
                    dd_color = "#E76E62"
                else:
                    dd_color = "#DE5143"
            
            # RSI 색상
            rsi = row['rsi']
            rsi_color = 'white'
            
            # 물타기 기준
            dca1, dca2 = dca_rules.get(ticker, ('-', '-'))
            
            # 물타기 1단계 충족 확인
            dca1_display = dca1
            if dca1 != '-':
                dca1_threshold = float(dca1.strip('%'))
                if dd <= dca1_threshold:
                    badge_inline = "<span class='badge-inline' style='background:#DE5143;color:#fff;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600;margin-left:4px'>충족</span>"
                    badge_block = "<span class='badge-block' style='background:#DE5143;color:#fff;padding:2px 6px;border-radius:999px;font-size:10px;font-weight:600'>충족</span>"
                    dca1_display = f"{dca1}{badge_inline}{badge_block}"
            
            # 물타기 2단계 충족 확인
            dca2_display = dca2
            if dca2 != '-':
                dca2_threshold = float(dca2.strip('%'))
                if dd <= dca2_threshold:
                    badge = "<span style='background:#DE5143;color:#fff;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600;margin-left:4px'>충족</span>"
                    dca2_display = f"{dca2}{badge}"
            
            rows.append([
                f"<td style='padding:6px 8px;text-align:center'><b>{ticker}</b></td>",
                f"<td style='padding:6px 8px;text-align:right'>{row['current_price']:.0f}</td>",
                f"<td style='padding:6px 8px;text-align:right'>{row['ath_90d']:.0f}</td>",
                f"<td style='padding:6px 8px;text-align:right;color:{dd_color};font-weight:500'>{dd:.1f}</td>",
                f"<td style='padding:6px 8px;text-align:center'>{dca1_display}</td>",
                # f"<td style='padding:6px 8px;text-align:center'>{dca2_display}</td>",
                f"<td style='padding:6px 8px;text-align:right;background-color:{rsi_color}'>{rsi:.0f}</td>",
            ])
        
        render_table("US Stocks", ["티커", "NOW", "ATH", "DD", "STEP1", "RSI"], rows)
    else:
        st.error("데이터를 불러올 수 없습니다.")
    
    st.caption("Yahoo Finance · 업데이트: 5분마다 캐시")