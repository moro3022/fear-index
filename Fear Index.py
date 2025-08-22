# app_tables.py
# pip install streamlit requests yfinance pandas
import re, datetime, zoneinfo, requests, pandas as pd, yfinance as yf, streamlit as st

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
    url=f"{ROOT}/_next/data/{_get_build_id()}{PATH}.json"
    j=requests.get(url,headers=UA,timeout=10).json()
    rows=j["pageProps"]["data"]["fgiData"]["fgi"]
    out=[{"날짜":str(r["date"])[:10],"FGI":int(r["now"])} for r in rows if isinstance(r.get("now"),(int,float))]
    df=pd.DataFrame(out).sort_values("날짜")
    return df

@st.cache_data(ttl=60)
def fetch_intraday_price(ticker:str):
    df=yf.download(tickers=ticker,period="1d",interval="1m",progress=False,threads=False,auto_adjust=False)
    if not df.empty:
        close_series = df["Close"].dropna()
        last = close_series.iloc[-1].item()  # float() 제거하고 .item() 사용
        prev = close_series.iloc[-2].item() if len(close_series)>=2 else None
        chg=None if prev is None else (last/prev-1)*100
        return last, chg
    df=yf.download(tickers=ticker,period="5d",interval="1d",progress=False,threads=False,auto_adjust=False)
    if df.empty:return None,None
    last=float(df["Close"].iloc[-1]); prev=float(df["Close"].iloc[-2]) if len(df)>=2 else None
    return last,(None if prev is None else (last/prev-1)*100)

@st.cache_data(ttl=3600)
def fetch_last10_daily(ticker: str) -> pd.DataFrame:
    df = yf.download(tickers=ticker, period="10y", interval="1d", progress=False, threads=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # 멀티인덱스/멀티컬럼 방지: Close 단일 Series 확보
    if isinstance(df.columns, pd.MultiIndex):
        # ('Close', 'QQQ') 같은 컬럼 찾기
        if ("Close", ticker) in df.columns:
            s = df[("Close", ticker)].astype(float)
        else:
            s = df.xs("Close", axis=1, level=0).iloc[:, 0].astype(float)
    else:
        # 단일컬럼 프레임에서 Close만
        s = df["Close"].astype(float)

    s = s.dropna()
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
    last10 = out.tail(20).reset_index(drop=True)
    return last10[["Date", "Close", "DoD_%", "MDD_%"]]

@st.cache_data(ttl=3600)
def fetch_last10_vix() -> pd.DataFrame:
    df = yf.download(tickers="^VIX", period="2y", interval="1d", progress=False, threads=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # Close 단일 Series 확보
    if isinstance(df.columns, pd.MultiIndex):
        s = df.xs("Close", axis=1, level=0).iloc[:, 0].astype(float)
    else:
        s = df["Close"].astype(float)

    s = s.dropna()
    date_idx = pd.to_datetime(s.index.date)
    dod = s.diff()
    wow = s.diff(5)

    out = pd.DataFrame({
        "Date": date_idx,
        "Close": s.values,
        "DoD": dod.values,
        "WoW": wow.values
    }, index=s.index)
    last10 = out.tail(20).reset_index(drop=True)
    return last10[["Date", "Close", "DoD", "WoW"]]

@st.cache_data(ttl=600)
def fetch_ma_daily(ticker: str, w5: int = 5, w20: int = 20):
    df = yf.download(tickers=ticker, period="6mo", interval="1d",
                     progress=False, threads=False, auto_adjust=False)
    if df is None or df.empty:
        return None, None
    # Close 단일 Series 확보
    if isinstance(df.columns, pd.MultiIndex):
        s = (df[("Close", ticker)] if ("Close", ticker) in df.columns
             else df.xs("Close", axis=1, level=0).iloc[:, 0]).astype(float)
    else:
        s = df["Close"].astype(float)
    s = s.dropna()
    ma5  = float(s.rolling(w5).mean().iloc[-1])  if len(s) >= w5  else None
    ma20 = float(s.rolling(w20).mean().iloc[-1]) if len(s) >= w20 else None
    return ma5, ma20

@st.cache_data(ttl=3600)
def fetch_last_daily_generic(ticker: str, n: int = 20) -> pd.DataFrame:
    import pandas as pd, yfinance as yf
    df = yf.download(tickers=ticker, period="1y", interval="1d", progress=False, threads=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # Close 단일 시리즈 확보
    if isinstance(df.columns, pd.MultiIndex):
        s = df.xs("Close", axis=1, level=0).iloc[:, 0].astype(float)
    else:
        s = df["Close"].astype(float)

    s = s.dropna()
    out = pd.DataFrame(index=s.index)
    out["Date"]  = pd.to_datetime(s.index.date)
    out["Close"] = s.values
    out["DoD_%"] = s.pct_change() * 100
    out["ATH"]   = s.cummax()
    out["MDD_%"] = (s / out["ATH"] - 1) * 100
    out = out.tail(n).reset_index(drop=True)
    return out[["Date", "Close", "DoD_%", "MDD_%"]]

@st.cache_data(ttl=900)
def fetch_last_close(ticker: str):
    # 한국 ETF는 분봉이 빈약할 수 있어 일봉 종가 사용
    import yfinance as yf
    df = yf.download(tickers=ticker, period="5d", interval="1d", progress=False, threads=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        s = df.xs("Close", axis=1, level=0).iloc[:, 0].astype(float)
    else:
        s = df["Close"].astype(float)
    return float(s.iloc[-1]) if len(s) else None

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
    html = f"<div><div class='card-title'>{title}</div><table style='width:100%;border-collapse:collapse;font-size:16px;border:1px solid #e5e7eb'><thead><tr>"
    # 헤더는 무조건 가운데 정렬
    for col in columns:
        html += f"<th style='border-bottom:1px solid #e5e7eb;text-align:center;padding:6px 8px'>{col}</th>"
    html += "</tr></thead><tbody>"
    for r in rows:
        html += "<tr>" + "".join(r) + "</tr>"
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

st.set_page_config(page_title="공포 지표 대시보드",layout="wide")
st.markdown(
    "<style>.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}"
    ".card{background:#eee;border:1px solid #e5e7eb;border-radius:16px;padding:10px 20px}"
    ".card-title{font-weight:600;font-size:20px;color:#444}"
    ".card-value{font-weight:700;font-size:32px}</style>",
    unsafe_allow_html=True
)
st.markdown("<div style='font-weight:600;font-size:24px'>하락장 공포 지표 대시보드</div>",unsafe_allow_html=True)
st.caption(datetime.datetime.now(KST).strftime("기준: %Y-%m-%d %H:%M:%S KST"))

fgi_df=fetch_fgi_history()
fgi_now=int(fgi_df["FGI"].iloc[-1]) if not fgi_df.empty else None
fgi_label_now=fgi_label(fgi_now) if fgi_now is not None else "—"
qqq_now,qqq_chg=fetch_intraday_price("QQQ")
vix_now,vix_chg=fetch_intraday_price("^VIX")
qqq_now, qqq_chg = fetch_intraday_price("QQQ")
ma5, ma20 = fetch_ma_daily("QQQ")

tab1, tab2 = st.tabs(["Fear", "Target"])

with tab1:
    def render_top_cards(fgi_now, fgi_label_now, qqq_now, qqq_chg, vix_now, vix_chg, qqq_ma5=None, qqq_ma20=None):
        def badge_html(text):
            color_map = {
                "Extreme Greed": ("#4CB43C", "#fff"),
                "Greed": ("#AEB335", "#fff"),
                "Neutral": ("#FDB737", "#fff"),
                "Fear": ("#FF9E9E", "#fff"),
                "Extreme Fear": ("#FF8A65", "#fff")
            }
            bg, fg = color_map.get(text, ("#EEE", "#444"))
            return f"<span style='background:{bg};color:{fg};padding:2px 10px;border-radius:8px;font-size:20px;margin-left:6px'>{text}</span>"

        def sub_txt(text):
            return f"<span style='font-size:16px; font-weight:600; color:#666;margin-left:6px'>{text}</span>"

        html = "<div class='grid'>"
        # FGI
        html += f"<div class='card'><div class='card-title'>F&G Index</div><div class='card-value'>{(fgi_now if fgi_now is not None else '—')} {badge_html(fgi_label_now)}</div></div>"

        # QQQ
        qqq_val = f"{qqq_now:.2f}" if qqq_now is not None else "—"
        qqq_delta = f"({qqq_chg:+.2f}%)" if qqq_chg is not None else ""
        qqq_ma_txt = ""
        if qqq_ma5 is not None or qqq_ma20 is not None:
            m5 = f"{qqq_ma5:.2f}" if qqq_ma5 is not None else "—"
            m20 = f"{qqq_ma20:.2f}" if qqq_ma20 is not None else "—"
            m5_dis = f"{((qqq_now/qqq_ma5))*100:.1f}" if (qqq_now is not None and qqq_ma5 not in [None,0]) else "—"
            m20_dis = f"{((qqq_now/qqq_ma20))*100:.1f}" if (qqq_now is not None and qqq_ma20 not in [None,0]) else "—"
            qqq_ma_txt = sub_txt(f"5MA : {m5} ({m5_dis}) / 20MA : {m20} ({m20_dis})")
        if qqq_ma5 is not None or qqq_ma20 is not None:
            m5_dis = f""
        html += f"<div class='card'><div class='card-title'>QQQ</div><div class='card-value'>{qqq_val}<span style='margin-left:4px'>{qqq_ma_txt}</div></div>"

        # VIX
        vix_val = f"{vix_now:.2f}" if vix_now is not None else "—"
        vix_delta = f"({vix_chg:+.2f}%)" if vix_chg is not None else ""
        html += f"<div class='card'><div class='card-title'>VIX</div><div class='card-value'>{vix_val}</div></div>"

        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    render_top_cards(fgi_now, fgi_label_now, qqq_now, qqq_chg, vix_now, vix_chg, qqq_ma5=ma5, qqq_ma20=ma20)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)

    with b1:
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

        table_html = "<colgroup>" + "".join(["<col style='width:33.3%'>" for _ in range(3)]) + "</colgroup>"
        render_table("FGI · 최근 20영업일", ["날짜", "FGI", "지표"], rows)

    with b2:
        qqq = fetch_last10_daily("QQQ").iloc[::-1]

        mdd_latest_zero_date = None
        if not qqq.empty:
            mdd_zero_rows = qqq[qqq["MDD_%"].abs() < 1e-12]
            if not mdd_zero_rows.empty:
                mdd_latest_zero_date = mdd_zero_rows["Date"].max()

        rows = []
        for _, r in qqq.iterrows():
            if (mdd_latest_zero_date is not None) and (r["Date"] == mdd_latest_zero_date):
                date = f"<td style='padding:6px 8px;text-align:center;background:#FFF3C4;border-radius:4px'>{r['Date'].strftime('%Y-%m-%d')}</td>"
            else:
                date = f"<td style='padding:6px 8px;text-align:center'>{r['Date'].strftime('%Y-%m-%d')}</td>"

            price = f"<td style='padding:6px 8px;text-align:right'>{r['Close']:.2f}</td>"
            dod_html = span_pct(r["DoD_%"])
            mdd_html = span_mdd(r["MDD_%"])
            rows.append([date, price,
                        f"<td style='padding:6px 8px;text-align:right'>{dod_html}</td>",
                        f"<td style='padding:6px 8px;text-align:right'>{mdd_html}</td>"])
        render_table("QQQ · 최근 20영업일", ["날짜","가격","전일대비","고점대비"], rows)

    with b3:
        vix = fetch_last10_vix().iloc[::-1]
        rows = []
        for _, r in vix.iterrows():
            date = r["Date"].strftime("%Y-%m-%d")
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
        render_table("VIX · 최근 20영업일", ["날짜", "가격", "전일대비", "전주대비"], rows)

    st.caption("FGI: feargreedmeter.com · QQQ/VIX: Yahoo Finance(약 20분 지연)")

with tab2:
    etfs=[
        ("379810.KS","KODEX 미국나스닥100"),
        ("487230.KS","KODEX 미국AI전력핵심인프라"),
        ("486450.KS","SOL 미국AI전력인프라"),
        # 필요 시 계속 추가
    ]
    cols=None
    for i,(etf_ticker,etf_name) in enumerate(etfs):
        if i%3==0:
            cols=st.columns(3)

        with cols[i%3]:
            etf_df = fetch_last_daily_generic(etf_ticker, n=20)
            mdd_badge = ""
            if not etf_df.empty:
                latest_mdd = float(etf_df.iloc[-1]["MDD_%"])
                if latest_mdd <= -5:  # -5% 이하만 표시
                    mdd_abs = abs(latest_mdd)
                    # 색상 그라데이션
                    if mdd_abs < 10:
                        color = "#F6A194"  # 연분홍
                    elif mdd_abs < 15:
                        color = "#E65A4C"  # 살몬
                    else:
                        color = "#D73027"  # 진한 붉은색
                    mdd_badge = f"<span style='background:{color};color:#fff;padding:2px 8px;border-radius:6px;font-size:14px;font-weight:600; margin-left:6px'>▼ {mdd_abs:.1f}%</span>"

            mdd_latest_zero_date = None
            if not etf_df.empty:
                mdd_zero_rows = etf_df[etf_df["MDD_%"].abs() < 1e-12]
                if not mdd_zero_rows.empty:
                    mdd_latest_zero_date = mdd_zero_rows["Date"].max()

            etf_now = fetch_last_close(etf_ticker)
            html = (
                f"<div class='card'><div class='card-title'>{etf_name}</div>"
                f"<div class='card-value'>{(f'{etf_now:,.0f}' if etf_now is not None else '—')}{mdd_badge}</div></div>"
            )
            st.markdown(html, unsafe_allow_html=True)
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            if not etf_df.empty:
                etf_df = etf_df.iloc[::-1]
                rows = []
                for _, r in etf_df.iterrows():
                    if (mdd_latest_zero_date is not None) and (r["Date"] == mdd_latest_zero_date):
                        date = f"<td style='padding:6px 8px;text-align:center;background:#FFF3C4;border-radius:4px'>{r['Date'].strftime('%Y-%m-%d')}</td>"
                    else:
                        date = f"<td style='padding:6px 8px;text-align:center'>{r['Date'].strftime('%Y-%m-%d')}</td>"

                    price = f"<td style='padding:6px 8px;text-align:right'>{r['Close']:,.0f}</td>"
                    dod = f"<td style='padding:6px 8px;text-align:right'>{span_pct(r['DoD_%'])}</td>"
                    mdd = f"<td style='padding:6px 8px;text-align:right'>{span_mdd(r['MDD_%'])}</td>"
                    
                    rows.append([date, price, dod, mdd])
                render_table(f"{etf_ticker}", ["날짜","가격","전일대비","고점대비"], rows)
