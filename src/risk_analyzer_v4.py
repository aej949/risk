import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def sync_market_data(db_path):
    conn = sqlite3.connect(db_path)
    try:
        last_date_df = pd.read_sql("SELECT MAX(Date) as max_date FROM wide_prices", conn)
        l_val = last_date_df['max_date'].iloc[0]
        last_date = pd.to_datetime(l_val, format='mixed') if l_val else datetime(2020, 1, 1)
    except: last_date = datetime(2020, 1, 1)
    s_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    today_str = datetime.now().strftime('%Y-%m-%d')
    if s_date <= today_str:
        tickers = {'GC=F':'Gold', 'SI=F':'Silver', 'UUP':'US Dollar Index', '^GSPC':'S&P500'}
        data = yf.download(list(tickers.keys()), start=s_date)['Close']
        if not data.empty:
            if isinstance(data, pd.Series): data = data.to_frame()
            data = data.rename(columns=tickers); data.index.name = 'Date'; data = data.reset_index()
            data['Date'] = pd.to_datetime(data['Date']).dt.normalize().dt.strftime('%Y-%m-%d')
            data = data.drop_duplicates(subset=['Date'])
            data.to_sql('wide_prices', conn, if_exists='append', index=False)
            if 'Gold' in data.columns and 'Silver' in data.columns:
                gsr_df = data[['Date', 'Gold', 'Silver']].copy()
                gsr_df['Ratio'] = gsr_df['Gold'] / gsr_df['Silver'].replace(0, np.nan)
                gsr_df[['Date', 'Ratio']].dropna().to_sql('gold_silver_ratio', conn, if_exists='append', index=False)
    conn.close()

def load_data(db_path='data/commodity_analysis.db'):
    sync_market_data(db_path)
    conn = sqlite3.connect(db_path)
    df_w = pd.read_sql("SELECT * FROM wide_prices", conn)
    df_w['Date'] = pd.to_datetime(df_w['Date'], format='mixed').dt.normalize()
    df_w = df_w.drop_duplicates(subset=['Date']).sort_values('Date')
    db_cols_map = {'S&P500':'S&P 500', 'US Dollar Index':'Dollar Index'}
    df_w = df_w.rename(columns=db_cols_map)
    cols = ['Date', 'Gold', 'Silver', 'Dollar Index', 'S&P 500']
    df_target = df_w[[c for c in cols if c in df_w.columns]].ffill()
    df_raw = df_target.melt(id_vars='Date', var_name='Ticker', value_name='Close')
    df_gsr = pd.read_sql("SELECT * FROM gold_silver_ratio", conn)
    df_gsr['Date'] = pd.to_datetime(df_gsr['Date'], format='mixed').dt.normalize()
    df_gsr = df_gsr.drop_duplicates(subset=['Date']).sort_values('Date')
    conn.close()
    return df_raw, df_gsr

def get_cohort_data(df_raw, buffer_days=60, window_days=150):
    cohorts = {'COVID-19':{'start':'2020-03-11'}, 'Russia-Ukraine':{'start':'2022-02-24'}, 
               'Fed Rate Hike':{'start':'2022-06-10'}, 'SVB Crisis':{'start':'2023-03-10'}, 
               'US-Iran':{'start':'2026-02-27'}}
    results = {}
    pivot_all = df_raw.pivot(index='Date', columns='Ticker', values='Close').sort_index()
    l_date = df_raw['Date'].max()
    for name, info in cohorts.items():
        s_date = pd.to_datetime(info['start'])
        if s_date not in pivot_all.index:
            v_dates = pivot_all.index[pivot_all.index <= s_date]
            if not v_dates.empty: s_date = v_dates[-1]
            else: continue
        all_dates = pivot_all.index.tolist(); s_index = all_dates.index(s_date)
        i_start, i_end = max(0, s_index - buffer_days), min(len(all_dates) - 1, s_index + window_days)
        df_win = pivot_all.iloc[i_start:i_end+1].copy().ffill()
        if name == 'US-Iran': df_win = df_win[df_win.index <= l_date]
        base = pivot_all.loc[s_date]
        df_norm = (df_win / base.replace(0, np.nan).fillna(1e-9) - 1) * 100
        df_norm['T_Days'] = [(d - s_date).days for d in df_norm.index]
        for col in ['Gold', 'Silver', 'Dollar Index', 'S&P 500']:
            if col in df_win.columns:
                df_norm[f'{col}_DD'] = (df_win[col] / df_win[col].cummax() - 1) * 100
        results[name] = df_norm
    return results, cohorts

def get_optimized_crisis_weights(cohort_results):
    """
    1. 과거 3대 위기(코인, 러-우, SVB) 학습 데이터 기반 최적 비중 산출
    로직: T=0 ~ T+60 구간의 자산별 변동성 역가중(Inverse Volatility) 평균
    """
    learning_cohorts = ['COVID-19', 'Russia-Ukraine', 'SVB Crisis']
    assets = ['Gold', 'Silver', 'Dollar Index', 'S&P 500']
    all_vols = []
    
    for name in learning_cohorts:
        if name not in cohort_results: continue
        df = cohort_results[name]
        # T=0 ~ T+60 구간 추출
        df_t60 = df[(df['T_Days'] >= 0) & (df['T_Days'] <= 60)][assets]
        # 일일 수익률 변동성 산출
        vols = df_t60.diff().std()
        all_vols.append(vols)
    
    if not all_vols: return dict(zip(assets, [0.25, 0.25, 0.25, 0.25]))
    
    # 3대 위기 평균 변동성 역수 계산
    avg_vols = pd.concat(all_vols, axis=1).mean(axis=1)
    inv_vols = 1 / (avg_vols + 1e-9)
    weights = inv_vols / inv_vols.sum()
    
    return weights.to_dict()

def calculate_homology(cohort_results, target_name='US-Iran', asset='Gold'):
    if target_name not in cohort_results: return {}
    target_data = cohort_results[target_name]
    target_p = target_data[target_data['T_Days'] >= 0][asset]
    sims = {}
    for name, df in cohort_results.items():
        if name == target_name: continue
        past_p = df[df['T_Days'] >= 0][asset]
        min_len = min(len(target_p), len(past_p))
        if min_len > 3:
            s1, s2 = target_p.iloc[:min_len], past_p.iloc[:min_len]
            corr = s1.corr(s2) if s1.std() != 0 and s2.std() != 0 else 0.0
            sims[name] = corr if not np.isnan(corr) else 0.0
        else: sims[name] = 0.0
    return sims

def calculate_risk_score(df_raw):
    pivot = df_raw.pivot(index='Date', columns='Ticker', values='Close').ffill()
    sp500 = pivot['S&P 500'].tail(30)
    sp500_mdd = ((sp500 / sp500.cummax() - 1).min()) * 100
    mdd_score = min(100, abs(sp500_mdd) * 10)
    usd_vol = pivot['Dollar Index'].tail(30).pct_change().std() * np.sqrt(252) * 100
    vol_score = min(100, usd_vol * 10)
    final_score = (mdd_score * 0.7) + (vol_score * 0.3)
    return {'total': min(100, max(0, final_score)), 'mdd_part': mdd_score, 'vol_part': vol_score, 'mdd_val': sp500_mdd, 'vol_val': usd_vol}

def analyze_strategy_v4(df_cohort, risk_score, benchmark_asset='S&P 500'):
    cols = ['Gold', 'Silver', 'Dollar Index', 'S&P 500']
    pivot_rets = df_cohort[cols].diff().fillna(0) / 100
    if risk_score >= 70: m_name, w = "철저 방어형", [0.60, 0.00, 0.30, 0.10]
    elif risk_score >= 40: m_name, w = "균형 관리형", [0.40, 0.10, 0.20, 0.30]
    else: m_name, w = "적극 수익형", [0.10, 0.30, 0.00, 0.60]
    p_ret = sum(pivot_rets[asset] * weight for asset, weight in zip(cols, w))
    bench_ret = pivot_rets[benchmark_asset]
    return {'model_name': m_name, 'weights': dict(zip(cols, w)),
            'strategy_cum': (1 + p_ret).cumprod() - 1, 'bench_cum': (1 + bench_ret).cumprod() - 1,
            'strategy_mdd': ((1+p_ret).cumprod() / (1+p_ret).cumprod().cummax() - 1).min(),
            'bench_mdd': ((1+bench_ret).cumprod() / (1+bench_ret).cumprod().cummax() - 1).min()}

def get_forward_test_result(target_df, weights):
    """
    2. 미-이란 전쟁(현재 데이터) 포워드 테스트 적용
    로직: T=0일(2026-02-27) 종가 대비 오늘 가격의 누적 수익률 계산
    """
    assets = ['Gold', 'Silver', 'Dollar Index', 'S&P 500']
    # T=0 데이터 (정규화된 %가 아닌 가격 변동률 기반)
    df_t = target_df[target_df['T_Days'] >= 0][assets]
    if df_t.empty: return 0.0, 0.0
    
    # 2-1. 복리 누적 수익률 계산 (T=0 기준)
    # (오늘 가격 / 시작가) - 1
    # df_t는 이미 (price / base - 1) * 100 형식이므로, 이를 역산
    p_value = sum((df_t[asset] / 100 + 1) * weights[asset] for asset in assets)
    final_cum_ret = (p_value.iloc[-1] - 1) * 100 # % 단위
    
    # 벤치마크 (S&P 500 100% 보유)
    bench_cum_ret = ((df_t['S&P 500'].iloc[-1] / 100 + 1) - 1) * 100
    
    return final_cum_ret, bench_cum_ret

def get_gsr_metrics(df_gsr): return {'mean': df_gsr['Ratio'].mean(), 'current': df_gsr['Ratio'].iloc[-1]}
