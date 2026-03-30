import sqlite3
import pandas as pd
import numpy as np

def load_data(db_path='gspjt/data/commodity_analysis.db'):
    conn = sqlite3.connect(db_path)
    df_wide = pd.read_sql("SELECT * FROM wide_prices", conn)
    df_wide['Date'] = pd.to_datetime(df_wide['Date'])
    db_cols_map = {'S&P500': 'S&P 500', 'US Dollar Index': 'Dollar Index', '10Y TY': '10Y Yield'}
    df_wide = df_wide.rename(columns=db_cols_map)
    cols = ['Date', 'Gold', 'Silver', 'Dollar Index', 'S&P 500', '10Y Yield']
    df_target = df_wide[[c for c in cols if c in df_wide.columns]]
    df_raw = df_target.melt(id_vars='Date', var_name='Ticker', value_name='Close')
    df_gsr = pd.read_sql("SELECT * FROM gold_silver_ratio", conn)
    df_gsr['Date'] = pd.to_datetime(df_gsr['Date'])
    df_gsr = df_gsr.drop_duplicates('Date').sort_values('Date')
    conn.close()
    return df_raw, df_gsr

def get_cohort_data(df_raw, buffer_days=60, window_days=150):
    cohorts = {'COVID-19': {'start': '2020-03-11'}, 'Russia-Ukraine': {'start': '2022-02-24'}, 'US-Iran': {'start': '2026-02-27'}}
    results = {}
    pivot_all = df_raw.pivot(index='Date', columns='Ticker', values='Close').ffill()
    for name, info in cohorts.items():
        start_date = pd.to_datetime(info['start'])
        if start_date not in pivot_all.index:
            valid_dates = pivot_all.index[pivot_all.index <= start_date]
            if not valid_dates.empty: start_date = valid_dates[-1]
            else: continue
        all_dates = pivot_all.index.tolist()
        start_idx = all_dates.index(start_date)
        idx_start, idx_end = max(0, start_idx - buffer_days), min(len(all_dates) - 1, start_idx + window_days)
        df_window = pivot_all.iloc[idx_start:idx_end+1].copy()
        base_price = pivot_all.loc[start_date]
        df_norm = (df_window / base_price - 1) * 100
        df_norm['T_Days'] = [(d - start_date).days for d in df_norm.index]
        results[name] = df_norm
    return results, cohorts

def calculate_homology(cohort_results, target_name='US-Iran', asset='Gold'):
    if target_name not in cohort_results: return {}
    target_path = cohort_results[target_name][asset]
    similarities = {}
    for name, df in cohort_results.items():
        if name == target_name: continue
        similarities[name] = target_path.corr(df[asset])
    return similarities

def calculate_volatility_stats(df_raw):
    pivot = df_raw.pivot(index='Date', columns='Ticker', values='Close').pct_change().dropna()
    return pivot.iloc[-60:].std() * np.sqrt(252), pivot.std() * np.sqrt(252)

def get_gsr_metrics(df_gsr):
    return {'mean': df_gsr['Ratio'].mean(), 'current': df_gsr['Ratio'].iloc[-1]}

def calculate_risk_score(df_raw, weights={'vix': 0.4, 'yield': 0.3, 'fx': 0.3}):
    pivot = df_raw.pivot(index='Date', columns='Ticker', values='Close').ffill()
    sp500_ret = pivot['S&P 500'].pct_change()
    vix_proxy = (sp500_ret.rolling(5).std() * np.sqrt(252) * 100).fillna(0)
    vix_score = (vix_proxy / (vix_proxy.expanding().max() + 1e-9)) * 100
    yield_change = pivot['10Y Yield'].diff(5).abs().fillna(0)
    yield_score = (yield_change / (yield_change.expanding().max() + 1e-9)) * 100
    fx_score = ( (pivot['Dollar Index'] - pivot['Dollar Index'].rolling(20).mean()).abs().fillna(0) / 1e-9).expanding().max() * 0 # Dummy simplified
    fx_score = ( (pivot['Dollar Index'] - pivot['Dollar Index'].rolling(20).mean()).abs().fillna(0) / ( (pivot['Dollar Index'] - pivot['Dollar Index'].rolling(20).mean()).abs().expanding().max() + 1e-9 ) ) * 100
    score = (vix_score * weights['vix'] + yield_score * weights['yield'] + fx_score * weights['fx']).iloc[-1]
    return min(100, max(0, score))

def analyze_strategy_v4(df_cohort, risk_score):
    """
    동적 자산 배분 전략 분석 엔진 V4
    """
    pivot_rets = df_cohort.drop(columns=['T_Days'], errors='ignore').pct_change().fillna(0)
    if risk_score >= 70: model_name, w = "철저 방어형", [0.50, 0.05, 0.20, 0.20, 0.05]
    elif risk_score >= 40: model_name, w = "균형 관리형", [0.30, 0.20, 0.15, 0.15, 0.20]
    else: model_name, w = "적극 수익형", [0.10, 0.30, 0.00, 0.10, 0.50]
    assets = ['Gold', 'Silver', 'Dollar Index', '10Y Yield', 'S&P 500']
    p_ret = sum(pivot_rets[asset] * weight for asset, weight in zip(assets, w))
    bench_ret = pivot_rets['S&P 500']
    return {
        'model_name': model_name, 'weights': dict(zip(assets, w)),
        'strategy_cum': (1 + p_ret).cumprod() - 1, 'bench_cum': (1 + bench_ret).cumprod() - 1,
        'strategy_mdd': ((1+p_ret).cumprod() / (1+p_ret).cumprod().cummax() - 1).min(),
        'bench_mdd': ((1+bench_ret).cumprod() / (1+bench_ret).cumprod().cummax() - 1).min()
    }
