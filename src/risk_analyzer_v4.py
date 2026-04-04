import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def sync_market_data(db_path):
    """
    1. Data Pipeline: yfinance 실시간 연동 및 가짜 데이터 차단
    DB의 실제 컬럼명을 동적으로 파악하여 yfinance 데이터를 업데이트함.
    """
    conn = sqlite3.connect(db_path)
    try:
        # DB의 마지막 날짜 및 컬럼명 확인
        df_schema = pd.read_sql("SELECT * FROM wide_prices LIMIT 1", conn)
        db_cols = list(df_schema.columns)
        
        l_val = pd.read_sql("SELECT MAX(Date) as max_date FROM wide_prices", conn)['max_date'].iloc[0]
        last_date = pd.to_datetime(l_val, format='mixed').normalize() if l_val else pd.Timestamp('2020-01-01').normalize()
        
        # 마지막 날짜 다음날부터 오늘까지 데이터 요청
        s_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        today = pd.Timestamp.today().normalize()
        
        if s_date <= today.strftime('%Y-%m-%d'):
            # yfinance 티커와 DB 컬럼 동적 매핑
            ticker_map = {'GC=F': 'Gold', 'SI=F': 'Silver', 'UUP': 'Dollar', '^GSPC': 'S&P'}
            target_tickers = list(ticker_map.keys())
            
            data = yf.download(target_tickers, start=s_date, end=(today + timedelta(days=1)).strftime('%Y-%m-%d'))['Close']
            
            if not data.empty:
                if isinstance(data, pd.Series): data = data.to_frame()
                
                # DB의 실제 컬럼명 찾기
                final_rename = {}
                for ticker, keyword in ticker_map.items():
                    actual_col = next((c for c in db_cols if keyword.lower() in c.lower() or ticker.lower() in c.lower()), None)
                    if actual_col:
                        final_rename[ticker] = actual_col
                
                data = data.rename(columns=final_rename)
                data.index.name = 'Date'
                data = data.reset_index()
                
                # 날짜 정규화 및 미래 데이터 차단
                data['Date'] = pd.to_datetime(data['Date'], format='mixed').dt.normalize()
                data = data[data['Date'] <= today].copy()
                data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
                
                # DB에 존재하는 컬럼만 선별하여 저장
                save_cols = [c for c in data.columns if c in db_cols]
                data[save_cols].to_sql('wide_prices', conn, if_exists='append', index=False)
                
                # GSR 업데이트
                gold_col = next((c for c in db_cols if 'gold' in c.lower()), None)
                silver_col = next((c for c in db_cols if 'silver' in c.lower()), None)
                if gold_col in data.columns and silver_col in data.columns:
                    gsr_df = data[['Date', gold_col, silver_col]].copy()
                    gsr_df['Ratio'] = gsr_df[gold_col] / gsr_df[silver_col].replace(0, np.nan)
                    gsr_df[['Date', 'Ratio']].dropna().to_sql('gold_silver_ratio', conn, if_exists='append', index=False)
    except Exception as e:
        print(f"Sync error: {e}")
    finally:
        conn.close()

def load_data(db_path='data/commodity_analysis.db'):
    """
    전체 데이터를 로드하고 전처리를 수행함. 미래 데이터는 dropna()로 엄격히 잘라냄.
    """
    sync_market_data(db_path)
    conn = sqlite3.connect(db_path)
    
    # 1. 메인 자산 데이터 로드
    df_w = pd.read_sql("SELECT * FROM wide_prices", conn)
    db_cols = list(df_w.columns)
    df_w['Date'] = pd.to_datetime(df_w['Date'], format='mixed').dt.normalize()
    df_w = df_w.drop_duplicates(subset=['Date']).sort_values('Date')
    
    # 앱에서 사용할 표준 명칭으로 매핑
    rename_map = {}
    standard_names = {'Gold': 'Gold', 'Silver': 'Silver', 'Dollar': 'Dollar Index', 'S&P': 'S&P 500', '^GSPC': 'S&P 500'}
    for actual_col in db_cols:
        for keyword, standard in standard_names.items():
            if keyword.lower() in actual_col.lower():
                rename_map[actual_col] = standard
                break
    
    df_w = df_w.rename(columns=rename_map)
    assets = ['Gold', 'Silver', 'Dollar Index', 'S&P 500']
    
    # 오늘 이전까지만 데이터를 남기고 결측치는 ffill() (미래 ffill 절대 금지)
    today = pd.Timestamp.today().normalize()
    df_w = df_w[df_w['Date'] <= today].copy()
    
    # 필요한 컬럼만 추출하여 전처리
    target_cols = ['Date'] + [a for a in assets if a in df_w.columns]
    df_target = df_w[target_cols].copy()
    existing_assets = [a for a in assets if a in df_target.columns]
    df_target[existing_assets] = df_target[existing_assets].ffill()
    df_target = df_target.dropna()
    
    # Melt 형식의 raw 데이터 생성
    df_raw = df_target.melt(id_vars='Date', var_name='Ticker', value_name='Close')
    
    # 2. GSR 데이터 로드
    df_gsr = pd.read_sql("SELECT * FROM gold_silver_ratio", conn)
    df_gsr['Date'] = pd.to_datetime(df_gsr['Date'], format='mixed').dt.normalize()
    df_gsr = df_gsr[df_gsr['Date'] <= pd.Timestamp.today().normalize()].sort_values('Date').drop_duplicates(subset=['Date'])
    
    conn.close()
    return df_raw, df_gsr

def get_cohort_data(df_raw, buffer_days=60, window_days=150):
    """
    2. Zero-Anchoring: T=0 수익률 영점 강제 고정
    3. True Drawdown: 구간 내 최댓값 기준 독립적 낙폭 계산
    """
    # 위기 리스트 및 하드코딩된 T=0
    cohorts = {
        'COVID-19': {'start': '2020-03-11'}, 
        'Russia-Ukraine': {'start': '2022-02-24'}, 
        'Fed Rate Hike': {'start': '2022-06-10'}, 
        'SVB Crisis': {'start': '2023-03-10'}, 
        'US-Iran': {'start': '2026-02-27'} # 현재 진행 중인 위기
    }
    
    assets = ['Gold', 'Silver', 'Dollar Index', 'S&P 500']
    pivot_all = df_raw.pivot(index='Date', columns='Ticker', values='Close').sort_index()
    results = {}
    
    for name, info in cohorts.items():
        t0_date = pd.to_datetime(info['start']).normalize()
        
        # T=0 영점 고정: 해당일 데이터가 없으면 '직전' 거래일을 찾음
        if t0_date not in pivot_all.index:
            past_dates = pivot_all.index[pivot_all.index < t0_date]
            if not past_dates.empty:
                t0_date = past_dates[-1]
            else:
                continue
                
        # 윈도우 계산 (T-60 ~ T+150)
        all_dates = pivot_all.index.tolist()
        t0_idx = all_dates.index(t0_date)
        start_idx = max(0, t0_idx - buffer_days)
        end_idx = min(len(all_dates) - 1, t0_idx + window_days)
        
        df_win = pivot_all.iloc[start_idx:end_idx+1].copy()
        
        # T=0 가격 (Base Price) 고정
        base_prices = pivot_all.loc[t0_date]
        
        # 공식: (현재 가격 / Base Price) - 1 적용
        df_norm = (df_win / base_prices.replace(0, np.nan) - 1) * 100
        df_norm['T_Days'] = [(d - t0_date).days for d in df_norm.index]
        
        # 3. True Drawdown: 해당 구간 내 누적 최고가(Cumulative Max) 기준 계산
        for col in assets:
            if col in df_win.columns:
                # 구간 내 누적 최고가 (윈도우 내에서의 고점만 인정)
                win_cum_max = df_win[col].cummax()
                df_norm[f'{col}_DD'] = (df_win[col] / win_cum_max - 1) * 100
        
        results[name] = df_norm
        
    return results, cohorts

def calculate_homology(cohort_results, target_name='US-Iran', asset='Gold'):
    """
    4. Homology (상동성) 계산 길이 동기화
    두 시리즈 중 짧은 쪽에 맞춰 슬라이싱 후 상관계수 corr() 계산
    """
    if target_name not in cohort_results:
        return {}
        
    target_df = cohort_results[target_name]
    # T=0 이후 데이터만 추출
    target_p = target_df[target_df['T_Days'] >= 0][asset]
    
    sims = {}
    for name, df in cohort_results.items():
        if name == target_name:
            continue
            
        past_p = df[df['T_Days'] >= 0][asset]
        
        # 4-1. 데이터 길이 동기화 (짧은 쪽에 맞춤)
        min_len = min(len(target_p), len(past_p))
        
        if min_len > 5: # 최소 통계적 유의미성 확보
            s1 = target_p.iloc[:min_len]
            s2 = past_p.iloc[:min_len]
            
            # 상관계수 계산
            if s1.std() != 0 and s2.std() != 0:
                corr = s1.corr(s2)
                sims[name] = corr if not np.isnan(corr) else 0.0
            else:
                sims[name] = 0.0
        else:
            sims[name] = 0.0
            
    return sims

def get_optimized_crisis_weights(cohort_results):
    """
    5. 4대 자산 하드코딩 및 비중 합 1.0(100%) 보장
    과거 위기 데이터를 학습하여 변동성 역가중 기반 비중 산출
    """
    learning_cohorts = ['COVID-19', 'Russia-Ukraine', 'SVB Crisis']
    assets = ['Gold', 'Silver', 'Dollar Index', 'S&P 500']
    vols_list = []
    
    for name in learning_cohorts:
        if name not in cohort_results:
            continue
        df = cohort_results[name]
        # T=0 ~ T+60 구간 추출 (수익률 변동성 분석)
        df_t60 = df[(df['T_Days'] >= 0) & (df['T_Days'] <= 60)][assets]
        if not df_t60.empty:
            # 일일 수익률(diff)의 표준편차 계산
            vols = df_t60.diff().std()
            vols_list.append(vols)
            
    if not vols_list:
        # 데이터가 없을 경우 균등 배분 (전체 합 1.0 보장)
        return dict(zip(assets, [0.25, 0.25, 0.25, 0.25]))
        
    # 평균 변동성 계산 및 역수 가중치 산정
    avg_vols = pd.concat(vols_list, axis=1).mean(axis=1)
    inv_vols = 1 / (avg_vols + 1e-9)
    # 비중 합이 정확히 1.0이 되도록 정규화
    weights = inv_vols / inv_vols.sum()
    
    return weights.to_dict()

def get_forward_test_result(target_df, weights):
    """
    미-이란 전쟁(T=0)부터 오늘까지의 실제 포트폴리오 누적 수익률 계산
    target_df는 이미 (Price/Base - 1) * 100 형식으로 정규화됨
    """
    assets = ['Gold', 'Silver', 'Dollar Index', 'S&P 500']
    # T=0 이후 구간 추출
    df_t = target_df[target_df['T_Days'] >= 0][assets].copy()
    
    if df_t.empty:
        return 0.0, 0.0
        
    # 포트폴리오 가치 변화 계산
    # df_t[asset] / 100 + 1 은 (현재가 / T0가) 를 의미함
    port_cumulative = sum((df_t[asset] / 100 + 1) * weights.get(asset, 0) for asset in assets)
    
    # 최종 누적 수익률 (%)
    final_cum_ret = (port_cumulative.iloc[-1] - 1) * 100
    
    # 벤치마크 (S&P 500 100%)
    bench_cum_ret = ((df_t['S&P 500'].iloc[-1] / 100 + 1) - 1) * 100
    
    return float(final_cum_ret), float(bench_cum_ret)

def calculate_risk_score(df_raw):
    """
    최근 시장 상황(S&P 500 MDD 및 USD 변동성) 기반 종합 리스크 점수 산출
    """
    pivot = df_raw.pivot(index='Date', columns='Ticker', values='Close').ffill()
    if 'S&P 500' not in pivot.columns or 'Dollar Index' not in pivot.columns:
        return {'total': 50, 'mdd_part': 50, 'vol_part': 50, 'mdd_val': 0, 'vol_val': 0}
        
    # 최근 30거래일 기준 MDD
    sp500 = pivot['S&P 500'].tail(30)
    sp500_mdd = ((sp500 / sp500.cummax() - 1).min()) * 100
    mdd_score = min(100, abs(sp500_mdd) * 10)
    
    # 최근 30거래일 기준 USD 변동성 (연환산)
    usd_vol = pivot['Dollar Index'].tail(30).pct_change().std() * np.sqrt(252) * 100
    vol_score = min(100, usd_vol * 10)
    
    # 7:3 가중치 적용
    final_score = (mdd_score * 0.7) + (vol_score * 0.3)
    
    return {
        'total': min(100, max(0, final_score)),
        'mdd_part': mdd_score,
        'vol_part': vol_score,
        'mdd_val': sp500_mdd,
        'vol_val': usd_vol
    }

def analyze_strategy_v4(df_cohort, risk_score, benchmark_asset='S&P 500'):
    """
    리스크 점수에 따른 동적 자산배분 모델 성과 분석
    """
    assets = ['Gold', 'Silver', 'Dollar Index', 'S&P 500']
    
    # 일일 수익률 기반 계산 (정규화된 %가 아닌 원래 비율로 변환)
    # df_cohort 는 (P/P0 - 1)*100 이므로, 이를 P/P0 로 변환 후 증률 계산
    pivot_prices = (df_cohort[assets] / 100 + 1)
    pivot_rets = pivot_prices.pct_change().fillna(0)
    
    # 리스크 점수 기반 모델 결정
    if risk_score >= 70:
        m_name, w = "철저 방어형", [0.60, 0.00, 0.30, 0.10]
    elif risk_score >= 40:
        m_name, w = "균형 관리형", [0.40, 0.10, 0.20, 0.30]
    else:
        m_name, w = "적극 수익형", [0.10, 0.30, 0.00, 0.60]
        
    # 전략 수익률 합성
    p_ret = sum(pivot_rets[asset] * weight for asset, weight in zip(assets, w))
    bench_ret = pivot_rets[benchmark_asset]
    
    # 누적 수익률 및 MDD (단리/복리 혼합 방지 위해 복리 적용)
    strategy_cum = (1 + p_ret).cumprod() - 1
    bench_cum = (1 + bench_ret).cumprod() - 1
    
    strategy_mdd = (strategy_cum + 1) / (strategy_cum + 1).cummax() - 1
    bench_mdd = (bench_cum + 1) / (bench_cum + 1).cummax() - 1
    
    return {
        'model_name': m_name,
        'weights': dict(zip(assets, w)),
        'strategy_cum': strategy_cum,
        'bench_cum': bench_cum,
        'strategy_mdd': strategy_mdd.min(),
        'bench_mdd': bench_mdd.min()
    }

def get_gsr_metrics(df_gsr):
    """
    Gold-Silver Ratio 통계 지표 산출
    """
    if df_gsr.empty:
        return {'mean': 0, 'current': 0}
    return {
        'mean': df_gsr['Ratio'].mean(),
        'current': df_gsr['Ratio'].iloc[-1]
    }
