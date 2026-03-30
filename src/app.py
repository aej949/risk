import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import risk_analyzer_v4
import importlib

# 경로 설정 및 리로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)
importlib.reload(risk_analyzer_v4)

from risk_analyzer_v4 import (load_data, get_cohort_data, calculate_homology, 
                         get_gsr_metrics, calculate_risk_score, analyze_strategy_v4,
                         get_optimized_crisis_weights, get_forward_test_result)

# 페이지 설정
st.set_page_config(layout="wide", page_title="Global Macro Strategy Report V7")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; background-color: #f8f9fa; }
    .report-header { background-color: #1a1c23; color: white; padding: 1.5rem; border-radius: 0 0 15px 15px; margin-bottom: 2rem; border-bottom: 4px solid #f39c12; }
    .summary-box { background-color: white; border: 1px solid #dee2e6; border-left: 5px solid #f39c12; padding: 20px; border-radius: 8px; margin-bottom: 1.5rem; }
    .section-title { color: #1a1c23; border-bottom: 2px solid #f39c12; padding-bottom: 5px; font-weight: 700; margin: 2.5rem 0 1rem 0; }
    .insight-box { border-left: 4px solid #f39c12; background-color: #fdfcfb; padding: 20px; margin: 10px 0; border: 1px solid #eee; border-radius: 4px; }
    .term-box { background-color: #f1f3f5; padding: 10px; border-radius: 4px; font-size: 0.85rem; color: #495057; border: 1px dashed #ced4da; margin-top: 10px; }
    .term-label { font-weight: 700; color: #1a1c23; margin-right: 5px; }
    .weight-table { width: 100%; border-collapse: collapse; margin-top: 10px; border: 1px solid #eee; }
    .weight-table th { background-color: #1a1c23; color: white; padding: 8px; text-align: left; font-size: 0.9rem; }
    .weight-table td { padding: 8px; border-bottom: 1px solid #eee; font-size: 0.9rem; }
    .fact-metric { background-color: #ffffff; padding: 25px; border-radius: 12px; border: 2px solid #f39c12; text-align: center; }
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown("""<div class='report-header'><div style='display: flex; justify-content: space-between; align-items: center;'>
<div style='font-size: 20px; font-weight: 900; letter-spacing: 2px;'>GLOBAL MACRO STRATEGIST</div>
<div style='text-align: right;'><div style='font-size: 1.1rem; font-weight: 700;'>Multi-Crisis Back/Forward Analysis</div><div style='font-size: 0.9rem; opacity: 0.8;'>Prime V7 - Fixed Logic Execution</div></div>
</div></div>""", unsafe_allow_html=True)

# 데 로드
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, '../../gspjt/data/commodity_analysis.db'))
try:
    df_raw, df_gsr = load_data(DB_PATH)
    cohort_results, cohort_info = get_cohort_data(df_raw)
except Exception as e:
    st.error(f"데이터 로드 치공 오류: {e}"); st.stop()

# 사이드바 (슬라이더 삭제 및 간소화)
st.sidebar.title("💎 리포트 제어반")
selected_crisis = st.sidebar.selectbox("🎯 분석 대상 위기 국면", list(cohort_info.keys()), index=len(cohort_info)-1)
st.sidebar.info("본 리포트는 4대 자산(Gold, Silver, USD, S&P 500) 최적화 로직에 따른 실전 검증 보고서입니다.")

# 분석 지계 계산
risk_data = calculate_risk_score(df_raw)
risk_score = risk_data['total']
gsr_metrics = get_gsr_metrics(df_gsr)

# --- [Section 1] Global Macro Alert ---
st.markdown("<div class='section-title'>📍 [Section 1] Global Macro Alert: 리스크 산출 근거</div>", unsafe_allow_html=True)
risk_reason = f"""위험 지수 **{risk_score:.1f}점** 산출 근거: **최근 30일 S&P 500 MDD(-{abs(risk_data['mdd_val']):.1f}%)** 70% 가중치 + **USD 연환산 변동성({risk_data['vol_val']:.2f}%)** 30% 가중치 적용."""
st.markdown(f"<div class='summary-box'>{risk_reason}</div>", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
m1.metric("종합 리스크 점수", f"{risk_score:.1f}"); m2.metric("S&P 500 MDD 기여", f"{risk_data['mdd_part']:.1f}점"); m3.metric("USD 변동성 기여", f"{risk_data['vol_part']:.1f}점")

st.write("#### 📉 위기별 낙폭률(Drawdown) 정밀 비교 (Y축 고정)")
dd_asset = st.selectbox("분석 대상 자산군", ["S&P 500", "Gold", "Silver", "Dollar Index"], index=0)
fig_dd = go.Figure()
color_map = {'COVID-19':'#1f77b4', 'Russia-Ukraine':'#ff7f0e', 'SVB Crisis':'#d62728', 'US-Iran':'#1a1c23'}
for name, df in cohort_results.items():
    df_win = df[(df['T_Days'] >= -30) & (df['T_Days'] <= 60)]
    is_target = (name == selected_crisis)
    fig_dd.add_trace(go.Scatter(x=df_win['T_Days'], y=df_win[f'{dd_asset}_DD'], name=f"[{name}]",
        line=dict(color=color_map.get(name, "#adb5bd"), width=4 if is_target else 1.5), opacity=1.0 if is_target else 0.4))
y_range = [-10, 2] if dd_asset == "Dollar Index" else [-50, 5]
fig_dd.add_vline(x=0, line_dash="solid", line_color="red", annotation_text="T=0")
fig_dd.update_layout(height=400, template="simple_white", xaxis=dict(range=[-30, 60]), yaxis=dict(range=y_range))
st.plotly_chart(fig_dd, use_container_width=True)

# --- [Section 2] Mirror Tracking & Homology Proof ---
st.markdown("<div class='section-title'>📊 [Section 2] Mirror Tracking: 통계적 상동성(Homology) 증명</div>", unsafe_allow_html=True)
track_asset = st.radio("추적 타겟 자산", ["S&P 500", "Gold", "Silver"], horizontal=True)
sims = calculate_homology(cohort_results, target_name=selected_crisis, asset=track_asset)
most_similar = max(sims, key=sims.get) if sims and len(sims) > 0 else "데이터 부재"
similarity_val = sims.get(most_similar, 0) if sims else 0.0
col_a1, col_a2 = st.columns([2.5, 1])
with col_a1:
    fig_track = go.Figure()
    for name, df in cohort_results.items():
        is_target = (name == selected_crisis); color = color_map.get(name, "#adb5bd")
        fig_track.add_trace(go.Scatter(x=df['T_Days'], y=df[track_asset], name=f"[{name}]",
            line=dict(color=color, width=4 if is_target else 1.5), opacity=1.0 if is_target else 0.4))
    fig_track.add_vline(x=0, line_dash="dash", line_color="black")
    fig_track.update_layout(height=450, template="simple_white", xaxis_title="T-Days", yaxis_title="누적 수익률 (%)")
    st.plotly_chart(fig_track, use_container_width=True)
with col_a2:
    st.markdown("#### 🔍 통계적 상동성 판정")
    st.info(f"현재 국면 발 후의 **{track_asset}** 수익률 곡선은 과거 **[{most_similar}]** 위기와 **상관계수(Pearson R) {similarity_val:.2f}**를 기록하며 가장 높은 통계적 유사성을 보이고 있습니다.")

# --- [Section 3] Backtest ---
st.markdown(f"<div class='section-title'>🧪 [Section 3] Backtest: 전략 모델 vs {dd_asset} 단순 보유</div>", unsafe_allow_html=True)
current_sim = analyze_strategy_v4(cohort_results[selected_crisis], risk_score, benchmark_asset=dd_asset)
col_b1, col_b2 = st.columns([2.5, 1.2])
with col_b1:
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=current_sim['strategy_cum'].index, y=current_sim['strategy_cum']*100, name=f"전략 ({current_sim['model_name']})", line=dict(color="#f39c12", width=3)))
    fig_perf.add_trace(go.Scatter(x=current_sim['bench_cum'].index, y=current_sim['bench_cum']*100, name=f"{dd_asset} 단순보유", line=dict(color="#dee2e6", width=2, dash="dot")))
    fig_perf.update_layout(height=450, template="simple_white", xaxis_title="Period", yaxis_title="누적 수익률 (%)")
    st.plotly_chart(fig_perf, use_container_width=True)
with col_b2:
    st.metric(f"벤치마크 대비 우위", f"{(current_sim['strategy_cum'].iloc[-1]-current_sim['bench_cum'].iloc[-1])*100:.1f}%p", delta=f"{current_sim['strategy_cum'].iloc[-1]*100:.1f}%")
    st.markdown("##### 🛡️ 포트폴리오 비중 (Weights)")
    w_df = pd.DataFrame(list(current_sim['weights'].items()), columns=['Asset', 'Weight (%)'])
    w_df['Weight (%)'] = (w_df['Weight (%)'] * 100).astype(int)
    w_html = "<table class='weight-table'><tr><th>자산군</th><th>비중</th></tr>"
    for _, r in w_df.iterrows(): w_html += f"<tr><td>{r['Asset']}</td><td><b>{r['Weight (%)']}%</b></td></tr>"
    w_html += "</table>"
    st.markdown(w_html, unsafe_allow_html=True)

# --- [Section 5] 실전 검증: 과거 위기 학습 모델 적용 결과 (NEW Section) ---
st.markdown("<div class='section-title'>🔥 [Section 5] 실전 검증: 과거 위기 학습 모델 적용 결과</div>", unsafe_allow_html=True)

# 1. 학습 로직 적용
opt_weights = get_optimized_crisis_weights(cohort_results)
target_crisis = "US-Iran"
fw_ret, fw_bench = get_forward_test_result(cohort_results[target_crisis], opt_weights)

col_d1, col_d2 = st.columns([1, 2])
with col_d1:
    st.markdown("<div class='fact-metric'>", unsafe_allow_html=True)
    st.write("### 🚀 실시간 포워드 테스트 성과")
    st.markdown(f"<div style='font-size: 3.5rem; font-weight: 900; color: #f39c12;'>{fw_ret:.2f}%</div>", unsafe_allow_html=True)
    st.write(f"(사건 발생일 2026-02-27 대비 오늘 기준 누적 수익률)")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown(f"""<div class='insight-box' style='background-color: #fff;'>
        <b>팩트 체크:</b> 과거 위기 최적화 가중치를 미-이란 전쟁 발발일에 100% 투자했다면 오늘까지 <b>{fw_ret:.2f}%</b>의 수익(또는 손실)을 기록 중입니다. 
        동일 기간 S&P 500 100% 매수 성과(<b>{fw_bench:.2f}%</b>)와 비교 시 <b>{fw_ret - fw_bench:.2f}%p</b>의 초과 성과를 기록 중입니다.
    </div>""", unsafe_allow_html=True)

with col_d2:
    st.write("#### 🛡️ 위기 최적화 가중치 (과거 3대 위기 평균 학습 결과)")
    fig_opt_pie = px.pie(values=list(opt_weights.values()), names=list(opt_weights.keys()), 
                         color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.5)
    fig_opt_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
    st.plotly_chart(fig_opt_pie, use_container_width=True)
    
    st.markdown("""
    <div style='font-size: 0.9rem; color: #555; border-top: 1px solid #eee; padding-top: 10px;'>
    *학습 데이터: 코로나19, 러시아-우크라이나 전쟁, SVB 파산 사태의 T=0 ~ T+60 구간 변동성 최적화 모델.<br>
    *검증 기준: 미-이란 전쟁 발생일(T=0)에 기계적으로 위 비중을 투입 후 누적 수익률 추적.
    </div>
    """, unsafe_allow_html=True)

# --- [Section 4] Asset Allocation Radar ---
st.markdown("<div class='section-title'>🎯 [Section 4] Asset Allocation Radar: 종합 권고 제언</div>", unsafe_allow_html=True)
col_c1, col_c2 = st.columns([1, 2])
with col_c1:
    fig_pie = px.pie(values=list(current_sim['weights'].values()), names=list(current_sim['weights'].keys()), color_discrete_sequence=px.colors.qualitative.Bold, hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)
with col_c2:
    st.markdown("<div class='insight-box'>#### 🛡️ 자산 보강 액션 플랜")
    st.write(f"🔸 **현행 모델**: {current_sim['model_name']} (위험도 {risk_score:.1f}점 기반)")
    st.write(f"🔸 **핵심 근거**: 리스크 산출 공식에 따른 주식 비중 조절 및 안전자산 방어막 형성")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='font-size: 0.8rem; color: #666; margin-top: 3rem; text-align: center; border-top: 1px solid #ddd; padding-top: 10px;'>본 자산배분 시스템은 개인 연구 목적으로 제작되었으며 어떠한 투자 결과도 보장하지 않습니다. [Prime V7]</div>", unsafe_allow_html=True)
