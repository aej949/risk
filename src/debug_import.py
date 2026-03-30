import os
import sys

# src 디렉토리 경로 확보
target_dir = r"c:\Users\kanda\.gemini\antigravity\scratch\fcicb6-proj2\commodity_risk_analysis\src"
if target_dir not in sys.path:
    sys.path.insert(0, target_dir)

try:
    import risk_analyzer
    print(f"로드된 risk_analyzer 경로: {risk_analyzer.__file__}")
    print(f"존재하는 함수 목록: {dir(risk_analyzer)}")
except Exception as e:
    print(f"임포트 실패: {e}")
