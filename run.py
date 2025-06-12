import sys
import os

# 현재 스크립트 파일의 디렉토리 경로를 얻습니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Mesa 프로젝트의 루트 디렉토리를 계산합니다.
# 예를 들어, my_simulation이 mesa 바로 아래에 있다면, 두 번 상위 디렉토리로 이동해야 합니다.
# C:\Users\1\Desktop\workplace\code\mesa\my_simulation -> C:\Users\1\Desktop\workplace\code\mesa
mesa_root_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Mesa 프로젝트의 루트 디렉토리가 sys.path에 없으면 추가합니다.
if mesa_root_dir not in sys.path:
    sys.path.append(mesa_root_dir)

# 이제 model 모듈을 임포트합니다.
# 이전에 mesa 모듈을 찾지 못했던 문제도 해결될 것입니다.
from model import EconomicModel
from mesa.visualization.modules import ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import Slider
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 깨짐 문제 해결
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지

# --- 시각화 설정 ---

# 1. 차트 정의
chart1 = ChartModule([{"Label": "EmployedWorkers", "Color": "blue", "label": "고용된 노동자 수"},
                      {"Label": "TotalWorkers", "Color": "orange", "label": "총 노동자 수"}],
                     data_collector_name='datacollector')

chart2 = ChartModule([{"Label": "TotalProfit", "Color": "green", "label": "고용주 총 이익"}],
                     data_collector_name='datacollector')

chart3 = ChartModule([{"Label": "AvgWorkerDemand", "Color": "red", "label": "노동자 평균 요구 임금"},
                      {"Label": "AvgWorkerIncome", "Color": "purple", "label": "고용된 노동자 평균 수입"}],
                     data_collector_name='datacollector')

chart4 = ChartModule([{"Label": "EnvironmentK_AttributeA", "Color": "brown", "label": "환경 요인 K 속성 A"}],
                     data_collector_name='datacollector')

# 2. 모델 파라미터 정의 (사용자 UI)
model_params = {
    "num_workers": Slider("초기 노동자 수", 50, 1, 200, 1),
    "num_employers": Slider("초기 고용주 수", 10, 1, 50, 1),
    "x_percent_reduction": Slider("임금 차감 비율 (%)", 5.0, 0.0, 20.0, 0.1),
    "k_age_factor": Slider("연령 반영 계수", 2.0, 0.0, 10.0, 0.1),
    "k_distance_factor": Slider("거리 반영 계수 (km당)", 5.0, 0.0, 20.0, 0.1),
    "environment_event_frequency": Slider("환경 이벤트 주기", 10, 1, 50, 1),
    "population_change_rate": Slider("인구 변화율", 0.05, 0.0, 0.5, 0.01),

    # --- 아래 파라미터들은 UI에 노출되지 않는 고정값입니다 ---
    "initial_wage_P_min": 100.0,
    "initial_wage_P_max": 200.0,
    "real_estate_size_min": 1000.0,
    "real_estate_size_max": 5000.0,
    "production_rate_per_size": 0.1,
    "operating_cost_per_employee": 50.0,
    "production_contribution_per_worker": 10.0,
    "num_negotiation_attempts": 5,
}

# 3. 서버 설정
server = ModularServer(
    EconomicModel,
    [chart1, chart2, chart3, chart4],
    "경제 시뮬레이션 모델",
    model_params
)

# 서버 포트 설정 (기본값은 8521)
server.port = 8653

# --- 서버 실행 ---
if __name__ == "__main__":
    try:
        server.launch()
    except KeyboardInterrupt:
        print("서버를 종료합니다.")
