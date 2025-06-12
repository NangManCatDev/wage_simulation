import mesa
# Deprecated된 RandomActivation 대신 BaseScheduler를 임포트하는 대신,
# 사용자 정의 스케줄러를 직접 구현하여 사용합니다.
# from mesa.time import BaseScheduler 
from mesa.datacollection import DataCollector
from agents import Worker, Employer, EnvironmentFactor
import random
import numpy as np

# --- 사용자 정의 스케줄러 클래스 정의 ---
class CustomScheduler:
    """
    Mesa의 기본 스케줄러를 대체하는 사용자 정의 스케줄러입니다.
    에이전트 목록을 관리하고, 스텝 카운트를 유지합니다.
    에이전트의 실행 순서는 모델의 step 메서드에서 직접 제어합니다.
    """
    def __init__(self, model):
        self.model = model
        self.steps = 0 # 현재 스텝 카운터
        self._agents = {} # unique_id를 키로 하는 에이전트 딕셔너리

    @property
    def agents(self):
        """
        DataCollector 등에서 에이전트 목록에 접근할 수 있도록
        에이전트 객체의 리스트를 반환하는 프로퍼티입니다.
        """
        return list(self._agents.values())

    def add(self, agent):
        """모델에 에이전트를 추가합니다."""
        self._agents[agent.unique_id] = agent

    def remove(self, agent):
        """모델에서 에이전트를 제거합니다."""
        if agent.unique_id in self._agents:
            del self._agents[agent.unique_id]

    def get_agent_count(self):
        """현재 스케줄러에 있는 에이전트의 총 수를 반환합니다."""
        return len(self._agents)

    def step(self):
        """
        이 사용자 정의 스케줄러는 에이전트의 step() 메서드를 직접 호출하지 않습니다.
        에이전트의 활성화 순서는 모델의 step() 메서드에서 처리됩니다.
        여기서는 단순히 스텝 카운트를 증가시킵니다.
        """
        self.steps += 1

# --- 모델 클래스 정의 ---

class EconomicModel(mesa.Model):
    """
    행위자 간의 경제 활동 및 환경 영향을 시뮬레이션하는 모델입니다.
    """
    def __init__(self, num_workers, num_employers,
                 initial_wage_P_min=100.0, initial_wage_P_max=200.0,
                 real_estate_size_min=1000.0, real_estate_size_max=5000.0,
                 x_percent_reduction=5.0, # 피고용인 임금 차감 비율 (x%)
                 k_age_factor=5.0, # 연령 반영 계수
                 k_distance_factor=10.0, # 거리 반영 계수 (km 당)
                 production_rate_per_size=0.1, # 부동산 규모당 생산량 비율
                 operating_cost_per_employee=50.0, # 피고용자당 운영비
                 production_contribution_per_worker=10.0, # 노동자 1인당 생산 기여도
                 num_negotiation_attempts=5, # 임금 협상 총 시도 횟수
                 environment_event_frequency=10, # 환경 이벤트 발생 주기 (스텝 단위)
                 population_change_rate=0.05 # 환경 요인 K에 의한 인구 변화율 (비율)
                ):
        """
        모델 생성자입니다. 시뮬레이션의 초기 파라미터들을 설정합니다.

        Args:
            num_workers (int): 초기 노동자 수.
            num_employers (int): 초기 고용주 수.
            initial_wage_P_min (float): 최초 요구 임금 P의 최솟값.
            initial_wage_P_max (float): 최초 요구 임금 P의 최댓값.
            real_estate_size_min (float): 부동산 규모의 최솟값.
            real_estate_size_max (float): 부동산 규모의 최댓값.
            x_percent_reduction (float): 피고용인 임금 차감 비율 (%).
            k_age_factor (float): 연령 반영 계수.
            k_distance_factor (float): 거리 반영 계수 (km 당).
            production_rate_per_size (float): 부동산 규모당 생산량 비율.
            operating_cost_per_employee (float): 피고용자당 운영비.
            production_contribution_per_worker (float): 노동자 1인당 생산 기여도.
            num_negotiation_attempts (int): 임금 협상 총 시도 횟수.
            environment_event_frequency (int): 환경 이벤트 발생 주기 (스텝 단위).
            population_change_rate (float): 환경 요인 K에 의한 인구 변화율 (비율).
        """
        self.num_workers = num_workers
        self.num_employers = num_employers
        self.x_percent_reduction = x_percent_reduction
        self.k_age_factor = k_age_factor
        self.k_distance_factor = k_distance_factor
        self.production_rate_per_size = production_rate_per_size
        self.operating_cost_per_employee = operating_cost_per_employee
        self.production_contribution_per_worker = production_contribution_per_worker
        self.num_negotiation_attempts = num_negotiation_attempts
        self.environment_event_frequency = environment_event_frequency
        self.population_change_rate = population_change_rate

        # 사용자 정의 스케줄러를 사용합니다.
        self.schedule = CustomScheduler(self) 
        self.running = True # 시뮬레이션 실행 상태
        self.current_id = 0 # 행위자 ID 부여를 위한 카운터

        # 데이터 수집기 설정
        self.datacollector = DataCollector(
            model_reporters={
                "TotalWorkers": lambda m: len([a for a in m.schedule.agents if isinstance(a, Worker)]),
                "EmployedWorkers": lambda m: len([a for a in m.schedule.agents if isinstance(a, Worker) and a.is_employed]),
                "TotalEmployers": lambda m: len([a for a in m.schedule.agents if isinstance(a, Employer)]),
                "TotalProfit": lambda m: sum(a.profit for a in m.schedule.agents if isinstance(a, Employer)),
                "AvgWorkerIncome": lambda m: np.mean([a.current_income for a in m.schedule.agents if isinstance(a, Worker) and a.is_employed]) if any(isinstance(a, Worker) and a.is_employed for a in m.schedule.agents) else 0,
                "AvgWorkerDemand": lambda m: np.mean([a.current_wage_demand for a in m.schedule.agents if isinstance(a, Worker)]) if any(isinstance(a, Worker) for a in m.schedule.agents) else 0,
                "EnvironmentK_AttributeA": lambda m: [a.attribute_a for a in m.schedule.agents if isinstance(a, EnvironmentFactor) and a.factor_type == 'K'][0] if any(isinstance(a, EnvironmentFactor) and a.factor_type == 'K' for a in m.schedule.agents) else 0
            },
            agent_reporters={
                "ID": "unique_id",
                "Type": lambda a: type(a).__name__,
                "IsEmployed": lambda a: a.is_employed if isinstance(a, Worker) else None,
                "CurrentWageDemand": lambda a: a.current_wage_demand if isinstance(a, Worker) else None,
                "Profit": lambda a: a.profit if isinstance(a, Employer) else None,
                "AttributeA": lambda a: a.attribute_a if isinstance(a, EnvironmentFactor) else None
            }
        )

        # --- 행위자 생성 ---
        # 노동자 생성
        for i in range(self.num_workers):
            self.current_id += 1
            age = random.randint(20, 60) # 연령 범위
            distance = random.uniform(1, 20) # 거리 범위 (km)
            initial_P = random.uniform(initial_wage_P_min, initial_wage_P_max)
            worker = Worker(self.current_id, self, initial_P, age, distance)
            self.schedule.add(worker)

        # 고용주 생성
        for i in range(self.num_employers):
            self.current_id += 1
            real_estate_size = random.uniform(real_estate_size_min, real_estate_size_max)
            # 초기 이익 목표는 임의로 설정
            initial_profit_target = random.uniform(5000, 15000)
            employer = Employer(self.current_id, self, real_estate_size, initial_profit_target)
            self.schedule.add(employer)

        # 환경 요인 생성 (K와 J)
        self.current_id += 1
        env_K = EnvironmentFactor(self.current_id, self, 'K', initial_attribute_a=random.uniform(0.1, 0.9))
        self.schedule.add(env_K)

        self.current_id += 1
        env_J = EnvironmentFactor(self.current_id, self, 'J', initial_attribute_a=random.uniform(0.1, 0.9))
        self.schedule.add(env_J)

        # '상황 인식' 요소들을 모델 속성으로 추가 (현재 고민 중이므로 기본값)
        self.rumor_level = 0.0
        self.information_flow = 0.0
        self.culture_influence = 0.0
        self.religion_influence = 0.0

    def adjust_agent_group_size(self, delta):
        """
        환경 요인 K의 발현으로 인한 행위자 그룹 수 조정을 처리합니다.
        문서의 '특정 속성을 가진 환경 요인 K의 발현(행사)으로 - 행위자 그룹의 수는 줄어든다(늘어난다).' 반영.
        """
        # 노동자 그룹의 수를 조정 (예시)
        # self.schedule.agents를 사용하여 현재 모든 에이전트 목록을 가져옵니다.
        num_workers_to_change = int(len([a for a in self.schedule.agents if isinstance(a, Worker)]) * abs(delta))
        
        if delta < 0: # 감소
            for _ in range(num_workers_to_change):
                worker_agents = [a for a in self.schedule.agents if isinstance(a, Worker)]
                if worker_agents:
                    worker_to_remove = random.choice(worker_agents)
                    self.schedule.remove(worker_to_remove)
                    # print(f"노동자 {worker_to_remove.unique_id}가 시뮬레이션에서 제거되었습니다 (환경 요인 K 영향).")
        elif delta > 0: # 증가
            for _ in range(num_workers_to_change):
                self.current_id += 1
                age = random.randint(20, 60)
                distance = random.uniform(1, 20)
                initial_P = random.uniform(100.0, 200.0) # 기본 범위 사용
                new_worker = Worker(self.current_id, self, initial_P, age, distance)
                self.schedule.add(new_worker)
                # print(f"새로운 노동자 {new_worker.unique_id}가 시뮬레이션에 추가되었습니다 (환경 요인 K 영향).")


    def step(self):
        """
        시뮬레이션의 한 스텝을 진행합니다.
        모든 행위자의 step() 메서드를 호출하고, 모델 수준의 로직을 처리합니다.
        """
        # Mesa의 Deprecated된 스케줄러 대신, CustomScheduler를 통해 관리되는 에이전트 목록을 가져와 직접 스텝을 수행합니다.
        # RandomActivation의 행동을 모방하기 위해 에이전트 목록을 무작위로 섞습니다.
        agent_list = list(self.schedule.agents)
        self.random.shuffle(agent_list) # 모델의 Random 객체를 사용하여 섞습니다.

        for agent in agent_list:
            agent.step()
        
        # 사용자 정의 스케줄러의 스텝 카운터를 수동으로 증가시킵니다.
        self.schedule.step() # CustomScheduler의 step() 메서드 호출

        # 모델 수준에서 관계 처리 (일반 관계 5, 6, 7, 8번 등)
        # 5. 행위자 A가 특정 행위(가격 조정, 고용자 수 등)를 하면, 행위자 B는 특정 공간에 소속된다.
        #    => 고용주가 노동자를 고용하면, 노동자는 해당 고용주(의 공간)에 소속된 것으로 간주 (is_employed, employer_id 업데이트로 구현)
        # 6. 행위자 A가 특정 행위(가격 조정, 고용자 수 등)를 하면, 행위자 B는 특정 일(생산, 피고용 상태 등)을 한다.
        #    => 고용주가 노동자를 고용하면, 노동자는 'work' 메서드를 통해 생산에 기여 (Agent.step() 내에서 처리)
        # 7. 행위자 B가 특정 공간에 소속되어 특정 일(생산)을 한다.
        #    => is_employed와 work() 메서드 호출로 구현

        # 8. 유동 자산의 소유(고용주의 경우 생산량, 노동자의 경우 임금)를 소비하는 경우 행위자의 속성에 따르며, 속성에 따라 그 소비의 정도가 다를 수 있다.
        #    => Worker.consume_assets()와 Employer.pay_wage_and_calculate_profit() 내에서 처리

        self.datacollector.collect(self) # 데이터 수집

        # 시뮬레이션 종료 조건 (예시: 특정 스텝 수 도달)
        if self.schedule.steps >= 100: # 100 스텝 후 종료
            self.running = False


# --- 데이터 수집 및 실행 관련 헬퍼 함수 (선택 사항, 서버/분석에 사용) ---

def get_total_profit(model):
    """모든 고용주의 총 이익을 반환합니다."""
    return sum(a.profit for a in model.schedule.agents if isinstance(a, Employer))

def get_total_employment(model):
    """고용된 노동자의 총 수를 반환합니다."""
    return len([a for a in model.schedule.agents if isinstance(a, Worker) and a.is_employed])