import mesa
import random
import numpy as np

# --- 행위자 클래스 정의 ---

class Worker(mesa.Agent):
    """
    노동자 행위자를 나타냅니다.
    임금 협상, 근로, 자산 소비 등의 행동을 수행합니다.
    """
    def __init__(self, unique_id, model, initial_wage_P, age, distance_to_work):
        """
        Worker 행위자의 생성자입니다.

        Args:
            unique_id (int): 행위자의 고유 ID.
            model (mesa.Model): 시뮬레이션 모델 인스턴스.
            initial_wage_P (float): 최초 요구 임금 P.
            age (int): 노동자의 연령.
            distance_to_work (float): 농장/주거지와의 거리 (km).
        """
        super().__init__(unique_id, model)
        self.initial_wage_P = initial_wage_P
        self.negotiation_attempts_left = model.num_negotiation_attempts # 문서에 명시된 총 시도 횟수 (예: 5회)
        self.age = age
        self.distance_to_work = distance_to_work
        self.current_wage_demand = self.calculate_initial_demand() # 현재 요구 임금
        self.is_employed = False # 고용 상태
        self.employer_id = None # 고용된 고용주의 ID
        self.current_income = 0.0 # 현재 수입 (고용 후 받을 임금)
        self.assets = {'liquid': random.uniform(500, 2000)} # 유동 자산 (예: 월급을 받을 때마다 증가)
        self.satisfaction = 0.5 # 만족도 (향후 확장 가능)

        # 행위자의 다양한 속성 (문서 9번 항목)
        self.attributes = {
            'productivity': random.uniform(0.8, 1.2),
            'flexibility': random.uniform(0.3, 0.7)
        }
        # 속성 간의 관계 설정 (예시: 생산성이 활성화되면 유연성도 활성화될 수 있음)
        # 이 부분은 step 메서드 내에서 로직으로 구현될 수 있습니다.

    def calculate_initial_demand(self):
        """
        노동자의 개별 속성(연령, 거리)을 반영하여 초기 임금 요구액을 계산합니다.
        문서의 '개별 속성'과 '연령: 30살을 기준으로 표준 분포표를 적용',
        '농장과 주거지와의 거리: 거리가 멀수록 요구 금액이 높아진다. km 당 반영액' 반영.
        """
        base_demand = self.initial_wage_P

        # 연령 반영: 30세를 기준으로 표준 분포표 적용 (간략화된 예시)
        # 실제 표준 분포 함수는 scipy.stats.norm 등을 사용할 수 있습니다.
        age_deviation = abs(self.age - 30)
        base_demand += age_deviation * self.model.k_age_factor # 나이 편차에 따른 조정

        # 거리에 따른 반영: 거리가 멀수록 요구액 증가
        base_demand += self.distance_to_work * self.model.k_distance_factor # km 당 반영액 k_distance_factor

        return base_demand

    def negotiate_wage(self):
        """
        고용주에게 임금을 요구하는 협상 로직을 구현합니다.
        문서의 'n차 시도에서 받고자 하는 임금의 수준' 공식 w_n = P * (1 - x/100)^(n-1) 반영.
        """
        if self.is_employed:
            return self.current_wage_demand # 이미 고용된 경우 협상하지 않음

        if self.negotiation_attempts_left > 0:
            # P는 1차 시도에서 w가 되는 initial_wage_P
            # n은 총 시도 횟수에서 남은 시도 횟수를 뺀 값 + 1
            n_attempt = self.model.num_negotiation_attempts - self.negotiation_attempts_left + 1
            
            # w_n = initial_wage_P * (1 - x/100)^(n-1)
            self.current_wage_demand = self.initial_wage_P * \
                                       (1 - self.model.x_percent_reduction / 100)**(n_attempt - 1)
            self.negotiation_attempts_left -= 1
            return self.current_wage_demand
        else:
            return 0 # 모든 협상 시도 실패

    def work(self):
        """
        고용주에게 고용된 경우 일을 수행하는 로직을 구현합니다.
        '행위자가 일을 한다는 사실에서 일은 생산과 같이 수치로 환원할 수 있다.' 반영.
        """
        if self.is_employed and self.employer_id is not None:
            # 고용주를 찾아 생산 활동에 기여 (모델에서 고용주가 이를 취합)
            # 생산량은 Worker의 생산성 속성에도 영향을 받을 수 있습니다.
            return self.attributes['productivity']
        return 0

    def consume_assets(self):
        """
        유동 자산을 소비하는 로직을 구현합니다.
        '유동 자산의 소유(고용주의 경우 생산량, 노동자의 경우 임금)를 소비하는 경우 행위자의 속성에 따르며, 속성에 따라 그 소비의 정도가 다를 수 있다.' 반영.
        """
        consumption_rate = 0.1 # 예시 소비율
        # 소비 정도는 행위자의 속성(예: 생활 수준, 가족 구성 등)에 따라 달라질 수 있습니다.
        actual_consumption = self.assets['liquid'] * consumption_rate
        self.assets['liquid'] -= actual_consumption
        if self.assets['liquid'] < 0:
            self.assets['liquid'] = 0 # 자산이 0 미만으로 내려가지 않도록
        return actual_consumption

    def step(self):
        """
        매 스텝마다 노동자가 수행하는 행동입니다.
        """
        # 현재 상태에 따라 행동 결정
        if not self.is_employed:
            # 고용되지 않았다면 임금 협상 시도
            self.negotiate_wage()
        else:
            # 고용되었다면 일 수행 (생산 기여)
            self.work()
            # 임금 소비
            self.consume_assets()
            # (선택 사항) 만족도 변화 로직 등 추가

        # 속성 간의 관계 형성 (문서 9번 항목)
        # 예시: 생산성 속성이 특정 임계값을 넘으면 유연성 속성이 활성화(높아짐)
        if self.attributes['productivity'] > 1.0 and self.attributes['flexibility'] < 1.0:
            self.attributes['flexibility'] += 0.05 # 활성화 예시


class Employer(mesa.Agent):
    """
    고용주 행위자를 나타냅니다.
    부동산을 소유하고, 생산량을 결정하며, 피고용자를 고용하고 임금을 지급하며 이익을 극대화합니다.
    """
    def __init__(self, unique_id, model, real_estate_size, initial_profit_target):
        """
        Employer 행위자의 생성자입니다.

        Args:
            unique_id (int): 행위자의 고유 ID.
            model (mesa.Model): 시뮬레이션 모델 인스턴스.
            real_estate_size (float): 고용주가 소유한 부동산(농장, 공장 등)의 규모.
            initial_profit_target (float): 사건 이전의 이익 목표.
        """
        super().__init__(unique_id, model)
        self.real_estate_size = real_estate_size
        self.initial_profit_target = initial_profit_target # 사건 이전 이익 목표
        self.production_capacity = self.calculate_production() # 부동산 규모에 따른 생산량
        self.num_employees_hired = 0 # 현재 고용한 피고용자 수
        self.profit = 0.0 # 현재 이익
        self.operating_cost_per_employee = model.operating_cost_per_employee # 피고용자당 운영비
        self.assets = {'liquid': random.uniform(5000, 20000), 'real_estate': real_estate_size} # 유동 자산 및 부동산 자산

    def calculate_production(self):
        """
        부동산 규모에 따라 생산량을 결정하는 로직을 구현합니다.
        """
        # 문서: '부동산의 규모에 따라 생산량이 결정된다.'
        # 단순 비례 관계로 가정, 향후 더 복잡한 함수로 확장 가능
        return self.real_estate_size * self.model.production_rate_per_size

    def determine_employees_and_wage(self, available_workers):
        """
        최대 이익을 기준으로 고용할 피고용자 수와 임금을 결정합니다.
        문서: '고용주의 최대 이익을 결정하는 수준에서 임금과 피고용인의 수를 결정한다.' (적분 함수의 최댓값)
        환경 변화(인구수 감소 등)로 인한 최적화도 고려.

        Args:
            available_workers (list): 고용 가능한 Worker Agent 목록.

        Returns:
            tuple: (고용할 노동자 수, 제안 임금)
        """
        # --- 단순화된 이익 극대화 로직 (적분 함수 최적화 대신) ---
        # 실제 이익 극대화는 생산 함수, 비용 함수, 노동 시장 등을 복합적으로 고려해야 합니다.
        # 여기서는 고용 가능한 노동자 중 가장 낮은 요구 임금을 가진 노동자들을 우선적으로 고용하되,
        # 목표 생산량을 달성할 수 있는 수준에서 결정합니다.

        # 이용 가능한 노동자들을 요구 임금이 낮은 순으로 정렬
        sorted_workers = sorted(available_workers, key=lambda w: w.current_wage_demand)

        best_num_employees = 0
        best_wage_offer = 0
        max_profit_found = -float('inf')

        # 잠재적 고용 시나리오를 탐색
        for i in range(1, len(sorted_workers) + 1):
            potential_employees = sorted_workers[:i]
            total_wage_demand = sum(w.current_wage_demand for w in potential_employees)
            
            # 가상의 생산량 계산 (각 노동자가 일정 생산량을 기여한다고 가정)
            # 실제로는 각 노동자의 생산성 속성(worker.attributes['productivity'])을 반영해야 함
            total_production = self.calculate_production() * (i / len(available_workers) if len(available_workers) > 0 else 0)

            # 운영비 계산 (피고용자당 운영비)
            total_operating_cost = i * self.operating_cost_per_employee

            # 이익 계산 (가격은 임의로 100으로 가정)
            current_profit = (total_production * 100) - total_wage_demand - total_operating_cost

            if current_profit > max_profit_found:
                max_profit_found = current_profit
                best_num_employees = i
                # 제안 임금은 고용한 노동자들의 평균 요구 임금으로 설정
                best_wage_offer = total_wage_demand / i if i > 0 else 0

        return best_num_employees, best_wage_offer

    def hire_workers(self, num_to_hire, offered_wage, available_workers):
        """
        실제로 노동자를 고용하는 로직을 구현합니다.
        Args:
            num_to_hire (int): 고용할 노동자 수.
            offered_wage (float): 고용주가 제안하는 임금.
            available_workers (list): 고용 가능한 Worker Agent 목록.
        Returns:
            list: 실제로 고용된 Worker Agent 목록.
        """
        hired_workers = []
        # 요구 임금이 낮은 순으로 정렬된 노동자 목록
        sorted_workers = sorted(available_workers, key=lambda w: w.current_wage_demand)

        for worker in sorted_workers:
            if len(hired_workers) >= num_to_hire:
                break
            # 고용주의 제안 임금이 노동자의 현재 요구 임금보다 높거나 같으면 고용
            if offered_wage >= worker.current_wage_demand and not worker.is_employed:
                worker.is_employed = True
                worker.employer_id = self.unique_id
                worker.current_income = offered_wage # 계약된 임금
                self.num_employees_hired += 1
                hired_workers.append(worker)
                self.model.schedule.remove(worker) # 기존 스케줄러에서 제거
                self.model.schedule.add(worker) # 새로운 순서로 추가 또는 상태 업데이트
        return hired_workers

    def pay_wage_and_calculate_profit(self, hired_workers):
        """
        고용된 노동자에게 임금을 지급하고 이익을 계산합니다.
        """
        total_wages_paid = sum(w.current_income for w in hired_workers)
        total_operating_costs = self.num_employees_hired * self.operating_cost_per_employee

        # 총 생산량 (고용된 노동자들의 기여를 합산)
        # 각 노동자의 생산성 속성(worker.attributes['productivity'])을 반영
        total_production_by_workers = sum(worker.attributes['productivity'] for worker in hired_workers)
        total_production = self.production_capacity + (total_production_by_workers * self.model.production_contribution_per_worker)

        # 이익 계산 (판매 가격을 100으로 가정)
        self.profit = (total_production * 100) - total_wages_paid - total_operating_costs
        self.assets['liquid'] += self.profit # 이익만큼 유동자산 증가

        # 노동자 자산 업데이트
        for worker in hired_workers:
            worker.assets['liquid'] += worker.current_income # 노동자에게 임금 지급

    def step(self):
        """
        매 스텝마다 고용주가 수행하는 행동입니다.
        """
        # 고용 가능한 노동자들을 모델에서 가져옴
        available_workers = [
            agent for agent in self.model.schedule.agents
            if isinstance(agent, Worker) and not agent.is_employed
        ]

        # 고용할 노동자 수와 제안 임금 결정
        num_to_hire, offered_wage = self.determine_employees_and_wage(available_workers)

        # 실제로 노동자 고용
        hired_workers = self.hire_workers(num_to_hire, offered_wage, available_workers)

        # 고용된 노동자들에게 임금 지급 및 이익 계산
        self.pay_wage_and_calculate_profit(hired_workers)


class EnvironmentFactor(mesa.Agent):
    """
    환경 요인 K와 J를 나타냅니다.
    특정 속성을 가지며, 발현 시 다른 행위자나 환경 요인에 영향을 줄 수 있습니다.
    """
    def __init__(self, unique_id, model, factor_type, initial_attribute_a):
        """
        EnvironmentFactor 행위자의 생성자입니다.

        Args:
            unique_id (int): 행위자의 고유 ID.
            model (mesa.Model): 시뮬레이션 모델 인스턴스.
            factor_type (str): 환경 요인의 유형 (예: 'K', 'J', '소문', '정보').
            initial_attribute_a (float): 환경 요인의 특정 속성 a.
        """
        super().__init__(unique_id, model)
        self.factor_type = factor_type
        self.attribute_a = initial_attribute_a # 특정 속성 a
        self.activated = False # 활성화 상태 (발현 여부)
        self.influence_strength = 0.1 # 다른 행위자/환경에 미치는 영향 강도

    def activate(self):
        """환경 요인을 활성화(발현)시킵니다."""
        self.activated = True
        # print(f"환경 요인 {self.factor_type}이 활성화되었습니다.")

    def deactivate(self):
        """환경 요인을 비활성화시킵니다."""
        self.activated = False
        # print(f"환경 요인 {self.factor_type}이 비활성화되었습니다.")

    def apply_influence(self):
        """
        활성화된 환경 요인이 다른 행위자나 환경 요인에 영향을 미칩니다.
        문서의 '일반 관계' 1, 2, 3, 4번 항목을 반영합니다.
        """
        if self.activated:
            # 1. 특정 속성을 가진 환경 요인 K의 발현으로 - 행위자 그룹의 수는 줄어든다(늘어난다).
            if self.factor_type == 'K':
                if self.attribute_a > 0.7: # 예시 조건
                    self.model.adjust_agent_group_size(delta=-self.model.population_change_rate) # 인구수 감소
                elif self.attribute_a < 0.3:
                    self.model.adjust_agent_group_size(delta=self.model.population_change_rate) # 인구수 증가

            # 2. 특정 속성을 가진 환경 요인 K의 발현으로 – 다른 환경 요인 J의 특정 속성에 a의 수치를 높인다(줄인다).
            if self.factor_type == 'K':
                for agent in self.model.schedule.agents:
                    if isinstance(agent, EnvironmentFactor) and agent.factor_type == 'J':
                        agent.attribute_a += self.influence_strength # J의 속성 a 증가
                        # print(f"환경 요인 J의 속성 a가 {agent.attribute_a}로 변경되었습니다.")
                        break

            # 3. 특정 속성을 가진 환경 요인 K의 발현으로 – 행위자 A의 특정 속성 a의 수치를 높인다(줄인다, 활성화 한다).
            if self.factor_type == 'K':
                # 모든 Worker의 특정 속성 (예: 만족도)에 영향
                for agent in self.model.schedule.agents:
                    if isinstance(agent, Worker):
                        agent.satisfaction -= self.influence_strength / 2 # 만족도 감소 예시

            # 4. 환경 요인 K의 속성 a가 환경 요인 K를 활성화(강화 혹은 약화)한다.
            # 이 로직은 주로 step() 메서드 내에서 attribute_a 값에 따라 activate/deactivate를 호출합니다.
            pass

    def step(self):
        """
        매 스텝마다 환경 요인이 수행하는 행동입니다.
        """
        # 환경 요인 K의 속성 a에 따라 활성화/비활성화 상태를 결정
        # 이 부분은 '환경 요인 K의 속성 a가 환경 요인 K를 활성화(강화 혹은 약화)한다.'를 반영
        if self.factor_type == 'K':
            if self.model.schedule.steps % self.model.environment_event_frequency == 0: # N 스텝마다 이벤트 발생
                if random.random() < self.attribute_a: # 속성 a가 높으면 활성화될 확률 높음
                    self.activate()
                else:
                    self.deactivate()

        # 활성화된 경우 영향 적용
        self.apply_influence()

        # (선택 사항) 환경 요인 자체의 속성 변화 로직 (시간 흐름에 따른 자연적 변화 등)