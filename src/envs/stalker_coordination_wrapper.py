import numpy as np
from .multiagentenv import MultiAgentEnv


class StalkerCoordinationRewardWrapper(MultiAgentEnv):
    """
    Stalker가 동일한 적을 공격할 때 추가 리워드를 주는 래퍼
    """
    
    def __init__(self, env, coordination_reward=0.1, stalker_role_id=0):
        """
        Args:
            env: 원본 환경
            coordination_reward: 동일한 적을 공격할 때의 추가 리워드
            stalker_role_id: Stalker의 role ID
        """
        self.env = env
        self.coordination_reward = coordination_reward
        self.stalker_role_id = stalker_role_id
        
        # 에피소드 정보
        self.n_agents = env.n_agents
        self.episode_limit = env.episode_limit
        
        # 타겟 추적을 위한 변수들
        self.last_actions = None
        self.target_counts = {}  # 각 타겟에 대한 공격 횟수 추적
        
    def step(self, actions):
        """액션을 실행하고 추가 리워드를 계산"""
        # 원본 환경에서 스텝 실행
        obss, reward, terminated, truncated, info = self.env.step(actions)
        
        # 추가 리워드 계산
        additional_reward = self._calculate_coordination_reward(actions)
        
        # 리워드에 추가 리워드 더하기
        if isinstance(reward, (int, float)):
            reward += additional_reward
        elif isinstance(reward, list):
            reward = [r + additional_reward for r in reward]
        
        # 액션 저장 (다음 스텝에서 사용)
        self.last_actions = actions
        
        return obss, reward, terminated, truncated, info
    
    def _calculate_coordination_reward(self, actions):
        """Stalker의 협력 공격 리워드 계산"""
        try:
            # 현재 관찰에서 적 정보 추출
            obs = self.env.get_obs()
            if not obs:
                return 0.0
            
            # 타겟 정보 추출
            current_targets = self._extract_targets_from_obs(obs, actions)
            
            if not current_targets:
                return 0.0
            
            # Stalker가 동일한 타겟을 공격하는지 확인
            coordination_bonus = 0.0
            
            # 각 타겟에 대해 공격하는 Stalker 수 계산
            target_attackers = {}
            stalker_actions = []
            
            for agent_id, action in enumerate(actions):
                if self._is_stalker(agent_id) and self._is_attack_action(action):
                    target = current_targets[agent_id]
                    stalker_actions.append((agent_id, action, target))
                    
                    if target is not None:
                        if target not in target_attackers:
                            target_attackers[target] = []
                        target_attackers[target].append(agent_id)
            
            # 동일한 타겟을 공격하는 Stalker가 2명 이상이면 보너스
            for target, attackers in target_attackers.items():
                if len(attackers) >= 2:
                    # 공격자 수에 비례한 보너스 (2명이면 1배, 3명이면 2배, ...)
                    bonus_multiplier = len(attackers) - 1
                    
                    # 추가적인 협력 보너스: 더 많은 Stalker가 협력할수록 더 큰 보너스
                    if len(attackers) >= 3:
                        # 3명 이상이 협력하면 추가 보너스
                        bonus_multiplier += 0.5
                    
                    coordination_bonus += self.coordination_reward * bonus_multiplier
            
            # 디버깅 정보 출력 (선택적)
            if coordination_bonus > 0:
                print(f"Coordination reward: {coordination_bonus:.3f}, "
                      f"Targets: {list(target_attackers.keys())}, "
                      f"Attackers per target: {[len(attackers) for attackers in target_attackers.values()]}")
            
            return coordination_bonus
            
        except Exception as e:
            print(f"Warning: Could not calculate coordination reward: {e}")
            return 0.0
    
    def _extract_targets_from_obs(self, obs, actions):
        """관찰에서 타겟 정보 추출"""
        targets = []
        
        for agent_id, (agent_obs, action) in enumerate(zip(obs, actions)):
            if self._is_attack_action(action):
                # 간단한 휴리스틱: 관찰의 특정 차원을 타겟 ID로 사용
                # 실제로는 더 정교한 방법이 필요할 수 있음
                try:
                    if hasattr(agent_obs, 'numpy'):
                        obs_array = agent_obs.numpy()
                    elif hasattr(agent_obs, 'cpu'):
                        obs_array = agent_obs.cpu().numpy()
                    else:
                        obs_array = np.array(agent_obs)
                    
                    # 관찰에서 타겟 정보 추출 (예: 적의 위치나 ID)
                    # 이 부분은 실제 SMAC 관찰 구조에 맞게 조정 필요
                    target_id = self._extract_target_id_from_obs(obs_array, action)
                    targets.append(target_id)
                    
                except Exception as e:
                    targets.append(None)
            else:
                targets.append(None)
        
        return targets
    
    def _extract_target_id_from_obs(self, obs_array, action):
        """개별 관찰에서 타겟 ID 추출"""
        try:
            # SMAC에서 공격 액션은 보통 적의 인덱스를 나타냄
            # 액션 ID에서 타겟 정보 추출
            if isinstance(action, (int, np.integer)):
                # 액션이 타겟 ID를 직접 나타내는 경우
                # SMAC에서 액션 0은 "아무것도 하지 않음", 1~n은 적 유닛 공격
                if action > 0:  # 공격 액션인 경우
                    # SMAC에서는 액션 ID가 적 유닛의 인덱스를 나타냄
                    # 하지만 실제로는 관찰에서 적 정보를 추출해야 함
                    return self._find_target_from_action(obs_array, action)
            elif isinstance(action, np.ndarray):
                if len(action) > 0 and action[0] > 0:
                    return self._find_target_from_action(obs_array, action[0])
            
            return None
            
        except Exception:
            return None
    
    def _find_target_from_action(self, obs_array, action_id):
        """액션 ID로부터 실제 타겟 유닛 찾기"""
        try:
            # SMAC 관찰 구조에서 적 유닛 정보 추출
            # 관찰은 보통 [자신의 정보, 아군 정보, 적 정보, 지도 정보] 순서로 구성됨
            
            # 간단한 휴리스틱: 관찰의 중간 부분에서 적 정보 추출
            obs_len = len(obs_array)
            
            # 관찰에서 적 정보가 있는 부분 추정 (실제로는 SMAC 관찰 구조를 정확히 알아야 함)
            # 일반적으로 관찰의 후반부에 적 정보가 있음
            enemy_info_start = obs_len // 2
            enemy_info_end = min(enemy_info_start + 20, obs_len)  # 적 정보는 보통 20차원 정도
            
            if enemy_info_end <= obs_len:
                enemy_info = obs_array[enemy_info_start:enemy_info_end]
                
                # 적이 있는지 확인 (0이 아닌 값이 있으면 적이 있음)
                if np.any(enemy_info != 0):
                    # 액션 ID에 해당하는 적 찾기
                    # 간단한 휴리스틱: 액션 ID를 적 정보 인덱스로 사용
                    if action_id <= len(enemy_info):
                        return action_id
                    
                    # 또는 가장 가까운 적의 ID 반환
                    return self._find_nearest_enemy_id(obs_array)
            
            return None
            
        except Exception:
            return None
    
    def _find_nearest_enemy_id(self, obs_array):
        """관찰에서 가장 가까운 적의 ID 찾기"""
        try:
            # 간단한 휴리스틱: 관찰의 특정 부분에서 적 정보 추출
            # 실제로는 SMAC의 관찰 구조를 정확히 알아야 함
            
            # 예: 관찰의 중간 부분에서 적 정보 추출
            if len(obs_array) > 10:
                # 관찰의 특정 범위에서 적 정보 추출
                enemy_info_start = len(obs_array) // 2
                enemy_info_end = enemy_info_start + 5
                
                if enemy_info_end <= len(obs_array):
                    enemy_info = obs_array[enemy_info_start:enemy_info_end]
                    # 적이 있는지 확인 (0이 아닌 값이 있으면 적이 있음)
                    if np.any(enemy_info != 0):
                        # 가장 큰 값을 적 ID로 사용
                        return int(np.argmax(enemy_info))
            
            return None
            
        except Exception:
            return None
    
    def _is_stalker(self, agent_id):
        """에이전트가 Stalker인지 확인"""
        # Role embedding이 활성화된 경우 role ID로 확인
        if hasattr(self, 'agent_roles') and self.agent_roles is not None:
            return self.agent_roles[agent_id] == self.stalker_role_id
        
        # 기본적으로 맵에 따라 Stalker 에이전트 식별
        # 일반적인 SMAC 맵에서 Stalker는 보통 앞쪽 인덱스에 위치
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'get_env_info'):
            env_info = self.env.env.get_env_info()
            map_name = getattr(self.env.env, 'map_name', '')
            
            # 맵별 Stalker 에이전트 인덱스 정의
            stalker_indices = self._get_stalker_indices(map_name)
            if stalker_indices:
                return agent_id in stalker_indices
        
        # 기본적으로 모든 에이전트를 Stalker로 간주 (테스트용)
        return True
    
    def _get_stalker_indices(self, map_name):
        """맵별 Stalker 에이전트 인덱스 반환"""
        stalker_map_configs = {
            '2s3z': [0, 1],  # 2 Stalkers, 3 Zealots
            '3s5z': [0, 1, 2],  # 3 Stalkers, 5 Zealots
            '3s5z_vs_3s6z': [0, 1, 2],  # 3 Stalkers, 5 Zealots vs 3 Stalkers, 6 Zealots
            '8m': [],  # Marines only
            '8m_vs_9m': [],  # Marines only
            '5m_vs_6m': [],  # Marines only
            '10m_vs_11m': [],  # Marines only
            '27m_vs_30m': [],  # Marines only
            'MMM': [],  # Marines, Marauders, Medivacs
            'MMM2': [],  # Marines, Marauders, Medivacs
            '2s_vs_1sc': [0, 1],  # 2 Stalkers vs 1 Stalker, 1 Colossus
            '3s_vs_3z': [0, 1, 2],  # 3 Stalkers vs 3 Zealots
            '3s_vs_4z': [0, 1, 2],  # 3 Stalkers vs 4 Zealots
            '3s_vs_5z': [0, 1, 2],  # 3 Stalkers vs 5 Zealots
        }
        
        return stalker_map_configs.get(map_name, [])
    
    def _is_attack_action(self, action):
        """액션이 공격 액션인지 확인"""
        try:
            if isinstance(action, (int, np.integer)):
                # SMAC에서 공격 액션은 보통 0보다 큰 값
                return action > 0
            elif isinstance(action, np.ndarray):
                return len(action) > 0 and action[0] > 0
            return False
        except Exception:
            return False
    
    def reset(self, seed=None, options=None):
        """에피소드 리셋"""
        self.last_actions = None
        self.target_counts = {}
        return self.env.reset(seed, options)
    
    def get_obs(self):
        return self.env.get_obs()
    
    def get_obs_agent(self, agent_id):
        return self.env.get_obs_agent(agent_id)
    
    def get_obs_size(self):
        return self.env.get_obs_size()
    
    def get_state(self):
        return self.env.get_state()
    
    def get_state_size(self):
        return self.env.get_state_size()
    
    def get_avail_actions(self):
        return self.env.get_avail_actions()
    
    def get_avail_agent_actions(self, agent_id):
        return self.env.get_avail_agent_actions(agent_id)
    
    def get_total_actions(self):
        return self.env.get_total_actions()
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()
    
    def seed(self, seed=None):
        return self.env.seed(seed)
    
    def save_replay(self):
        return self.env.save_replay()
    
    def get_env_info(self):
        return self.env.get_env_info()
    
    def get_stats(self):
        return self.env.get_stats()
    
    def set_agent_roles(self, agent_roles):
        """에이전트 역할 설정"""
        self.agent_roles = agent_roles
