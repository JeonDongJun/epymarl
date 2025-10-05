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
            
            # 타겟 정보 추출 (간단한 휴리스틱)
            current_targets = self._extract_targets_from_obs(obs, actions)
            
            if not current_targets:
                return 0.0
            
            # Stalker가 동일한 타겟을 공격하는지 확인
            coordination_bonus = 0.0
            
            # 각 타겟에 대해 공격하는 Stalker 수 계산
            target_attackers = {}
            for agent_id, action in enumerate(actions):
                if self._is_stalker(agent_id) and self._is_attack_action(action):
                    target = current_targets[agent_id]
                    if target is not None:
                        if target not in target_attackers:
                            target_attackers[target] = []
                        target_attackers[target].append(agent_id)
            
            # 동일한 타겟을 공격하는 Stalker가 2명 이상이면 보너스
            for target, attackers in target_attackers.items():
                if len(attackers) >= 2:
                    # 공격자 수에 비례한 보너스
                    coordination_bonus += self.coordination_reward * (len(attackers) - 1)
            
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
                if action > 0:  # 0은 일반적으로 "아무것도 하지 않음"
                    return action
            elif isinstance(action, np.ndarray):
                if len(action) > 0:
                    return action[0]
            
            # 관찰에서 타겟 정보 추출 (더 정교한 방법)
            # 예: 관찰의 특정 차원에서 가장 가까운 적의 ID 추출
            return self._find_nearest_enemy_id(obs_array)
            
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
        
        # 기본적으로 모든 에이전트를 Stalker로 간주 (테스트용)
        return True
    
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
