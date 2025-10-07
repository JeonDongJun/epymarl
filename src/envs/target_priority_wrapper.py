import numpy as np
from .multiagentenv import MultiAgentEnv


class TargetPriorityRewardWrapper(MultiAgentEnv):
    """
    특정 타겟을 우선적으로 공격하도록 하는 보상 래퍼
    """
    
    def __init__(self, env, priority_target="zealot", priority_reward=0.2, 
                 stalker_role_id=0, zealot_role_id=1):
        """
        Args:
            env: 원본 환경
            priority_target: 우선 공격할 타겟 ("zealot" 또는 "stalker")
            priority_reward: 우선 타겟 공격 시 추가 보상
            stalker_role_id: Stalker의 role ID
            zealot_role_id: Zealot의 role ID
        """
        self.env = env
        self.priority_target = priority_target
        self.priority_reward = priority_reward
        self.stalker_role_id = stalker_role_id
        self.zealot_role_id = zealot_role_id
        
        # 에피소드 정보
        self.n_agents = env.n_agents
        self.episode_limit = env.episode_limit
        
        # 타겟 추적을 위한 변수들
        self.last_actions = None
        self.target_counts = {}  # 각 타겟에 대한 공격 횟수 추적
        
    def step(self, actions):
        """액션을 실행하고 우선 타겟 공격 보상을 계산"""
        # 원본 환경에서 스텝 실행
        obss, reward, terminated, truncated, info = self.env.step(actions)
        
        # 우선 타겟 공격 보상 계산
        additional_reward = self._calculate_priority_reward(actions)
        
        # 리워드에 추가 리워드 더하기
        if isinstance(reward, (int, float)):
            reward += additional_reward
        elif isinstance(reward, list):
            reward = [r + additional_reward for r in reward]
        
        # 액션 저장 (다음 스텝에서 사용)
        self.last_actions = actions
        
        return obss, reward, terminated, truncated, info
    
    def _calculate_priority_reward(self, actions):
        """우선 타겟 공격 보상 계산"""
        try:
            # 현재 관찰에서 적 정보 추출
            obs = self.env.get_obs()
            if not obs:
                return 0.0
            
            # 타겟 정보 추출
            current_targets = self._extract_targets_from_obs(obs, actions)
            
            if not current_targets:
                return 0.0
            
            # 우선 타겟 공격 보상 계산
            priority_bonus = 0.0
            
            for agent_id, action in enumerate(actions):
                if self._is_attack_action(action):
                    target = current_targets[agent_id]
                    if target is not None:
                        # 타겟이 우선 타겟인지 확인
                        if self._is_priority_target(target):
                            priority_bonus += self.priority_reward
            
            return priority_bonus
            
        except Exception as e:
            print(f"Warning: Could not calculate priority reward: {e}")
            return 0.0
    
    def _extract_targets_from_obs(self, obs, actions):
        """관찰에서 타겟 정보 추출"""
        targets = []
        
        for agent_id, (agent_obs, action) in enumerate(zip(obs, actions)):
            if self._is_attack_action(action):
                try:
                    if hasattr(agent_obs, 'numpy'):
                        obs_array = agent_obs.numpy()
                    elif hasattr(agent_obs, 'cpu'):
                        obs_array = agent_obs.cpu().numpy()
                    else:
                        obs_array = np.array(agent_obs)
                    
                    # 관찰에서 타겟 정보 추출
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
            if isinstance(action, (int, np.integer)):
                if action > 0:  # 0은 일반적으로 "아무것도 하지 않음"
                    return action
            elif isinstance(action, np.ndarray):
                if len(action) > 0:
                    return action[0]
            
            # 관찰에서 타겟 정보 추출
            return self._find_nearest_enemy_id(obs_array)
            
        except Exception:
            return None
    
    def _find_nearest_enemy_id(self, obs_array):
        """관찰에서 가장 가까운 적의 ID 찾기"""
        try:
            # 간단한 휴리스틱: 관찰의 특정 부분에서 적 정보 추출
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
    
    def _is_priority_target(self, target_id):
        """타겟이 우선 타겟인지 확인"""
        try:
            # 간단한 휴리스틱: 타겟 ID를 기반으로 유닛 타입 추정
            # 실제로는 더 정교한 방법이 필요할 수 있음
            
            if self.priority_target == "zealot":
                # Zealot은 보통 특정 ID 범위에 있다고 가정
                # 3s5z_vs_3s6z에서 적 Zealot은 ID 3-8 범위에 있을 가능성이 높음
                return 3 <= target_id <= 8
            elif self.priority_target == "stalker":
                # Stalker는 보통 특정 ID 범위에 있다고 가정
                # 3s5z_vs_3s6z에서 적 Stalker는 ID 0-2 범위에 있을 가능성이 높음
                return 0 <= target_id <= 2
            
            return False
            
        except Exception:
            return False
    
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
