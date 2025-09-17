import gymnasium as gym
from gymnasium.spaces import flatdim
from gymnasium.wrappers import TimeLimit
import smaclite  # noqa

from .multiagentenv import MultiAgentEnv


class SMACliteWrapper(MultiAgentEnv):
    def __init__(self, map_name, seed, time_limit, **kwargs):
        self.env = gym.make(f"smaclite/{map_name}-v0", seed=seed, **kwargs)
        self.env = TimeLimit(self.env, max_episode_steps=time_limit)

        self.n_agents = self.env.unwrapped.n_agents
        self.episode_limit = time_limit

        self.longest_action_space = max(self.env.action_space, key=lambda x: x.n)

    def _validate_actions(self, actions):
        """
        액션 유효성을 검사하고 유효하지 않은 액션을 수정
        """
        avail_actions = self.get_avail_actions()
        
        for i, action in enumerate(actions):
            if i < len(avail_actions):
                if not avail_actions[i][action]:
                    # 유효하지 않은 액션인 경우, 유효한 액션 중에서 선택
                    valid_actions = [j for j, avail in enumerate(avail_actions[i]) if avail]
                    if valid_actions:
                        # 첫 번째 유효한 액션을 선택
                        actions[i] = valid_actions[0]
                        print(f"[WARNING] Agent {i} selected invalid action {action}, "
                              f"corrected to {valid_actions[0]}")
                    else:
                        # 모든 액션이 유효하지 않은 경우 (죽은 에이전트), 액션 0 선택
                        actions[i] = 0
                        print(f"[WARNING] Agent {i} has no valid actions, "
                              f"selecting action 0 (no-op)")
        return actions

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        actions = [int(act) for act in actions]
        
        # 액션 유효성 검사 및 수정
        actions = self._validate_actions(actions)
        
        try:
            obs, reward, terminated, truncated, info = self.env.step(actions)
            return obs, reward, terminated, truncated, info
        except ValueError as e:
            # 마지막 수단으로 모든 액션을 0으로 설정
            safe_actions = [0] * len(actions)
            obs, reward, terminated, truncated, info = self.env.step(safe_actions)
            return obs, reward, terminated, truncated, info

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self.env.unwrapped.get_obs()

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        return self.env.unwrapped.get_obs()[agent_id]

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return self.env.unwrapped.obs_size

    def get_state(self):
        return self.env.unwrapped.get_state()

    def get_state_size(self):
        """Returns the shape of the state"""
        return self.env.unwrapped.state_size

    def get_avail_actions(self):
        return self.env.unwrapped.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return self.env.unwrapped.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return flatdim(self.longest_action_space)

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        obs = self.env.reset(seed=seed, options=options)
        return obs, {}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)
