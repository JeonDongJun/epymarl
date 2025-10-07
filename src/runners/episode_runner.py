from functools import partial

import numpy as np

from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from envs import register_smac, register_smacv2, register_stalker_coordination, register_target_priority


class EpisodeRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        # registering both smac and smacv2 causes a pysc2 error
        # --> dynamically register the needed env
        if self.args.env == "sc2":
            register_smac()
        elif self.args.env == "sc2v2":
            register_smacv2()
        elif self.args.env == "sc2_stalker_coordination":
            register_stalker_coordination()
        elif self.args.env == "sc2_target_priority":
            register_target_priority()

        self.env = env_REGISTRY[self.args.env](
            **self.args.env_args,
            common_reward=self.args.common_reward,
            reward_scalarisation=self.args.reward_scalarisation,
        )
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        if self.args.common_reward:
            episode_return = 0
        else:
            episode_return = np.zeros(self.args.n_agents)
        self.mac.init_hidden(batch_size=self.batch_size)
        
        # Initialize agent roles if role embedding is enabled
        agent_roles = None
        if getattr(self.args, "use_role_embedding", False):
            agent_roles = self._get_agent_roles()
            
        # Set agent roles in environment if it supports it
        if hasattr(self.env, 'set_agent_roles') and agent_roles is not None:
            self.env.set_agent_roles(agent_roles)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }
            
            # Add agent roles if role embedding is enabled
            if agent_roles is not None:
                pre_transition_data["agent_roles"] = [agent_roles]

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )

            _, reward, terminated, truncated, env_info = self.env.step(actions[0])
            terminated = terminated or truncated
            if test_mode and self.args.render:
                self.env.render()
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            if self.args.common_reward:
                post_transition_data["reward"] = [(reward,)]
            else:
                post_transition_data["reward"] = [tuple(reward)]

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        
        # Add agent roles if role embedding is enabled
        if agent_roles is not None:
            last_data["agent_roles"] = [agent_roles]
        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
        )
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update(
            {
                k: cur_stats.get(k, 0) + env_info.get(k, 0)
                for k in set(cur_stats) | set(env_info)
            }
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        if self.args.common_reward:
            self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
            self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        else:
            for i in range(self.args.n_agents):
                self.logger.log_stat(
                    prefix + f"agent_{i}_return_mean",
                    np.array(returns)[:, i].mean(),
                    self.t_env,
                )
                self.logger.log_stat(
                    prefix + f"agent_{i}_return_std",
                    np.array(returns)[:, i].std(),
                    self.t_env,
                )
            total_returns = np.array(returns).sum(axis=-1)
            self.logger.log_stat(
                prefix + "total_return_mean", total_returns.mean(), self.t_env
            )
            self.logger.log_stat(
                prefix + "total_return_std", total_returns.std(), self.t_env
            )
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(
                    prefix + k + "_mean", v / stats["n_episodes"], self.t_env
                )
        stats.clear()
    
    def _get_agent_roles(self):
        """Generate agent roles based on unit types for the current episode"""
        n_roles = getattr(self.args, "n_roles", self.args.n_agents)
        
        # Check if we should extract roles from observations
        extract_from_obs = getattr(self.args, "extract_roles_from_obs", True)
        extraction_method = getattr(self.args, "role_extraction_method", "observation_signature")
        
        agent_roles = None
        
        if extract_from_obs:
            if extraction_method == "observation_signature":
                agent_roles = self._extract_roles_from_observations()
            elif extraction_method == "fixed_mapping":
                agent_roles = self._get_fixed_role_mapping()
            elif extraction_method == "cyclic":
                agent_roles = self._get_cyclic_roles(n_roles)
        
        if agent_roles is None:
            # Fallback: assign roles cyclically
            agent_roles = self._get_cyclic_roles(n_roles)
            
        return agent_roles
    
    def _get_cyclic_roles(self, n_roles):
        """Assign roles cyclically based on agent index"""
        return np.array([i % n_roles for i in range(self.args.n_agents)])
    
    def _get_fixed_role_mapping(self):
        """Get fixed role mapping for specific maps"""
        fixed_mapping = getattr(self.args, "fixed_role_mapping", {})
        
        # Try to get map name from environment
        map_name = None
        try:
            if hasattr(self.env, 'env') and hasattr(self.env.env, 'map_name'):
                map_name = self.env.env.map_name
        except:
            pass
        
        if map_name and map_name in fixed_mapping:
            roles = fixed_mapping[map_name]
            if len(roles) == self.args.n_agents:
                return np.array(roles)
        
        return None
    
    def _extract_roles_from_observations(self):
        """Extract agent roles from observations based on unit types"""
        try:
            # Get current observations
            obs = self.env.get_obs()
            if not obs:
                return None
                
            # For SMAC environments, unit type information is typically in the first few dimensions
            # of the observation. We'll use a simple heuristic to extract unit types.
            agent_roles = []
            
            for i, agent_obs in enumerate(obs):
                # Extract unit type from observation
                # In SMAC, the first few dimensions often contain unit-specific information
                unit_type = self._extract_unit_type_from_obs(agent_obs)
                agent_roles.append(unit_type)
            
            return np.array(agent_roles)
            
        except Exception as e:
            print(f"Warning: Could not extract roles from observations: {e}")
            return None
    
    def _extract_unit_type_from_obs(self, obs):
        """Extract unit type from individual agent observation"""
        try:
            # Convert to numpy if it's a tensor
            if hasattr(obs, 'numpy'):
                obs = obs.numpy()
            elif hasattr(obs, 'cpu'):
                obs = obs.cpu().numpy()
            
            # Simple heuristic: use the first few dimensions to determine unit type
            # This is a simplified approach - in practice, you might need to adjust
            # based on the specific SMAC map and observation structure
            
            # Use the sum of first 5 dimensions as a unit type identifier
            unit_signature = np.sum(obs[:5]) if len(obs) >= 5 else np.sum(obs)
            
            # Map to role IDs (0, 1, 2, etc.)
            n_roles = getattr(self.args, "n_roles", 3)
            role_id = int(abs(unit_signature)) % n_roles
            
            return role_id
            
        except Exception as e:
            print(f"Warning: Could not extract unit type from observation: {e}")
            return 0  # Default role
