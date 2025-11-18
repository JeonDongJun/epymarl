from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Union

import numpy as np


def _ensure_float(value: Union[float, Iterable[float]]) -> float:
    """Convert different value types to float."""

    if isinstance(value, (list, tuple)):
        if not value:
            return 0.0
        return float(np.sum(np.asarray(value, dtype=np.float32)))

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        return float(np.sum(value.astype(np.float32)))

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _traverse_info(info: Dict, key: str) -> tuple[float, bool]:
    """Retrieve nested values from env info dictionaries."""

    current = info
    for part in key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return 0.0, False
    return _ensure_float(current), True


@dataclass
class WeightedRewardFunction:
    """Simple weighted-sum reward shaping."""

    base_reward_weight: float = 1.0
    bias: float = 0.0
    info_weights: Optional[Dict[str, float]] = None
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None
    scale: float = 1.0
    default_missing_value: float = 0.0

    def __post_init__(self) -> None:
        if self.info_weights is None:
            self.info_weights = {}

    def __call__(
        self,
        base_reward: float,
        info: Dict,
        per_agent_reward: Optional[np.ndarray],
        missing_callback=None,
    ) -> float:
        shaped_reward = self.base_reward_weight * base_reward + self.bias

        for key, weight in self.info_weights.items():
            if weight == 0:
                continue
            metric, found = _traverse_info(info, key)
            if not found:
                if missing_callback is not None:
                    missing_callback(key)
                metric = self.default_missing_value
            shaped_reward += weight * metric

        shaped_reward *= self.scale

        if self.clip_min is not None:
            shaped_reward = max(self.clip_min, shaped_reward)
        if self.clip_max is not None:
            shaped_reward = min(self.clip_max, shaped_reward)

        return shaped_reward


class GroupRewardManager:
    """Apply reward shaping profiles for specified agent groups."""

    def __init__(self, args, logger=None):
        self.logger = logger.console_logger if logger is not None else None
        self.enabled = getattr(args, "group_reward_mode", False)
        self.n_agents = getattr(args, "n_agents", None)
        self.profile_functions: Dict[str, WeightedRewardFunction] = {}
        self.agent_to_profile: Dict[int, str] = {}
        self._missing_warned: set[str] = set()

        if not self.enabled:
            return

        if self.n_agents is None:
            raise ValueError("GroupRewardManager requires `args.n_agents` to be set before initialisation.")

        if getattr(args, "common_reward", True):
            raise ValueError(
                "Group reward mode requires `common_reward` to be False so each agent can receive a shaped reward."
            )

        profiles_config = getattr(args, "group_reward_profiles", None)
        if not profiles_config:
            raise ValueError("`group_reward_mode` is enabled but no `group_reward_profiles` were provided in the config.")

        default_profile_name = getattr(args, "group_reward_default_profile", None)

        for profile in profiles_config:
            name = profile.get("name")
            if not name:
                raise ValueError("Each entry in `group_reward_profiles` must define a `name`.")

            function_cfg = profile.get("function", {})
            fn_type = function_cfg.get("type", "weighted_sum")
            if fn_type != "weighted_sum":
                raise ValueError(f"Unsupported reward function type '{fn_type}'. Only 'weighted_sum' is available.")

            fn = WeightedRewardFunction(
                base_reward_weight=function_cfg.get("base_reward_weight", 1.0),
                bias=function_cfg.get("bias", 0.0),
                info_weights=function_cfg.get("info_weights"),
                clip_min=function_cfg.get("clip", {}).get("min") if function_cfg.get("clip") else None,
                clip_max=function_cfg.get("clip", {}).get("max") if function_cfg.get("clip") else None,
                scale=function_cfg.get("scale", 1.0),
                default_missing_value=function_cfg.get("default_missing_value", 0.0),
            )
            self.profile_functions[name] = fn

            for agent_id in profile.get("agents", []):
                if not isinstance(agent_id, int):
                    raise TypeError("Agent identifiers in `group_reward_profiles` must be integers.")
                if agent_id < 0 or agent_id >= self.n_agents:
                    raise ValueError(
                        f"Agent id {agent_id} in profile '{name}' is outside the valid range [0, {self.n_agents})."
                    )
                if agent_id in self.agent_to_profile:
                    raise ValueError(
                        f"Agent id {agent_id} is assigned to multiple reward profiles:"
                        f" '{self.agent_to_profile[agent_id]}' and '{name}'."
                    )
                self.agent_to_profile[agent_id] = name

        if len(self.agent_to_profile) != self.n_agents:
            if default_profile_name is None:
                missing = sorted(set(range(self.n_agents)) - set(self.agent_to_profile))
                raise ValueError(
                    "Some agents do not have a reward profile assigned and no `group_reward_default_profile` was provided."
                    f" Missing agents: {missing}"
                )
            if default_profile_name not in self.profile_functions:
                raise ValueError(
                    f"Default profile '{default_profile_name}' is not defined in `group_reward_profiles`."
                )
            for agent_id in range(self.n_agents):
                if agent_id not in self.agent_to_profile:
                    self.agent_to_profile[agent_id] = default_profile_name

        self.enabled = True

    def apply(self, reward, info):
        if not self.enabled:
            return reward

        per_agent_reward = np.asarray(reward, dtype=np.float32)
        if per_agent_reward.ndim == 0:
            per_agent_reward = np.full(self.n_agents, float(per_agent_reward))
        elif per_agent_reward.size != self.n_agents:
            per_agent_reward = np.resize(per_agent_reward, self.n_agents)

        base_reward = float(np.mean(per_agent_reward))

        shaped_rewards = np.zeros(self.n_agents, dtype=np.float32)
        cache: Dict[str, float] = {}
        for agent_id in range(self.n_agents):
            profile = self.agent_to_profile.get(agent_id)
            if profile is None:
                shaped_rewards[agent_id] = per_agent_reward[agent_id]
                continue

            if profile not in cache:
                cache[profile] = self.profile_functions[profile](
                    base_reward,
                    info,
                    per_agent_reward,
                    missing_callback=self._handle_missing_info,
                )
            shaped_rewards[agent_id] = cache[profile]

        return shaped_rewards

    def _handle_missing_info(self, key: str) -> None:
        if self.logger is None or key in self._missing_warned:
            return

        self.logger.warning(
            "Reward shaping requested info key '%s' which is not provided by the environment. Using the configured default value.",
            key,
        )
        self._missing_warned.add(key)
