import os
import sys

from .multiagentenv import MultiAgentEnv
from .gymma import GymmaWrapper
from .smaclite_wrapper import SMACliteWrapper


if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )


def __check_and_prepare_smac_kwargs(kwargs):
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    assert kwargs[
        "common_reward"
    ], "SMAC only supports common reward. Please set `common_reward=True` or choose a different environment that supports general sum rewards."
    del kwargs["common_reward"]
    del kwargs["reward_scalarisation"]
    assert "map_name" in kwargs, "Please specify the map_name in the env_args"
    return kwargs


def smaclite_fn(**kwargs) -> MultiAgentEnv:
    kwargs = __check_and_prepare_smac_kwargs(kwargs)
    return SMACliteWrapper(**kwargs)


def gymma_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return GymmaWrapper(**kwargs)


REGISTRY = {}
REGISTRY["smaclite"] = smaclite_fn
REGISTRY["gymma"] = gymma_fn


# registering both smac and smacv2 causes a pysc2 error
# --> dynamically register the needed env
def register_smac():
    from .smac_wrapper import SMACWrapper

    def smac_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACWrapper(**kwargs)

    REGISTRY["sc2"] = smac_fn


def register_smacv2():
    from .smacv2_wrapper import SMACv2Wrapper

    def smacv2_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACv2Wrapper(**kwargs)

    REGISTRY["sc2v2"] = smacv2_fn


def register_stalker_coordination():
    from .smac_wrapper import SMACWrapper
    from .stalker_coordination_wrapper import StalkerCoordinationRewardWrapper

    def stalker_coordination_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        
        # Extract coordination-specific parameters before creating base environment
        coordination_reward = kwargs.pop("coordination_reward", 0.1)
        stalker_role_id = kwargs.pop("stalker_role_id", 0)
        
        base_env = SMACWrapper(**kwargs)
        
        # Stalker 협력 리워드 래퍼 적용
        return StalkerCoordinationRewardWrapper(
            base_env, 
            coordination_reward=coordination_reward,
            stalker_role_id=stalker_role_id
        )

    REGISTRY["sc2_stalker_coordination"] = stalker_coordination_fn


def register_target_priority():
    from .smac_wrapper import SMACWrapper
    from .target_priority_wrapper import TargetPriorityRewardWrapper

    def target_priority_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        
        # Extract priority-specific parameters before creating base environment
        priority_target = kwargs.pop("priority_target", "zealot")
        priority_reward = kwargs.pop("priority_reward", 0.2)
        stalker_role_id = kwargs.pop("stalker_role_id", 0)
        zealot_role_id = kwargs.pop("zealot_role_id", 1)
        
        base_env = SMACWrapper(**kwargs)
        
        # 타겟 우선순위 리워드 래퍼 적용
        return TargetPriorityRewardWrapper(
            base_env, 
            priority_target=priority_target,
            priority_reward=priority_reward,
            stalker_role_id=stalker_role_id,
            zealot_role_id=zealot_role_id
        )

    REGISTRY["sc2_target_priority"] = target_priority_fn