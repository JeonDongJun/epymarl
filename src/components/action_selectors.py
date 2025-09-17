import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule
REGISTRY = {}


class BaseActionSelector:
    """액션 선택기의 기본 클래스"""
    
    def _validate_and_fix_actions(self, picked_actions, avail_actions):
        """
        선택된 액션들이 유효한지 확인하고, 유효하지 않은 경우 수정
        """
        batch_size, n_agents = picked_actions.shape
        
        for b in range(batch_size):
            for a in range(n_agents):
                action = picked_actions[b, a].item()
                
                if avail_actions[b, a, action] == 0:
                    # 유효하지 않은 액션인 경우, 유효한 액션 중에서 선택
                    valid_actions = th.where(avail_actions[b, a] == 1)[0]
                    
                    if len(valid_actions) > 0:
                        # 첫 번째 유효한 액션을 선택 (또는 랜덤하게 선택 가능)
                        picked_actions[b, a] = valid_actions[0]
                        print(f"[WARNING] Agent {a} in batch {b} selected invalid action {action}, "
                              f"corrected to {valid_actions[0].item()}")
                    else:
                        # 모든 액션이 유효하지 않은 경우 (죽은 에이전트), 액션 0 선택
                        picked_actions[b, a] = 0
                        print(f"[WARNING] Agent {a} in batch {b} has no valid actions, "
                              f"selecting action 0 (no-op)")
        
        return picked_actions


class MultinomialActionSelector(BaseActionSelector):

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        # 액션 유효성 검사 및 수정
        picked_actions = self._validate_and_fix_actions(picked_actions, avail_actions)
        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector(BaseActionSelector):

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        
        # 액션 유효성 검사 및 수정
        picked_actions = self._validate_and_fix_actions(picked_actions, avail_actions)
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class SoftPoliciesSelector(BaseActionSelector):

    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        m = Categorical(agent_inputs)
        picked_actions = m.sample().long()
        
        # 액션 유효성 검사 및 수정
        picked_actions = self._validate_and_fix_actions(picked_actions, avail_actions)
        return picked_actions


REGISTRY["soft_policies"] = SoftPoliciesSelector