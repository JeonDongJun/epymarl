import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule
REGISTRY = {}


class MultinomialActionSelector():

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

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

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
        
        # 안전한 랜덤 액션 선택
        random_actions = []
        for i in range(agent_inputs.shape[1]):  # 각 에이전트에 대해
            valid_actions = avail_actions[0, i].nonzero().flatten()
            if len(valid_actions) > 0:
                # 유효한 액션 중에서 랜덤 선택
                random_action = valid_actions[th.randint(0, len(valid_actions), (1,)).item()]
                random_actions.append(random_action)
            else:
                # 유효한 액션이 없으면 액션 0 선택
                random_actions.append(0)
        
        random_actions = th.tensor(random_actions, device=agent_inputs.device).unsqueeze(0)
        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        
        # 강제로 유효한 액션만 선택하도록 수정
        for i, action in enumerate(picked_actions[0]):  # batch_size=1이므로 [0] 사용
            if avail_actions[0, i, action.item()] == 0:
                print(f"WARNING: Agent {i} selected invalid action {action.item()}")
                print(f"  Available actions: {avail_actions[0, i].nonzero().flatten().tolist()}")
                print(f"  Q-values: {masked_q_values[0, i].tolist()}")
                
                # 유효한 액션 중에서 선택
                valid_actions = avail_actions[0, i].nonzero().flatten()
                if len(valid_actions) > 0:
                    # 유효한 액션 중에서 가장 높은 Q-value를 가진 액션 선택
                    valid_q_values = masked_q_values[0, i, valid_actions]
                    best_valid_action = valid_actions[valid_q_values.argmax()]
                    picked_actions[0, i] = best_valid_action
                    print(f"  Corrected to action: {picked_actions[0, i].item()}")
                else:
                    print(f"  ERROR: No valid actions available for agent {i}")
                    # 액션 1을 기본값으로 사용 (일반적으로 유효한 액션)
                    picked_actions[0, i] = 1
                    print(f"  Fallback to action 1")
        
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class SoftPoliciesSelector():

    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        m = Categorical(agent_inputs)
        picked_actions = m.sample().long()
        return picked_actions


REGISTRY["soft_policies"] = SoftPoliciesSelector