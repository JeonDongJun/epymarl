from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        if getattr(self.args, "obs_agent_role", False):
            # Add role embedding to agent input
            agent_roles = None
            if hasattr(batch.data.transition_data, 'agent_roles'):
                agent_roles = batch.data.transition_data['agent_roles']
            elif 'agent_roles' in batch.data.transition_data:
                agent_roles = batch.data.transition_data['agent_roles']
            
            if agent_roles is not None:
                role_embed_dim = getattr(self.args, "role_embed_dim", 8)
                role_embeddings = self._get_role_embeddings(agent_roles[:, t], role_embed_dim, batch.device)
                inputs.append(role_embeddings)

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            if isinstance(input_shape, tuple):
                input_shape = input_shape[0] + scheme["actions_onehot"]["vshape"][0]
            else:
                input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            if isinstance(input_shape, tuple):
                input_shape = input_shape[0] + self.n_agents
            else:
                input_shape += self.n_agents

        return input_shape
    
    def _get_role_embeddings(self, agent_roles, role_embed_dim, device):
        """Generate role embeddings for agents"""
        bs = agent_roles.size(0)
        # Create a simple role embedding lookup table
        # For now, we'll use a simple one-hot encoding approach
        # In practice, this could be learned embeddings
        n_roles = getattr(self.args, "n_roles", self.n_agents)
        role_embeddings = th.zeros(bs, self.n_agents, role_embed_dim, device=device)
        
        for i in range(bs):
            for j in range(self.n_agents):
                role_id = agent_roles[i, j].item()
                if role_id < n_roles:
                    # Simple one-hot encoding for role
                    role_embeddings[i, j, role_id % role_embed_dim] = 1.0
                    
        return role_embeddings
