import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim
        
        # Role embedding parameters
        self.use_role_embedding = getattr(args, "use_role_embedding", False)
        if self.use_role_embedding:
            self.n_roles = getattr(args, "n_roles", self.n_agents)
            self.role_embed_dim = getattr(args, "role_embed_dim", 8)
            self.role_embedding = nn.Embedding(self.n_roles, self.role_embed_dim)
            # Role-aware hypernetworks
            self.role_state_dim = self.state_dim + self.role_embed_dim * self.n_agents
        else:
            self.role_state_dim = self.state_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.role_state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.role_state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.role_state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim * self.n_agents),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.role_state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim),
            )
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.role_state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.role_state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, agent_qs, states, agent_roles=None):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        
        # Prepare role-aware state input
        if self.use_role_embedding and agent_roles is not None:
            # agent_roles shape: (batch_size, n_agents) - each element is role_id
            agent_roles = agent_roles.reshape(-1, self.n_agents)  # (bs, n_agents)
            role_embeddings = self.role_embedding(agent_roles)  # (bs, n_agents, role_embed_dim)
            role_embeddings_flat = role_embeddings.reshape(-1, self.role_embed_dim * self.n_agents)  # (bs, role_embed_dim * n_agents)
            # Concatenate state with role embeddings
            role_state_input = th.cat([states, role_embeddings_flat], dim=1)  # (bs, role_state_dim)
        else:
            role_state_input = states
            
        # First layer
        w1 = th.abs(self.hyper_w_1(role_state_input))
        b1 = self.hyper_b_1(role_state_input)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(role_state_input))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(role_state_input).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
