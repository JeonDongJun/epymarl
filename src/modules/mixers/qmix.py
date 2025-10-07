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
            # Role-aware hypernetworks - calculate dynamically
            self.role_state_dim = self.state_dim + self.role_embed_dim * self.n_agents
        else:
            self.role_state_dim = self.state_dim
        
        # Initialize with a reasonable default, will be adjusted dynamically
        self._initialized = False

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
        states_flat = states.reshape(states.size(0), -1)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        
        # Prepare role-aware state input
        if self.use_role_embedding and agent_roles is not None:
            # Debug information
            print(f"Debug - states shape: {states.shape}")
            print(f"Debug - agent_roles shape: {agent_roles.shape}")
            print(f"Debug - agent_roles total elements: {agent_roles.numel()}")
            print(f"Debug - states_flat shape: {states_flat.shape}")
            print(f"Debug - expected bs: {bs}, n_agents: {self.n_agents}")
            
            # Calculate expected total elements for agent_roles
            expected_elements = bs * self.n_agents
            print(f"Debug - expected agent_roles elements: {expected_elements}")
            
            # Handle different agent_roles shapes
            if agent_roles.numel() == expected_elements:
                # Correct size - reshape normally
                agent_roles = agent_roles.reshape(bs, self.n_agents)
            elif agent_roles.numel() == self.n_agents:
                # Single batch - broadcast to all batches
                agent_roles = agent_roles.unsqueeze(0).expand(bs, -1)
            elif agent_roles.numel() % self.n_agents == 0:
                # Multiple batches but not matching current batch size
                actual_bs = agent_roles.numel() // self.n_agents
                print(f"Warning: agent_roles has {actual_bs} batches, but current batch size is {bs}")
                if actual_bs >= bs:
                    # Take first bs batches
                    agent_roles = agent_roles.reshape(actual_bs, self.n_agents)[:bs]
                else:
                    # Repeat the last batch to match batch size
                    agent_roles = agent_roles.reshape(actual_bs, self.n_agents)
                    if actual_bs > 0:
                        last_batch = agent_roles[-1:].expand(bs - actual_bs, -1)
                        agent_roles = th.cat([agent_roles, last_batch], dim=0)
                    else:
                        # Fallback: use zeros
                        agent_roles = th.zeros(bs, self.n_agents, dtype=agent_roles.dtype, device=agent_roles.device)
            else:
                # Fallback: create default roles
                print(f"Warning: Cannot reshape agent_roles with {agent_roles.numel()} elements to ({bs}, {self.n_agents})")
                agent_roles = th.zeros(bs, self.n_agents, dtype=agent_roles.dtype, device=agent_roles.device)
            
            print(f"Debug - final agent_roles shape: {agent_roles.shape}")
            
            role_embeddings = self.role_embedding(agent_roles)  # (bs, n_agents, role_embed_dim)
            role_embeddings_flat = role_embeddings.reshape(bs, self.role_embed_dim * self.n_agents)  # (bs, role_embed_dim * n_agents)
            
            print(f"Debug - role_embeddings shape: {role_embeddings.shape}")
            print(f"Debug - role_embeddings_flat shape: {role_embeddings_flat.shape}")
            print(f"Debug - expected role_state_dim: {self.role_state_dim}")
            
            # Ensure both tensors have the same batch size
            if states_flat.shape[0] != role_embeddings_flat.shape[0]:
                print(f"Warning: Batch size mismatch - states: {states_flat.shape[0]}, roles: {role_embeddings_flat.shape[0]}")
                # Take the minimum batch size
                min_bs = min(states_flat.shape[0], role_embeddings_flat.shape[0])
                states_flat = states_flat[:min_bs]
                role_embeddings_flat = role_embeddings_flat[:min_bs]
            
            # Concatenate state with role embeddings
            role_state_input = th.cat([states_flat, role_embeddings_flat], dim=1)  # (bs, actual_dim)
        else:
            role_state_input = states_flat
            
        # Update role_state_dim dynamically based on actual input
        actual_dim = role_state_input.shape[1]
        if actual_dim != self.role_state_dim:
            print(f"Warning: Updating role_state_dim from {self.role_state_dim} to {actual_dim}")
            self.role_state_dim = actual_dim
            
            # Recreate hypernetworks with correct dimensions
            if getattr(self.args, "hypernet_layers", 1) == 1:
                self.hyper_w_1 = nn.Linear(self.role_state_dim, self.embed_dim * self.n_agents).to(role_state_input.device)
                self.hyper_w_final = nn.Linear(self.role_state_dim, self.embed_dim).to(role_state_input.device)
            elif getattr(self.args, "hypernet_layers", 1) == 2:
                hypernet_embed = self.args.hypernet_embed
                self.hyper_w_1 = nn.Sequential(
                    nn.Linear(self.role_state_dim, hypernet_embed),
                    nn.ReLU(),
                    nn.Linear(hypernet_embed, self.embed_dim * self.n_agents),
                ).to(role_state_input.device)
                self.hyper_w_final = nn.Sequential(
                    nn.Linear(self.role_state_dim, hypernet_embed),
                    nn.ReLU(),
                    nn.Linear(hypernet_embed, self.embed_dim),
                ).to(role_state_input.device)
            
            # Recreate bias and V networks
            self.hyper_b_1 = nn.Linear(self.role_state_dim, self.embed_dim).to(role_state_input.device)
            self.V = nn.Sequential(
                nn.Linear(self.role_state_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, 1),
            ).to(role_state_input.device)
            
        # First layer
        w1 = th.abs(self.hyper_w_1(role_state_input))
        b1 = self.hyper_b_1(role_state_input)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        
        # Debug batch size compatibility
        print(f"Debug - agent_qs shape: {agent_qs.shape}")
        print(f"Debug - w1 shape: {w1.shape}")
        print(f"Debug - b1 shape: {b1.shape}")
        
        # Ensure batch sizes match
        actual_bs = min(agent_qs.shape[0], w1.shape[0], b1.shape[0])
        if actual_bs != agent_qs.shape[0] or actual_bs != w1.shape[0] or actual_bs != b1.shape[0]:
            print(f"Warning: Batch size mismatch - agent_qs: {agent_qs.shape[0]}, w1: {w1.shape[0]}, b1: {b1.shape[0]}")
            print(f"Using batch size: {actual_bs}")
            agent_qs = agent_qs[:actual_bs]
            w1 = w1[:actual_bs]
            b1 = b1[:actual_bs]
        
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(role_state_input))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(role_state_input).view(-1, 1, 1)
        
        # Ensure batch sizes match for second layer
        actual_bs_final = min(hidden.shape[0], w_final.shape[0], v.shape[0])
        if actual_bs_final != hidden.shape[0] or actual_bs_final != w_final.shape[0] or actual_bs_final != v.shape[0]:
            print(f"Warning: Batch size mismatch in second layer - hidden: {hidden.shape[0]}, w_final: {w_final.shape[0]}, v: {v.shape[0]}")
            print(f"Using batch size: {actual_bs_final}")
            hidden = hidden[:actual_bs_final]
            w_final = w_final[:actual_bs_final]
            v = v[:actual_bs_final]
        
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(actual_bs_final, -1, 1)
        return q_tot
