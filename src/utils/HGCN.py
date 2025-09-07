import torch
import torch.nn as nn
import torch.nn.functional as F

class HGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_agents, num_groups, num_layers, 
                 compression_ratio=0.9, compression_method='linear'):
        super(HGCN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_agents = num_agents
        self.num_groups = num_groups
        self.num_layers = num_layers
        self.compression_ratio = compression_ratio
        self.compression_method = compression_method

        # 确保 hidden_dim 能被 num_heads 整除
        num_heads = 4
        if hidden_dim % num_heads != 0:
            adjusted_hidden_dim = ((hidden_dim // num_heads) + 1) * num_heads
            print(f"Warning: hidden_dim {hidden_dim} is not divisible by num_heads {num_heads}. Adjusting to {adjusted_hidden_dim}")
            self.hidden_dim = adjusted_hidden_dim
        else:
            self.hidden_dim = hidden_dim

        print(f"[HGCN] Initializing with compression: ratio={compression_ratio}, method={compression_method}")

        self.layers = nn.ModuleList()
        self.layers.append(HGCNLayer(self.in_dim, self.hidden_dim, num_groups, 
                                   compression_ratio, compression_method))
        for _ in range(num_layers - 2):
            self.layers.append(HGCNLayer(self.hidden_dim, self.hidden_dim, num_groups, 
                                       compression_ratio, compression_method))
        self.layers.append(HGCNLayer(self.hidden_dim, out_dim, num_groups, 
                                   compression_ratio, compression_method))

    def forward(self, x, hypergraph=None):
        # hypergraph 파라미터는 호환성을 위해 유지하지만 사용하지 않음
        if x.dim() == 2:
            x = x.view(1, -1, self.in_dim)

        attention_weights_all_layers = []

        for layer in self.layers:
            x = layer(x)
            if hasattr(layer, 'attention_weights') and layer.attention_weights is not None:
                attention_weights_all_layers.append(layer.attention_weights.detach().clone())

        self.attention_weights = attention_weights_all_layers
        return x

    def get_attention_weights(self):
        return self.attention_weights

    def update_groups(self, new_num_groups):
        self.num_groups = new_num_groups
        for layer in self.layers:
            layer.update_groups(new_num_groups)

class HGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_groups, compression_ratio=0.5, compression_method='linear'):
        super(HGCNLayer, self).__init__()
        self.original_in_dim = in_dim
        self.out_dim = out_dim
        self.num_groups = num_groups
        self.compression_ratio = compression_ratio
        self.compression_method = compression_method

        # 确保 embed_dim 能被 num_heads 整除
        num_heads = 4
        if in_dim % num_heads != 0:
            adjusted_dim = ((in_dim // num_heads) + 1) * num_heads
            print(f"Warning: in_dim {in_dim} is not divisible by num_heads {num_heads}. Adjusting to {adjusted_dim}")
            self.in_dim = adjusted_dim
        else:
            self.in_dim = in_dim

        # 特征变换矩阵 - 실제 입력 차원에 맞춤
        # self.feature_transform = nn.Linear(in_dim, self.in_dim)
        self.feature_transform = nn.Linear(in_dim, 32)

        # 차원 축소 레이어 추가
        self.compressed_dim = min(32, int(self.in_dim * self.compression_ratio))
        
        if compression_method == 'linear':
            # 단순 선형 변환
            self.dimension_reducer = nn.Linear(self.in_dim, self.compressed_dim)
        elif compression_method == 'mlp':
            # MLP를 사용한 비선형 압축
            self.dimension_reducer = nn.Sequential(
                nn.Linear(self.in_dim, self.compressed_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.compressed_dim * 2, self.compressed_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        elif compression_method == 'conv':
            # 1D 컨볼루션을 사용한 압축
            self.dimension_reducer = nn.Sequential(
                nn.Conv1d(self.in_dim, self.compressed_dim, kernel_size=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            )
        elif compression_method == 'attention':
            # Self-attention을 사용한 압축
            self.dimension_reducer = nn.Sequential(
                nn.Linear(self.in_dim, self.compressed_dim),
                nn.ReLU(),
                nn.MultiheadAttention(embed_dim=self.compressed_dim, num_heads=2, batch_first=True),
                nn.Linear(self.compressed_dim, self.compressed_dim)
            )
        else:
            # 기본값: 단순 선형 변환
            self.dimension_reducer = nn.Linear(self.in_dim, self.compressed_dim)

        # 线性卷积层，用于最终的特征输出
        self.linear = nn.Linear(self.compressed_dim, out_dim)

        # 注意力机制 - 압축된 차원에서 attention 수행
        self.attention = nn.MultiheadAttention(embed_dim=self.compressed_dim, num_heads=num_heads, batch_first=True)
        
        # 注意力权重变量
        self.attention_weights = None

    def forward(self, x, hypergraph=None):
        # hypergraph 파라미터는 호환성을 위해 유지하지만 사용하지 않음
        device = x.device
        self.feature_transform = self.feature_transform.to(device)
        self.dimension_reducer = self.dimension_reducer.to(device)
        self.attention = self.attention.to(device)
        self.linear = self.linear.to(device)

        batch_size, num_agents, feature_dim = x.size()
        
        # 입력 차원이 예상과 다른 경우 경고 출력
        if feature_dim != self.original_in_dim:
            # print(f"Warning: Expected input dimension {self.original_in_dim}, but got {feature_dim}")
            # 동적으로 feature_transform 레이어 재구성
            if not hasattr(self, '_dynamic_feature_transform') or self._dynamic_feature_transform.in_features != feature_dim:
                self._dynamic_feature_transform = nn.Linear(feature_dim, 32).to(device)
            feature_transform_layer = self._dynamic_feature_transform
        else:
            feature_transform_layer = self.feature_transform

        # 特征变换：线性变换 X -> X' (feature_dim -> in_dim)
        x_transformed = F.relu(feature_transform_layer(x))
    
        
        # 하이퍼그래프 없이 직접적인 attention 사용 (압축된 차원에서)
        # 모든 에이전트 간의 상호작용을 attention으로 모델링
        x_with_attention, attention_weights = self.attention(x_transformed, x_transformed, x_transformed)
        self.attention_weights = attention_weights.detach().clone()

        # 对聚合后的特征通过线性层进行卷积变换
        x_combined = F.relu(self.linear(x_with_attention))
        return x_combined

    def update_groups(self, new_num_groups):
        if new_num_groups != self.num_groups:
            self.num_groups = new_num_groups
            # 하이퍼그래프 관련 레이어가 없으므로 단순히 그룹 수만 업데이트