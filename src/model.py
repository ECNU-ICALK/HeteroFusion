import torch
import torch.nn as nn
import torch.nn.functional as F

class ConflictAwareBlockEncoder(nn.Module):
    def __init__(self, block_size, rank, embed_dim, mu_gate=-0.5):
        super().__init__()
        self.block_size = block_size
        self.rank = rank
        self.flat_dim = block_size * rank
        self.mu_gate = mu_gate
        
        self.input_norm = nn.LayerNorm(self.flat_dim)
        
        self.proj_row = nn.Sequential(
            nn.Linear(self.flat_dim, embed_dim),
            nn.GELU()
        )
        self.proj_col = nn.Sequential(
            nn.Linear(self.flat_dim, embed_dim),
            nn.GELU()
        )
        
        self.gate_net = nn.Sequential(
            nn.Linear(rank, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim)
        )
        
        self.fusion_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_flat, s_values):
        B, N, _ = x_flat.shape
        x = self.input_norm(x_flat)
        
        feat_row = self.proj_row(x)
        x_reshaped = x.view(-1, self.block_size, self.rank)
        x_transposed = x_reshaped.transpose(1, 2).contiguous().view(B, N, -1)
        feat_col = self.proj_col(x_transposed)
        
        feat_spatial = feat_row + feat_col
        
        logits = self.gate_net(s_values)
        gate = torch.clamp(F.relu(logits + self.mu_gate), min=0.0, max=1.0)
        
        feat_final = feat_spatial * gate
        return self.fusion_norm(feat_final)

class TopologyAlignedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value):
        B, N_Q, _ = query.shape
        B, N_K, _ = key.shape
        q = self.q_proj(query).view(B, N_Q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, N_K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, N_K, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N_Q, -1)
        return self.out_proj(attn_out), weights.mean(dim=1)

class HeteroFusionTransferNet(nn.Module):
    def __init__(self, 
                 rank=8, 
                 block_size=4096, 
                 embed_dim=1024, 
                 num_heads=8,
                 max_pos_embeddings=2048,
                 mu_gate=-0.5): 
        super().__init__()
        self.block_size = block_size
        self.rank = rank
        input_dim = block_size * rank
        
        self.encoder = ConflictAwareBlockEncoder(block_size, rank, embed_dim, mu_gate)
        self.pos_embed = nn.Embedding(max_pos_embeddings, embed_dim)
        self.attention = TopologyAlignedAttention(embed_dim, num_heads)
        self.decoder_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, input_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.zeros_(self.decoder_layer[-1].weight)
        nn.init.zeros_(self.decoder_layer[-1].bias)

    def _add_position_embedding(self, x):
        B, N, _ = x.shape
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        return x + self.pos_embed(positions)

    def forward(self, tgt_flat, tgt_s, src_flat, src_s):
        tgt_emb = self.encoder(tgt_flat, tgt_s)
        src_emb = self.encoder(src_flat, src_s)
        
        tgt_emb_pos = self._add_position_embedding(tgt_emb)
        src_emb_pos = self._add_position_embedding(src_emb)
        
        context, attn_weights = self.attention(query=tgt_emb_pos, key=src_emb_pos, value=src_emb_pos)
        delta_flat = self.decoder_layer(context)
        
        return delta_flat, attn_weights, tgt_emb, src_emb


# Backward-compatible aliases for older experiment code.
HybridBlockEncoder = ConflictAwareBlockEncoder
HyperAttentionDecoder = TopologyAlignedAttention
HybridHyperNet = HeteroFusionTransferNet
