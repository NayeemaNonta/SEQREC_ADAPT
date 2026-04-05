import numpy as np
import torch
from torch import nn


class PointWiseFeedForward(nn.Module):
    """Code from https://github.com/pmixer/SASRec.pytorch."""

    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2)))))
        )
        outputs = outputs.transpose(-1, -2)  # Conv1d expects (N, C, L)
        outputs = outputs + inputs
        return outputs


class SASRec(nn.Module):
    """
    Base SASRec backbone.

    This version adds:
      - encode() -> returns sequence hidden states, shape (B, T, d)
      - get_last_hidden() -> returns final valid hidden state, shape (B, d)
      - score_from_hidden() -> scores candidate items from hidden states
    """

    def __init__(
        self,
        item_num,
        maxlen=128,
        hidden_units=64,
        num_blocks=2,
        num_heads=2,
        dropout_rate=0.1,
        initializer_range=0.02,
        add_head=True,
        pos_enc=True,
    ):
        super().__init__()

        self.item_num = item_num
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.add_head = add_head
        self.pos_enc = pos_enc

        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=hidden_units,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    batch_first=False,
                )
            )
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _build_embeddings(self, input_ids):
        """
        input_ids: (B, T)
        returns:
            seqs: (B, T, d)
            timeline_mask: (B, T) True where padding
        """
        seqs = self.item_emb(input_ids)
        seqs = seqs * (self.item_emb.embedding_dim ** 0.5)

        if self.pos_enc:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            positions = positions.expand(input_ids.shape[0], -1)
            seqs = seqs + self.pos_emb(positions)

        seqs = self.emb_dropout(seqs)

        timeline_mask = (input_ids == 0)  # (B, T)
        seqs = seqs * (~timeline_mask).unsqueeze(-1)

        return seqs, timeline_mask

    def encode(self, input_ids, attention_mask=None):
        """
        Returns contextualized sequence hidden states without item scoring head.

        Args:
            input_ids: (B, T)
            attention_mask: ignored, kept for API compatibility

        Returns:
            outputs: (B, T, d)
        """
        seqs, timeline_mask = self._build_embeddings(input_ids)

        tl = seqs.shape[1]
        causal_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=seqs.device)
        )  # True means masked

        for i in range(len(self.attention_layers)):
            # MultiheadAttention expects (T, B, d) since batch_first=False
            seqs = torch.transpose(seqs, 0, 1)  # (T, B, d)

            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=causal_mask
            )
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)  # (B, T, d)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs = seqs * (~timeline_mask).unsqueeze(-1)

        outputs = self.last_layernorm(seqs)  # (B, T, d)
        return outputs

    def get_last_hidden(self, input_ids, attention_mask=None):
        """
        Extract the final valid hidden state per sequence.

        Args:
            input_ids: (B, T)

        Returns:
            h_last: (B, d)
        """
        hidden = self.encode(input_ids, attention_mask=attention_mask)  # (B, T, d)

        valid_mask = (input_ids != 0)  # (B, T)
        lengths = valid_mask.sum(dim=1)  # (B,)
        lengths = torch.clamp(lengths, min=1)

        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        last_idx = lengths - 1
        h_last = hidden[batch_idx, last_idx, :]  # (B, d)

        return h_last

    def score_from_hidden(self, hidden, candidate_ids=None):
        """
        Score items from a hidden representation.

        Args:
            hidden: (B, d)
            candidate_ids:
                None -> score against full item vocabulary, returns (B, item_num+1)
                Tensor (B, K) -> returns (B, K)

        Returns:
            logits
        """
        if candidate_ids is None:
            return torch.matmul(hidden, self.item_emb.weight.transpose(0, 1))  # (B, N+1)

        candidate_emb = self.item_emb(candidate_ids)  # (B, K, d)
        logits = (candidate_emb * hidden.unsqueeze(1)).sum(dim=-1)  # (B, K)
        return logits

    def forward(self, input_ids, attention_mask=None):
        """
        Original SASRec behavior:
            returns token-level logits if add_head=True
            otherwise returns token-level hidden states
        """
        outputs = self.encode(input_ids, attention_mask=attention_mask)  # (B, T, d)

        if self.add_head:
            outputs = torch.matmul(outputs, self.item_emb.weight.transpose(0, 1))  # (B, T, N+1)

        return outputs

# ADDED MODULE
class ResidualMLPAdapter(nn.Module):
    """
    Single bottleneck residual adapter:
        f(h) = W2 sigma(W1 h + b1) + b2
    """

    def __init__(self, hidden_dim, bottleneck_dim, dropout=0.0, activation="gelu"):
        super().__init__()

        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, h):
        delta = self.up(self.dropout(self.activation(self.down(h))))
        return h + delta


class PrototypeResidualAdapter(nn.Module):
    """
    Cluster-conditioned adapter bank.

    Each cluster k has its own small residual MLP.
    User u uses adapter corresponding to cluster_id z_u.
    """

    def __init__(
        self,
        hidden_dim,
        bottleneck_dim,
        num_clusters,
        dropout=0.0,
        activation="gelu",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_clusters = num_clusters

        self.adapters = nn.ModuleList(
            [
                ResidualMLPAdapter(
                    hidden_dim=hidden_dim,
                    bottleneck_dim=bottleneck_dim,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_clusters)
            ]
        )

    def forward(self, h, cluster_ids):
        """
        Args:
            h: (B, d)
            cluster_ids: (B,) with values in [0, num_clusters-1]

        Returns:
            h_tilde: (B, d)
        """
        if h.dim() != 2:
            raise ValueError(f"h must have shape (B, d), got {tuple(h.shape)}")
        if cluster_ids.dim() != 1:
            raise ValueError(f"cluster_ids must have shape (B,), got {tuple(cluster_ids.shape)}")
        if h.size(0) != cluster_ids.size(0):
            raise ValueError("Batch size mismatch between h and cluster_ids")

        h_tilde = torch.empty_like(h)

        for k in range(self.num_clusters):
            mask = (cluster_ids == k)
            if mask.any():
                h_tilde[mask] = self.adapters[k](h[mask])

        return h_tilde


class SASRecPrototypeAdaptationModel(nn.Module):
    """
    Frozen SASRec backbone + prototype-shared residual adaptation.

    Intended use:
      - load pretrained T1 SASRec checkpoint into backbone
      - freeze backbone
      - train only adapter parameters on T2_adapt
    """

    def __init__(
        self,
        backbone: SASRec,
        num_clusters: int,
        bottleneck_dim: int,
        adapter_dropout: float = 0.0,
        adapter_activation: str = "gelu",
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.backbone = backbone
        self.adapter = PrototypeResidualAdapter(
            hidden_dim=backbone.hidden_units,
            bottleneck_dim=bottleneck_dim,
            num_clusters=num_clusters,
            dropout=adapter_dropout,
            activation=adapter_activation,
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def get_backbone_hidden(self, input_ids, attention_mask=None):
        return self.backbone.get_last_hidden(input_ids, attention_mask=attention_mask)

    def get_adapted_hidden(self, input_ids, cluster_ids, attention_mask=None):
        h = self.backbone.get_last_hidden(input_ids, attention_mask=attention_mask)  # (B, d)
        h_tilde = self.adapter(h, cluster_ids)  # (B, d)
        return h_tilde

    def score_candidates(self, input_ids, cluster_ids, candidate_ids=None, attention_mask=None):
        """
        Args:
            input_ids: (B, T)
            cluster_ids: (B,)
            candidate_ids:
                None -> full-vocab scores, shape (B, N+1)
                (B, K) -> candidate scores, shape (B, K)
        """
        h_tilde = self.get_adapted_hidden(
            input_ids=input_ids,
            cluster_ids=cluster_ids,
            attention_mask=attention_mask,
        )
        logits = self.backbone.score_from_hidden(h_tilde, candidate_ids=candidate_ids)
        return logits

    def forward(self, input_ids, cluster_ids, candidate_ids=None, attention_mask=None):
        return self.score_candidates(
            input_ids=input_ids,
            cluster_ids=cluster_ids,
            candidate_ids=candidate_ids,
            attention_mask=attention_mask,
        )


class ContextGateAdapter(nn.Module):
    """
    Context-conditioned scalar gate adapter.

    h̃ = h + α(h) · f_ϕ(h)

    α(h) = sigmoid(w_gate · h + b_gate)   — scalar in (0, 1)
    f_ϕ(h) = W2 σ(W1 h + b1) + b2        — bottleneck MLP

    Both gate and MLP are conditioned on h, so the edit magnitude
    and direction are context-dependent, not user-dependent.
    """

    def __init__(self, hidden_dim, bottleneck_dim, dropout=0.0, activation="gelu"):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, 1)
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up   = nn.Linear(bottleneck_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()

        # init gate bias to -2 so α starts near 0.12 — small initial edit
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, h):                                          # h: (B, d)
        alpha = torch.sigmoid(self.gate(h))                        # (B, 1)
        delta = self.up(self.dropout(self.act(self.down(h))))      # (B, d)
        return h + alpha * delta


class SASRecContextGateModel(nn.Module):
    """Frozen SASRec backbone + context-conditioned scalar gate adapter.

    use_gate=True  → ContextGateAdapter:   h̃ = h + σ(Wh+b) · MLP(h)
    use_gate=False → ResidualMLPAdapter:   h̃ = h + MLP(h)   (gate ablation)
    """

    def __init__(
        self,
        backbone: SASRec,
        bottleneck_dim: int,
        adapter_dropout: float = 0.0,
        adapter_activation: str = "gelu",
        freeze_backbone: bool = True,
        use_gate: bool = True,
    ):
        super().__init__()
        self.backbone  = backbone
        self.use_gate  = use_gate
        if use_gate:
            self.adapter = ContextGateAdapter(
                hidden_dim=backbone.hidden_units,
                bottleneck_dim=bottleneck_dim,
                dropout=adapter_dropout,
                activation=adapter_activation,
            )
        else:
            self.adapter = ResidualMLPAdapter(
                hidden_dim=backbone.hidden_units,
                bottleneck_dim=bottleneck_dim,
                dropout=adapter_dropout,
                activation=adapter_activation,
            )
        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad_(False)

    def get_adapted_hidden(self, input_ids):
        h = self.backbone.get_last_hidden(input_ids)  # (B, d)
        return self.adapter(h)                         # (B, d)

    def forward(self, input_ids, candidate_ids=None):
        h = self.get_adapted_hidden(input_ids)
        return self.backbone.score_from_hidden(h, candidate_ids)