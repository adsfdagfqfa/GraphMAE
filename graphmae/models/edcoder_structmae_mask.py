import torch

from .edcoder import PreModel


class StructMAEMaskPreModel(PreModel):
    """GraphMAE with StructMAE-style easy-to-hard structure-guided masking."""

    def __init__(
        self,
        *args,
        structmae_beta=0.25,
        structmae_score="pagerank",
        structmae_iters=20,
        structmae_damping=0.85,
        structmae_eps=1e-12,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if structmae_beta < 0.0:
            raise ValueError("structmae_beta must be non-negative.")
        if structmae_score not in ("pagerank", "degree"):
            raise ValueError("structmae_score must be 'pagerank' or 'degree'.")
        self._structmae_beta = structmae_beta
        self._structmae_score = structmae_score
        self._structmae_iters = structmae_iters
        self._structmae_damping = structmae_damping
        self._structmae_eps = structmae_eps
        self._current_epoch = None
        self._current_max_epoch = None

    def set_mask_epoch(self, epoch, max_epoch):
        self._current_epoch = epoch
        self._current_max_epoch = max_epoch

    def _schedule_scale(self):
        if self._current_epoch is None or self._current_max_epoch is None or self._current_max_epoch <= 1:
            return 1.0
        progress = min(max(float(self._current_epoch + 1) / float(self._current_max_epoch), 0.0), 1.0)
        return progress ** 0.5

    def _pagerank_scores(self, g, x):
        cache_name = "_graphmae_structmae_pagerank_scores"
        cached = getattr(g, cache_name, None)
        if cached is not None and cached.device == x.device and cached.numel() == g.num_nodes():
            return cached

        num_nodes = g.num_nodes()
        device = x.device
        if num_nodes == 0:
            return torch.empty(0, device=device)

        src, dst = g.edges()
        src = src.to(device)
        dst = dst.to(device)

        rank = torch.full((num_nodes,), 1.0 / num_nodes, device=device)
        if src.numel() == 0:
            setattr(g, cache_name, rank)
            return rank

        out_degree = torch.zeros(num_nodes, device=device)
        out_degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
        out_degree = out_degree.clamp_min(1.0)
        base = (1.0 - self._structmae_damping) / num_nodes

        for _ in range(self._structmae_iters):
            next_rank = torch.full((num_nodes,), base, device=device)
            message = rank[src] / out_degree[src]
            next_rank.scatter_add_(0, dst, self._structmae_damping * message)
            rank = next_rank

        rank = rank.clamp_min(self._structmae_eps)
        setattr(g, cache_name, rank)
        return rank

    def _structure_scores(self, g, x):
        if self._structmae_score == "degree":
            return g.in_degrees().to(device=x.device, dtype=torch.float)
        return self._pagerank_scores(g, x)

    def _sample_structmae_mask_nodes(self, g, x, num_mask_nodes):
        num_nodes = g.num_nodes()
        device = x.device

        if num_mask_nodes <= 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, torch.arange(num_nodes, device=device)
        if num_mask_nodes >= num_nodes:
            mask_nodes = torch.arange(num_nodes, device=device)
            keep_nodes = torch.empty(0, dtype=torch.long, device=device)
            return mask_nodes, keep_nodes

        scores = self._structure_scores(g, x)
        topk_size = int(round(num_mask_nodes * self._schedule_scale()))
        topk_size = min(max(topk_size, 0), num_nodes)

        gamma = torch.rand(num_nodes, device=device)
        if topk_size > 0:
            topk_nodes = torch.topk(scores, k=topk_size, largest=True).indices
            gamma[topk_nodes] += self._structmae_beta

        mask_nodes = torch.topk(gamma, k=num_mask_nodes, largest=True).indices
        keep_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        keep_mask[mask_nodes] = False
        keep_nodes = keep_mask.nonzero(as_tuple=False).view(-1)

        return mask_nodes, keep_nodes

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes, keep_nodes = self._sample_structmae_mask_nodes(g, x, num_mask_nodes)

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            if num_noise_nodes > 0:
                noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]
                noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]
            else:
                noise_nodes = torch.empty(0, dtype=torch.long, device=x.device)
                noise_to_be_chosen = torch.empty(0, dtype=torch.long, device=x.device)

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)
