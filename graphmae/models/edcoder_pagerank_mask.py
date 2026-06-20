import torch

from .edcoder import PreModel


class PageRankCurriculumMaskPreModel(PreModel):
    """GraphMAE with PageRank-guided masking that ramps up during training."""

    def __init__(
        self,
        *args,
        pagerank_mask_ratio=0.25,
        pagerank_mask_steps=5,
        pagerank_iters=20,
        pagerank_damping=0.85,
        pagerank_eps=1e-12,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not 0.0 <= pagerank_mask_ratio <= 1.0:
            raise ValueError("pagerank_mask_ratio must be in [0, 1].")
        if pagerank_mask_steps <= 0:
            raise ValueError("pagerank_mask_steps must be positive.")
        self._pagerank_mask_ratio = pagerank_mask_ratio
        self._pagerank_mask_steps = pagerank_mask_steps
        self._pagerank_iters = pagerank_iters
        self._pagerank_damping = pagerank_damping
        self._pagerank_eps = pagerank_eps
        self._current_epoch = None
        self._current_max_epoch = None

    def set_mask_epoch(self, epoch, max_epoch):
        self._current_epoch = epoch
        self._current_max_epoch = max_epoch

    def _curriculum_ratio(self):
        if self._current_epoch is None or self._current_max_epoch is None or self._current_max_epoch <= 1:
            return self._pagerank_mask_ratio

        progress = min(max(float(self._current_epoch + 1) / float(self._current_max_epoch), 0.0), 1.0)
        step = min(int(progress * self._pagerank_mask_steps), self._pagerank_mask_steps)
        return self._pagerank_mask_ratio * step / float(self._pagerank_mask_steps)

    def _pagerank_scores(self, g, x):
        cache_name = "_graphmae_pagerank_scores"
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
        base = (1.0 - self._pagerank_damping) / num_nodes

        for _ in range(self._pagerank_iters):
            next_rank = torch.full((num_nodes,), base, device=device)
            message = rank[src] / out_degree[src]
            next_rank.scatter_add_(0, dst, self._pagerank_damping * message)
            rank = next_rank

        rank = rank.clamp_min(self._pagerank_eps)
        setattr(g, cache_name, rank)
        return rank

    def _sample_pagerank_curriculum_mask_nodes(self, g, x, num_mask_nodes):
        num_nodes = g.num_nodes()
        device = x.device

        if num_mask_nodes <= 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, torch.arange(num_nodes, device=device)
        if num_mask_nodes >= num_nodes:
            mask_nodes = torch.arange(num_nodes, device=device)
            keep_nodes = torch.empty(0, dtype=torch.long, device=device)
            return mask_nodes, keep_nodes

        curriculum_ratio = self._curriculum_ratio()
        num_pagerank_nodes = int(round(num_mask_nodes * curriculum_ratio))
        num_random_nodes = num_mask_nodes - num_pagerank_nodes

        if num_pagerank_nodes > 0:
            weights = self._pagerank_scores(g, x).clone()
            if weights.sum() <= 0:
                pagerank_nodes = torch.empty(0, dtype=torch.long, device=device)
            else:
                pagerank_nodes = torch.multinomial(
                    weights / weights.sum(),
                    num_samples=num_pagerank_nodes,
                    replacement=False,
                )
        else:
            pagerank_nodes = torch.empty(0, dtype=torch.long, device=device)

        selected = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        selected[pagerank_nodes] = True

        remaining_nodes = (~selected).nonzero(as_tuple=False).view(-1)
        perm = torch.randperm(remaining_nodes.numel(), device=device)
        random_nodes = remaining_nodes[perm[:num_random_nodes]]
        mask_nodes = torch.cat([pagerank_nodes, random_nodes], dim=0)

        keep_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        keep_mask[mask_nodes] = False
        keep_nodes = keep_mask.nonzero(as_tuple=False).view(-1)

        return mask_nodes, keep_nodes

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes, keep_nodes = self._sample_pagerank_curriculum_mask_nodes(g, x, num_mask_nodes)

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
