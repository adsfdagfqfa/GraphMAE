import torch

from .edcoder_pagerank_mask import PageRankCurriculumMaskPreModel


class PageRankPreserveCurriculumMaskPreModel(PageRankCurriculumMaskPreModel):
    """GraphMAE masking that preserves high-PageRank nodes as context.

    The curriculum controls how much of the mask budget is sampled from
    inverse-normalized PageRank scores. Early epochs stay close to random
    masking; later epochs increasingly prefer low-PageRank nodes, leaving
    high-PageRank nodes visible as reconstruction context.
    """

    def __init__(self, *args, pagerank_preserve_power=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if pagerank_preserve_power <= 0.0:
            raise ValueError("pagerank_preserve_power must be positive.")
        self._pagerank_preserve_power = pagerank_preserve_power

    def _pagerank_preserve_weights(self, g, x):
        scores = self._pagerank_scores(g, x).clone()
        if scores.numel() == 0:
            return scores

        score_min = scores.min()
        score_max = scores.max()
        if (score_max - score_min) <= self._pagerank_eps:
            return torch.ones_like(scores)

        normalized = (scores - score_min) / (score_max - score_min)
        weights = (1.0 - normalized).clamp_min(self._pagerank_eps)
        return weights.pow(self._pagerank_preserve_power)

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
        num_preserve_nodes = int(round(num_mask_nodes * curriculum_ratio))
        num_random_nodes = num_mask_nodes - num_preserve_nodes

        if num_preserve_nodes > 0:
            weights = self._pagerank_preserve_weights(g, x)
            if weights.sum() <= 0:
                preserve_nodes = torch.empty(0, dtype=torch.long, device=device)
            else:
                preserve_nodes = torch.multinomial(
                    weights / weights.sum(),
                    num_samples=num_preserve_nodes,
                    replacement=False,
                )
        else:
            preserve_nodes = torch.empty(0, dtype=torch.long, device=device)

        selected = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        selected[preserve_nodes] = True

        remaining_nodes = (~selected).nonzero(as_tuple=False).view(-1)
        perm = torch.randperm(remaining_nodes.numel(), device=device)
        random_nodes = remaining_nodes[perm[:num_random_nodes]]
        mask_nodes = torch.cat([preserve_nodes, random_nodes], dim=0)

        keep_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        keep_mask[mask_nodes] = False
        keep_nodes = keep_mask.nonzero(as_tuple=False).view(-1)

        return mask_nodes, keep_nodes
