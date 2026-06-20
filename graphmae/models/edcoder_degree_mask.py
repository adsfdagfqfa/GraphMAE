import torch

from .edcoder import PreModel


class DegreeMixedMaskPreModel(PreModel):
    """GraphMAE with mixed random and degree-aware node masking."""

    def __init__(self, *args, degree_mask_ratio=0.3, degree_mask_power=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 <= degree_mask_ratio <= 1.0:
            raise ValueError("degree_mask_ratio must be in [0, 1].")
        self._degree_mask_ratio = degree_mask_ratio
        self._degree_mask_power = degree_mask_power

    def _sample_mixed_mask_nodes(self, g, x, num_mask_nodes):
        num_nodes = g.num_nodes()
        device = x.device

        if num_mask_nodes <= 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, torch.arange(num_nodes, device=device)
        if num_mask_nodes >= num_nodes:
            mask_nodes = torch.arange(num_nodes, device=device)
            keep_nodes = torch.empty(0, dtype=torch.long, device=device)
            return mask_nodes, keep_nodes

        num_degree_nodes = int(round(num_mask_nodes * self._degree_mask_ratio))
        num_random_nodes = num_mask_nodes - num_degree_nodes

        perm = torch.randperm(num_nodes, device=device)
        mask_nodes = perm[:num_random_nodes]

        if num_degree_nodes > 0:
            selected = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            selected[mask_nodes] = True

            degrees = g.in_degrees().to(device=device, dtype=torch.float)
            if self._degree_mask_power != 1.0:
                degrees = degrees.clamp_min(0).pow(self._degree_mask_power)

            weights = degrees + 1e-12
            weights[selected] = 0.0

            if weights.sum() <= 0:
                remaining = perm[num_random_nodes:]
                degree_nodes = remaining[:num_degree_nodes]
            else:
                degree_nodes = torch.multinomial(
                    weights / weights.sum(),
                    num_samples=num_degree_nodes,
                    replacement=False,
                )
            mask_nodes = torch.cat([mask_nodes, degree_nodes], dim=0)

        keep_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        keep_mask[mask_nodes] = False
        keep_nodes = keep_mask.nonzero(as_tuple=False).view(-1)

        return mask_nodes, keep_nodes

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes, keep_nodes = self._sample_mixed_mask_nodes(g, x, num_mask_nodes)

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
