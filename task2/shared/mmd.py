"""
Maximum Mean Discrepancy (MMD) with a sum of RBF kernels, per spec:
bandwidths = 0.5x, 1x, 2x the median pairwise squared distance in the CURRENT combined batch
(source+target features together). Shared by DAN (Task 2) and DAN-DG (Task 3) so both
methods use the identical discrepancy measure.
"""
import torch


def pairwise_sq_dists(x):
    # x: (N, D) -> (N, N) squared Euclidean distances
    sq = (x ** 2).sum(dim=1, keepdim=True)
    return sq + sq.t() - 2 * x @ x.t()


def rbf_kernel_sum(x, y, bandwidth_multipliers=(0.5, 1.0, 2.0)):
    """x: (Ns, D) source features, y: (Nt, D) target features (or another domain's features).
    Returns the combined (Ns+Nt, Ns+Nt) kernel matrix, summed across the 3 bandwidths.
    Bandwidth = multiplier * median pairwise squared distance of the COMBINED batch."""
    combined = torch.cat([x, y], dim=0)
    n = combined.size(0)
    dists = pairwise_sq_dists(combined)

    # median of the off-diagonal distances (exclude the zero diagonal)
    off_diag = dists[~torch.eye(n, dtype=torch.bool, device=dists.device)]
    median_dist = off_diag.median().clamp(min=1e-8)

    kernel_sum = torch.zeros_like(dists)
    for mult in bandwidth_multipliers:
        bandwidth = mult * median_dist
        kernel_sum = kernel_sum + torch.exp(-dists / (2 * bandwidth))
    return kernel_sum, n


def mmd_loss(source_feat, target_feat, bandwidth_multipliers=(0.5, 1.0, 2.0)):
    """Squared MMD between source_feat (Ns, D) and target_feat (Nt, D) in feature space,
    using the kernel trick (never materializes the explicit feature map phi)."""
    kernel_sum, n = rbf_kernel_sum(source_feat, target_feat, bandwidth_multipliers)
    ns = source_feat.size(0)
    nt = target_feat.size(0)

    K_ss = kernel_sum[:ns, :ns]
    K_tt = kernel_sum[ns:, ns:]
    K_st = kernel_sum[:ns, ns:]

    loss = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
    return loss
