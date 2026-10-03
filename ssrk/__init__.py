"""Self-Supervised Reconstruction Knockoffs (SSRK): reference implementation for the NeurIPS 2026 paper.

Submodules
----------
gates, determinism        shared numerics (exactly odd tanh, deterministic kernels, seeding)
ft_config                 FixedTargetConfig (torch-free; re-exported by fixed_target)
fixed_target              fixed-target mode for controlled discovery (Table A1)
gaussian_knockoffs        exact Gaussian knockoff sampler (eq. (3))
knockoff_plus             knockoff+ threshold (eq. (8))
selfrecon                 self-reconstruction (ranking) mode (App. A.3.1)
references                matched-marginal references (independent Gaussian marginal, within-column permutation)
ranking                   deterministic top-k ranking

Submodules are imported explicitly (e.g. ``from ssrk.fixed_target import fit_fixed_target_batched``); importing
the package itself does not import torch.
"""
