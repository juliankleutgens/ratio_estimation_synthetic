import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np, random
import torchsummary
from torchinfo import (
    summary)
import torch.nn.functional as F
from typing import Union, Optional, Dict, Tuple
import math
import time
from tqdm.auto import tqdm
from models import ClassiferNet, RatioNetAdaLN
import wandb

class LogLinearNoise(nn.Module):
    """σ(t) = exp(log σ_min + t (log σ_max − log σ_min)),   t∈[0,1]"""
    def __init__(self, sigma_min: float = 1e-3, sigma_max: float = 50.0):
        super().__init__()
        self.register_buffer("sigma_min", torch.tensor(sigma_min))
        self.register_buffer("sigma_max", torch.tensor(sigma_max))
    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_sigma = torch.log(self.sigma_min) + t * (torch.log(self.sigma_max) - torch.log(self.sigma_min))
        sigma = log_sigma.exp()
        dsigma = sigma * (torch.log(self.sigma_max) - torch.log(self.sigma_min))  # dσ/dt
        return sigma, dsigma


def _sample_categorical(logits_or_probs: torch.Tensor) -> torch.Tensor:
    """Draws one sample along last dim using Gumbel–max (no gradients)."""
    if logits_or_probs.dtype in (torch.float16, torch.bfloat16):
        logits_or_probs = logits_or_probs.float()
    if logits_or_probs.min() < 0:  # assume logits
        logits = logits_or_probs
    else:                          # assume probs
        logits = logits_or_probs.log()
    gumbel = -torch.empty_like(logits).exponential_().log()  # −log U
    return (logits + gumbel).argmax(dim=-1)


def joint_sample_two_positions(log1: torch.Tensor,
                               log2: torch.Tensor,
                               max_chunk: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Jointly sample (x1,x2) for a batch using the log‑probs of
    position 0 and position 1.
    log1 , log2 : (B, V)      – logits or log‑probs for the two slots
    Returns
    -------
    x1, x2 : LongTensor (B,)  – drawn coordinate indices
    """
    B, V = log1.shape
    out1 = torch.empty(B,  dtype=torch.long, device=log1.device)
    out2 = torch.empty_like(out1)

    # process the batch in chunks to avoid allocating (B,V,V) at once
    for start in range(0, B, max_chunk):
        end   = min(start + max_chunk, B)

        # log P(i,j)  =  log p1(i) + log p2(j)
        joint_log = log1[start:end, :, None] + log2[start:end, None, :]  # (b, V, V)
        joint_log = joint_log.view(end-start, -1)                        # (b, V²)

        # convert to probabilities, draw one index, map back to (i,j)
        probs = joint_log.softmax(-1)
        idx   = torch.multinomial(probs, 1).squeeze(1)                  # (b,)

        out1[start:end] = idx // V
        out2[start:end] = idx %  V

    return out1, out2


class DiffusionConfig:
    batch_size: int = 512 * 4
    vocab_size: int
    seq_len: int
    mask_idx: Union[int, None] = None  # absorbing state index; if None use uniform diffusion
    T_sampling: int = 30  # # steps for deterministic sampler
    gamma: float = 2.0  # guidance strength (γ)
    use_approx: bool = True  # first‑order guidance if True
    end_time: float = 1e-5  # end time for sampling
    use_plg: bool = False  # use Posterior Logit Guidance (PLG) (positive logit guidance)
    eta: float = 1  # PLG strength (η)
    stochastic_sampling_jitter_mask: float = 0.0  # jitter mask probability (for absorbing state diffusion)
    stochastic_sampling_eps_noise: float = 0.0  # noise fraction (for uniform diffusion)
    k_best_sampling: int = 0  # k-best sampling (0 = no k-best sampling)



class Diffusion(nn.Module):
    """Discrete diffusion with ratio‑based guidance.

    * `denoiser` maps (xt, sigma_t) ➜ (B,L,V) **logits**.
    * `ratio_model` is a `RatioGuidance` wrapper.
    """

    def __init__(self, denoiser: nn.Module, ratio_model: RatioNetAdaLN, cfg: DiffusionConfig, use_approx: bool = False):
        super().__init__()
        self.denoiser = denoiser
        self.ratio_model = ratio_model.eval()
        self.cfg = cfg
        self.noise = LogLinearNoise()  # could be swapped
        self.vocab_size = cfg.vocab_size
        self.mask_idx = cfg.mask_idx if cfg.mask_idx is not None else self.vocab_size  # sentinel
        self.batch_size = cfg.batch_size
        self.end_time = cfg.end_time
        self.use_plg = cfg.use_plg
        self.k_best_sampling = cfg.k_best_sampling if hasattr(cfg, "k_best_sampling") else 0
        self.jitter_mask = cfg.stochastic_sampling_jitter_mask if hasattr(cfg, "stochastic_sampling_jitter_mask") else 0.0
        self.eps_noise = cfg.stochastic_sampling_eps_noise if hasattr(cfg, "stochastic_sampling_eps_noise") else 0.0
        self.use_approx = use_approx
        if cfg.mask_idx is None:
            self.diffusion = "uniform"
        else:
            self.diffusion = "absorbing_state"

    # ---------------------------------------------------------------------
    # forward: one network pass ⇒ log‑probs over vocabulary
    # ---------------------------------------------------------------------
    def forward(self, x: torch.LongTensor, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 1:
            sigma = sigma[:, None]
        logits = self.denoiser(x, sigma)  # (B,L,V)
        return logits.log_softmax(dim=-1)  # log‑probs

    def model_prediction(self, xt: torch.LongTensor, sigma_t: torch.Tensor) -> torch.Tensor:
        """
        Perform chunked denoiser forward passes to avoid MPS out-of-memory errors.

        Args:
            xt (torch.LongTensor): current sequences tensor of shape (B, L).
            sigma_t (torch.Tensor): noise levels tensor of shape (B,).
            max_chunk (int): maximum number of sequences per chunk.

        Returns:
            torch.Tensor: log-probabilities tensor of shape (B, L, V).
        """
        outputs = []
        B = xt.size(0)
        # Process in chunks
        for start in range(0, B, self.batch_size):
            end = start + self.batch_size
            x_chunk = xt[start:end]
            sigma_chunk = sigma_t[start:end]
            # Use the existing forward method for denoiser + log_softmax
            logits_chunk = self.forward(x_chunk, sigma_chunk)
            outputs.append(logits_chunk)
        # Concatenate all chunks back into a full batch
        return torch.cat(outputs, dim=0)

    # ---------------------------------------------------------------------
    def _compute_posterior(self, log_x_theta: torch.Tensor, xt: torch.LongTensor,
                           alpha_s: torch.Tensor, alpha_t: torch.Tensor) -> torch.Tensor:
        """Uniform diffusion posterior p(x_s | x_t, x_theta)."""
        # x_theta : probs . We get exp(...) outside.
        x_theta = log_x_theta.exp()
        xt_onehot = F.one_hot(xt, self.vocab_size)
        post = (
                (alpha_t * self.vocab_size * x_theta * xt_onehot +
                 (alpha_t / alpha_s - alpha_t) * xt_onehot +
                 (alpha_s - alpha_t) * x_theta +
                 (1 - alpha_t / alpha_s) * (1 - alpha_s)) /
                (alpha_t * self.vocab_size * torch.gather(x_theta, -1, xt[..., None]) + (1 - alpha_t))
        )
        return post

    def _get_ratio(self, xt: torch.LongTensor, t: torch.Tensor) -> torch.Tensor:
        """
        Return log-ratio scores rψ(xₜ,ℓ,v,t)  ∈ ℝ^{B×L×V}.

        • If cfg.use_approx == True  ➜ first-order Taylor approximation
          (one network pass, one backward pass, O(B·L) memory).
        • Otherwise                ➜ exact but expensive enumeration.

        The approximation follows Eq. (8) of the paper:
            rψ(xₜ[ℓ←v]) ≈ rψ(xₜ) + ⟨∇_{xₜ} rψ(xₜ), e_{ℓ,v} − e_{ℓ,xₜ[ℓ]}⟩
        """
        # ------------------------------------------------------------------ #
        # 1. Fast first-order approximation (one fwd + bwd)                  #
        # ------------------------------------------------------------------ #
        if self.use_approx:
            with torch.enable_grad():  # ← turn grads back on
                xt_onehot = F.one_hot(xt, self.vocab_size).float()
                xt_onehot.requires_grad_(True)

                base = self.ratio_model(xt_onehot, t)  # forward pass
                grad, = torch.autograd.grad(base.sum(), xt_onehot, create_graph=False)
            # grad  ≜ ∂ rψ / ∂ xₜ           shape: (B,L,V)

            # baseline scores broadcast to token dimension
            base = base[:, None, None]  # (B,1,1)

            # For each position, substitute current token → build delta
            idx = xt.unsqueeze(-1)  # (B,L,1)
            scatter_grad = grad.gather(-1, idx)  # grad at current tokens (B,L,1)

            # rψ(xₜ[ℓ←v]) ≈ base + grad_{ℓ,v} − grad_{ℓ,xℓ}
            log_ratio = base + grad - scatter_grad  # (B,L,V)
            # grad - scatter_grad   ↔ their classifier_log_prob_ratio
            # base + grad - scatter_grad  ↔ their classifier_log_prob

            with torch.no_grad():
                log_ratio_exact = self._batched_ratio(xt, t)
            # print a set of diagnostics of how different the two methods are
            if not torch.allclose(log_ratio, log_ratio_exact, atol=1e-3):
                wandb.log({"log_ratio_diff": (log_ratio.detach() - log_ratio_exact.detach()).abs().mean().item()})
                print(f"Warning: log_ratio and log_ratio_ differ significantly: "
                      f"{(log_ratio - log_ratio_exact).abs().mean().item():.3f} (mean abs diff)")
            wandb.log({"log_ratio_approx": log_ratio.mean().item(),})

        # ------------------------------------------------------------------ #
        # 2. Exact enumeration (vector or scalar model)                      #
        # ------------------------------------------------------------------ #
        else:
            if self.ratio_model.__class__.__name__ == "RatioNetAdaLNVector":
                log_ratio = self._batched_ratio_vector(xt, t)
            else:
                log_ratio = self._batched_ratio(xt, t)
            log_ratio_exact = log_ratio

        # ------------------------------------------------------- #
        # 3. Diagnostics (unchanged)                              #
        # ------------------------------------------------------- #
        mask_mean = log_ratio[..., self.mask_idx:self.mask_idx + 1].mean(-1, keepdim=True)
        token_mean = log_ratio[..., :-1].mean(-1, keepdim=True)

        wandb.log({"mask_mean": mask_mean.mean().item(),
                   "token_mean": token_mean.mean().item(),
                   "mask_token_diff": (token_mean - mask_mean).mean().item(),
                   "log_ratio_model": log_ratio_exact.mean().item()})
        return log_ratio

    @torch.no_grad()
    def _batched_ratio(self, xt, t):
        """Return log‑ratio tensor[B,L,V] without overloading GPURAM."""
        B, L = xt.shape
        N = B * L * self.vocab_size  # total candidates
        out = torch.empty(N, device=xt.device)

        # build the index tensor only once
        jump_idx = torch.arange(N, device=xt.device)
        pos = jump_idx // self.vocab_size  # [N]
        tok = jump_idx % self.vocab_size  # [N]
        t_expand = t.repeat_interleave(L * self.vocab_size)

        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)

            # slice indices for this mini‑batch
            b_slice = (pos[start:end] // L)  # which original seq
            p_slice = pos[start:end] % L  # which position in seq
            tok_slice = tok[start:end]

            # construct mutated sequences on the fly (no full xt_expand)
            seq_batch = xt[b_slice].clone()
            seq_batch[torch.arange(seq_batch.size(0), device=xt.device), p_slice] = tok_slice

            # forward pass
            out[start:end] = self.ratio_model(seq_batch,
                                              t_expand[start:end])

        return out.view(B, L, self.vocab_size)

    @torch.no_grad()
    def _batched_ratio_vector(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute rψ(xₜ, ℓ, t) for *every* position ℓ without blowing up GPU RAM.

        Args
        ----
        xt : LongTensor  – current noised sequences   (B, L)
        t  : Tensor      – times *per-sequence*       (B,)

        Returns
        -------
        log_ratio : Tensor  – log-ratio scores        (B, L, V)
                     for each sequence, each position, and every token.
        """
        B, L = xt.shape
        V    = self.vocab_size
        device = xt.device

        # where we'll write the per-position vectors
        log_ratio = torch.empty(B, L, V, device=device)

        # the ratio net is already placed on the correct device & in eval() by __init__
        # We stream the computation over positions and over batch chunks
        # to keep peak memory <= self.batch_size × V floats.
        for pos in range(L):
            # A position tensor (B,) filled with the current position index
            pos_vec_full = torch.full((B,), pos, dtype=torch.long, device=device)

            # Optional: split over the *sequence* dimension as well
            for start in range(0, B, self.batch_size):
                end   = min(start + self.batch_size, B)

                # Slice the current chunk
                x_chunk   = xt[start:end]                # (b′, L)
                pos_chunk = pos_vec_full[start:end]      # (b′,)
                t_chunk   = t[start:end]                 # (b′,)

                # Forward pass – returns (b′, V)
                vec = self.ratio_model(x_chunk, pos_chunk, t_chunk)

                # Write into the correct slot of the output tensor
                log_ratio[start:end, pos, :] = vec

        return log_ratio

    def _rbg_denoise_stochastic(
        self,
        xt: torch.LongTensor,
        t: torch.Tensor,
        dt: float,
        cache: Union[dict, None]
    ) -> tuple[torch.LongTensor, torch.Tensor, dict]:
        """
        DDIM‑style reverse step with an optional stochastic sub‑step.

        New cfg fields
        --------------
        cfg.eps_noise   ∈ [0,1]   – fraction of stochasticity (ε = 0 → deterministic)
        cfg.jitter_mask ∈ [0,1]   – Bernoulli prob. for simple remask “jitter”
                                    (set exactly one of the two > 0)
        """

        # ------------------------------------------------------------------
        # 0 · Optional stochastic forward move
        # ------------------------------------------------------------------
        if self.jitter_mask > 0 and self.diffusion == "absorbing_state":
            p = (self.jitter_mask * t[0])**2 # (B,1) noise level
            jump = torch.rand_like(xt.float()) < p        # (B,L) boolean
            xt = torch.where(jump, torch.full_like(xt, self.mask_idx), xt)
            cache = None                                  # xt changed ⇒ invalidate cache

        elif self.eps_noise > 0:
            eps = self.eps_noise
            t_mid = t - eps * dt                          # small forward step
            # for absorbing‑state diffusion a forward jump is just remasking
            if self.diffusion == "absorbing_state":
                jump = torch.rand_like(xt.float()) < eps
                xt = torch.where(jump, torch.full_like(xt, self.mask_idx), xt)
                cache = None
            # for uniform diffusion add Gaussian noise (not shown here)

            dt = dt + eps * dt                         # deterministic share

        # ------------------------------------------------------------------
        # 1 · Retrieve / compute model predictions
        # ------------------------------------------------------------------
        if cache is not None and torch.allclose(t, cache["t"]):
            log_x_theta = cache["log_x_theta"]
            log_ratio   = cache["log_ratio"]
        else:
            sigma_t, _   = self.noise(t)
            log_x_theta  = self.model_prediction(xt, sigma_t)

            log_ratio = torch.zeros_like(log_x_theta)
            if self.cfg.gamma > 0:
                log_ratio = self._get_ratio(xt, t)

            cache = {"log_x_theta": log_x_theta,
                     "log_ratio":   log_ratio,
                     "t":           t.clone()}

        # ------------------------------------------------------------------
        # 2 · Construct posterior q(x_{t‑dt} | x_t, x̂_θ)
        # ------------------------------------------------------------------
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)

        move_t = 1 - torch.exp(-sigma_t)          # (B,1)
        move_s = 1 - torch.exp(-sigma_s)
        move_t = move_t[:, None, None]
        move_s = move_s[:, None, None]

        if self.diffusion == "absorbing_state":
            q_log = log_x_theta + torch.log1p(-(move_s / move_t))
            q_log[..., self.mask_idx] = torch.log(move_s / move_t)[:, :, 0]
        else:                                      # uniform diffusion
            q_log = self._compute_posterior(
                        log_x_theta, xt,
                        1 - move_s, 1 - move_t
                    ).log()

        # ------------------------------------------------------------------
        # 3 · PLG guidance and sampling
        # ------------------------------------------------------------------
        if self.use_plg:
            log_ratio = log_ratio - self.cfg.eta * torch.log1p(torch.exp(log_ratio))

        guided_log   = q_log + self.cfg.gamma * log_ratio
        guided_probs = guided_log.softmax(dim=-1)
        xs           = _sample_categorical(guided_probs)

        if self.diffusion == "absorbing_state":
            xs = torch.where((xt != self.mask_idx), xt, xs)   # keep real tokens

        return xs, guided_probs, cache


    # ---------------------------------------------------------------------
    # denoising step with ratio guidance
    # ---------------------------------------------------------------------
    def _rbg_denoise(self, xt: torch.LongTensor, t: torch.Tensor, dt: float, cache: Union[dict, None]) -> tuple[
        torch.LongTensor, torch.Tensor, dict]:


        if cache is not None and torch.allclose(t, cache["t"]):
            log_x_theta = cache["log_x_theta"]
            log_ratio = cache["log_ratio"]
        else:
            # ------- model prediction -------
            sigma_t, _ = self.noise(t)
            log_x_theta = self.model_prediction(xt, sigma_t)

            # ------- ratio model -------
            log_ratio = torch.zeros_like(log_x_theta)
            if self.cfg.gamma > 0:
                log_ratio = self._get_ratio(xt, t)

            cache = {"log_x_theta": log_x_theta, "log_ratio": log_ratio, "t": t.clone()}

        # construct posterior log‑probs
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        move_t = 1 - torch.exp(-sigma_t)  # (B,1)
        move_s = 1 - torch.exp(-sigma_s)
        move_t = move_t[:, None, None]
        move_s = move_s[:, None, None]

        if self.diffusion == "absorbing_state":
            q_log = log_x_theta + torch.log1p(-(move_s / move_t))
            q_log[..., self.mask_idx] = torch.log(move_s / move_t)[:, :, 0]
        else:  # uniform
            q_log = self._compute_posterior(log_x_theta, xt, 1 - move_s, 1 - move_t).log()

        # Results from week 8
        if self.use_plg:
            log_ratio = log_ratio - self.cfg.eta * torch.log(1 + torch.exp(log_ratio))
        guided_log = q_log + self.cfg.gamma * log_ratio

        # --- forbid mask where we already have a real token ---------------
        if self.k_best_sampling > 0:
            # k-best sampling: keep the top-k tokens at each position
            k = self.k_best_sampling
            topk_vals, topk_ids = guided_log.topk(k, dim=-1)
            masked_log = guided_log.new_full(guided_log.shape, -1e9)  # −∞ elsewhere
            guided_log = masked_log.scatter(-1, topk_ids, topk_vals)  # keep top‑k only

        guided_probs = guided_log.softmax(dim=-1)
        xs = _sample_categorical(guided_probs)
        if self.diffusion == "absorbing_state":
            copy_flag = (xt != self.mask_idx)
            xs = torch.where(copy_flag, xt, xs)
        return xs, guided_probs, cache

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, num_of_samples: int, device: Union[torch.device, str] = "cpu") -> torch.LongTensor:
        """Deterministic DDPM‑like sampler (fixed #steps) with RBG."""
        device = torch.device(device)
        xt = (torch.full((num_of_samples, self.cfg.seq_len), self.mask_idx, dtype=torch.long, device=device)
              if self.diffusion == "absorbing_state"
              else torch.randint(self.vocab_size, (num_of_samples, self.cfg.seq_len), device=device))

        timesteps = torch.linspace(1.0, self.cfg.end_time, self.cfg.T_sampling + 1, device=device)
        dt = (1.0 - self.cfg.end_time) / self.cfg.T_sampling

        cache = None

        for i in tqdm(range(self.cfg.T_sampling), desc="Sampling", unit="step", total=self.cfg.T_sampling):
            self.i = i
            t = timesteps[i].expand(num_of_samples)
            xt, _, cache = self._rbg_denoise(xt, t, dt, cache)
            # optional: clear cache every step if speed not critical
        return xt  # (B,L)

    def sample_trace(self,
                     num_samples: int,
                     device: Union[str, torch.device] = "cpu"
                     ) -> torch.LongTensor:
        """
        Return a full trajectory tensor with shape
            (T_sampling+1,num_samples,seq_len)

        traj[0] is the initial noise / mask state,
        traj[-1] is the final denoised sequence.
        """
        device = torch.device(device)
        T = self.cfg.T_sampling
        traj = torch.empty(T + 1, num_samples, self.cfg.seq_len,
                           dtype=torch.long, device=device)

        # --- initialise x_T -------------------------------------------------
        xt = (torch.full((num_samples, self.cfg.seq_len), self.mask_idx,
                         dtype=torch.long, device=device)
              if self.diffusion == "absorbing_state"
              else torch.randint(self.vocab_size,
                                 (num_samples, self.cfg.seq_len), device=device))
        traj[0] = xt

        # --- deterministic DDPM schedule -----------------------------------
        timesteps = torch.linspace(1.0, self.cfg.end_time, self.cfg.T_sampling, device=device)
        dt = (1.0 - self.cfg.end_time) / self.cfg.T_sampling
        cache = None

        for i in range(T):
            t = timesteps[i].expand(num_samples)
            xt, _, cache = self._rbg_denoise(xt, t, dt, cache)
            traj[i + 1] = xt  # save x_{t‑1}

        return traj.cpu()  # (T+1, B, L)

