import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np, random
import torchsummary
from torchinfo import (
    summary)
from typing import Union, Optional, Dict, Tuple
import math
import time
from tqdm.auto import tqdm
from utils import hamming_matrix
from diffusion import LogLinearNoise
from inspect import signature
import torch.nn.functional as F
import wandb

# -----------------------------------------------------------------------
# 0. Laplacian penalty for smoothness
# -----------------------------------------------------------------------
def laplacian_penalty(preds: torch.Tensor,
                      seqs: torch.Tensor,
                      radius: int = 1) -> torch.Tensor:
    """
    preds : [B]  real     - model outputs r_ϕ(x) (or log‑ratio)
    seqs  : [B,L] long    – same mini‑batch the preds came from
    radius: int           – connect edges with Hamming distance ≤ radius
    returns a scalar penalty  λ · Σ_(i~j) (pred_i − pred_j)² / |E|
    """
    # 1. build adjacency mask ─────────────────────────────────────────
    D   = hamming_matrix(seqs)                 # [B,B]  integer
    mask = (D > 0) & (D <= radius)             # exclude self‑edges

    # 2. compute squared diffs only for edges in the mask ────────────
    diff = preds.unsqueeze(1) - preds.unsqueeze(0)   # [B,B]
    if mask.any():
        penalty = (diff[mask] ** 2).mean()
    else:                                            # rare if batch very small
        penalty = torch.tensor(0., device=preds.device)

    return penalty

# =====================================================================
# 3. Forward diffusion for discrete sequences
# =====================================================================
def corrupt(
    x0: torch.LongTensor,
    sigma: torch.Tensor,
    *,
    diffusion: str = "absorbing_state",
    vocab_size: Optional[int] = None,
    mask_idx: Optional[int] = None,
):
    """Forward‑diffuse *discrete* sequences.

    Parameters
    ----------
    x0         : (B, L) long        – clean tokens
    sigma      : (B,)  float        – noise level produced by noise schedule
    diffusion  : "absorbing_state" | "uniform"
    vocab_size : required if `diffusion == 'uniform'`
    mask_idx   : required if `diffusion == 'absorbing_state'`

    Returns
    -------
    x_t        : (B, L) long        – corrupted sequence at time‑step *t*
    move_mask  : (B, L) bool        – True where corruption happened
    """
    move_prob = 1 - torch.exp(-sigma)              # (B,)
    B, L = x0.shape
    move_mask = (torch.rand_like(x0.float()) < move_prob[:, None])

    if diffusion == "absorbing_state":
        assert mask_idx is not None, "mask_idx must be set for absorbing‑state diffusion"
        x_t = x0.clone()
        x_t[move_mask] = mask_idx
    else:  # uniform diffusion
        assert vocab_size is not None, "vocab_size must be set for uniform diffusion"
        x_t = x0.clone()
        x_t[move_mask] = torch.randint(0, vocab_size, (move_mask.sum(),), device=x0.device)

    return x_t, move_mask


# ==================================================================================
# 1. Train time‑independent classifier  d_ω(x)
# ==================================================================================
def train_domain_classifier(model, source_data, target_data, epochs=10, batch_size=256, lr=1e-4, eps=0.1, device="cpu"
                            ,unbalance_data=False, classifier_output_with_sigmoid=False):
    """Train a domain classifier ``d_ω(x)``.

    Parameters
    ----------
    model : nn.Module
        Neural network returning a single logit or probability.
    source_data : torch.Tensor
        Samples from the source domain labelled as ``1``.
    target_data : torch.Tensor
        Samples from the target domain labelled as ``0``.
    epochs : int, optional
        Number of training epochs.
    batch_size : int, optional
        Mini batch size.
    lr : float, optional
        Learning rate for Adam.
    device : str or torch.device, optional
        Device used for training.
    unbalance_data : bool, optional
        If ``True`` and ``classifier_output_with_sigmoid`` is ``False`` the loss
        is weighted by ``pos_weight = len(target_data)/len(source_data)`` to
        compensate class imbalance.
    classifier_output_with_sigmoid : bool, optional
        When ``True`` the model output is assumed to already be passed through a
        sigmoid and ``BCELoss`` is used instead of ``BCEWithLogitsLoss``.

    Returns
    -------
    nn.Module
        The trained classifier moved back to CPU.
    """

    source_labels = torch.ones(source_data.size(0), dtype=torch.float32)
    target_labels = torch.zeros(target_data.size(0), dtype=torch.float32)

    dataset = TensorDataset(torch.cat([source_data, target_data]), torch.cat([source_labels, target_labels]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-4, lr=lr)
    if classifier_output_with_sigmoid:
        bce_loss = nn.BCELoss()
    else:
        pos_weight = None
        if unbalance_data:
            pos_weight = torch.tensor(len(target_data) / len(source_data), device=device)
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        for batch in loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # compute binary cross-entropy loss
            loss = bce_loss(outputs, labels.unsqueeze(-1))
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 1 == 0 and (epoch + 1) > 0:
            print(f"[Classifier] epoch {epoch + 1}/{epochs}, loss={loss.item():.4f}")
            wandb.log({"Classifier/train_loss": loss.item(),"epoch": epoch + 1,})
    model.to('cpu')
    return model

# =====================================================================
# 7. Training the time-dependent domain classifier from the TLDM paper (Pseudo-Code 2 in the Appendix)
# =====================================================================
def train_time_dependent_classifier(
    model: nn.Module,
    source_data: torch.Tensor,
    target_data: torch.Tensor,
    noise_sched: nn.Module = LogLinearNoise(),
    *,
    diffusion: str = "absorbing_state",
    mask_idx: Optional[int] = None,
    vocab_size: Optional[int] = None,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-4,
    device: Union[str, torch.device] = "cpu",
    unbalance_data: bool = True,  # if True, balance source and target data
    classifier_output_with_sigmoid: bool = False,  # if True, use sigmoid output
):
    """Train the time dependent classifier ``d_ω(x_t, t)``.

    A random time step ``t`` is sampled for each sequence which is then corrupted
    according to ``diffusion``. The model is optimised with binary
    cross-entropy to predict whether a noisy sequence originates from the source
    (label ``1``) or the target (label ``0``).

    When ``unbalance_data`` is ``True`` and logits are used, the positive class
    is weighted by ``pos_weight = len(target_data)/len(source_data)`` to reduce
    bias from highly imbalanced datasets.

    Returns the trained model on CPU.
    """
    # 1. prepare data loader
    labels_src = torch.ones(source_data.size(0), dtype=torch.float32)
    labels_tgt = torch.zeros(target_data.size(0), dtype=torch.float32)
    dataset = TensorDataset(
        torch.cat([source_data, target_data], dim=0),
        torch.cat([labels_src, labels_tgt], dim=0)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 2. move model & setup optimizer + loss
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if classifier_output_with_sigmoid:
        bce_loss = nn.BCELoss()
    else:
        pos_weight = None
        if unbalance_data:
            pos_weight = torch.tensor(len(target_data) / len(source_data), device=device)
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 3. training loop
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for x0, labels in loader:
            x0 = x0.to(device)
            labels = labels.to(device).unsqueeze(-1)

            B = x0.size(0)
            # 3.1 sample time & compute noise
            t = torch.rand(B, device=device)                  # U(0,1)
            sigma_t, _ = noise_sched(t)                       # σ(t)

            # 3.2 corrupt to x_t
            x_t, _ = corrupt(
                x0,
                sigma_t,
                diffusion=diffusion,
                mask_idx=mask_idx,
                vocab_size=vocab_size,
            )

            # 3.3 forward + loss
            outputs = model(x_t, t)                             # [B,1]
            loss = bce_loss(outputs.unsqueeze(-1), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * B

        avg = epoch_loss / len(loader.dataset)
        print(f"[Time‐Cond Classifier] epoch {epoch}/{epochs}, BCE={avg:.4f}")
        wandb.log({"Time-Dependent Classifier/train_loss": avg, "epoch": epoch})

    model.to("cpu")
    return model

# =====================================================================
# 4.  Validation for the domain-classifier dω
# =====================================================================
@torch.no_grad()
def validate_domain_classifier(
    classifier: nn.Module,
    src_val: torch.Tensor,
    tgt_val: torch.Tensor,
    batch_size: int  = 512,
    device: str      = "cpu",
):
    """
    Draws new sequences from P (label=1) and Q (label=0) and evaluates
    the classifier.  Prints and returns a dict with loss/accuracy stats.
    """
    classifier.eval().to(device)
    bce = nn.BCELoss(reduction="sum")   # we'll divide later

    # --------------------------------------------------------------
    # 1.  Fresh validation data
    # --------------------------------------------------------------
    seqs    = torch.cat([src_val, tgt_val])          # [N,L]
    labels  = torch.cat([torch.ones(src_val.shape[0]),  # 1 = source
                         torch.zeros(tgt_val.shape[0])])    # 0 = target

    loader = DataLoader(TensorDataset(seqs, labels),
                        batch_size=batch_size,
                        shuffle=False)

    # --------------------------------------------------------------
    # 2.  Forward pass  +  metrics
    # --------------------------------------------------------------
    total_loss = 0.0
    TP = FP = TN = FN = 0

    for batch_seqs, batch_labels in loader:
        batch_seqs  = batch_seqs.to(device)
        batch_labels = batch_labels.to(device)

        # input size
        n_inputs = len(signature(classifier.forward).parameters)
        if n_inputs == 2:
            t = torch.zeros(batch_seqs.shape[0], device=device)  # dummy time-step
            probs = classifier (batch_seqs, t).squeeze(-1)        # [B]
        else:
            probs = classifier(batch_seqs).squeeze(-1)   # [B]
        total_loss += bce(probs, batch_labels).item()

        preds = (probs >= 0.5).float()
        TP += ((preds == 1) & (batch_labels == 1)).sum().item()
        FP += ((preds == 1) & (batch_labels == 0)).sum().item()
        TN += ((preds == 0) & (batch_labels == 0)).sum().item()
        FN += ((preds == 0) & (batch_labels == 1)).sum().item()

    N = src_val.shape[0] + tgt_val.shape[0]
    acc  = (TP + TN) / N
    loss = total_loss / N

    which_classifer = "[Time-Dependent Classifier-Val]" if n_inputs == 2 else "[Classifier-Val]"
    print(f"{which_classifer}  loss = {loss:.4f}  acc = {acc:.2%} "
          f"TP={TP} FP={FP} TN={TN} FN={FN}")
    wandb.log({
        "Classifier-Val/loss": loss,
        "Classifier-Val/accuracy": acc,
        "Classifier-Val/TP": TP,
        "Classifier-Val/FP": FP,
        "Classifier-Val/TN": TN,
        "Classifier-Val/FN": FN,
    })

    return {
        "bce": loss,
        "accuracy": acc,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
    }




# =====================================================================
# 4. Train the denoiser p(x₀ | x_t, σ_t)
# =====================================================================
def train_denoiser(
    denoiser: nn.Module,
    train_seqs: torch.Tensor,                          # (N, L) integer tokens
    val_seqs: Union[torch.Tensor, None] = None,
    *,
    diffusion: str = "absorbing_state",               # "absorbing_state" | "uniform"
    mask_idx: Union[int, None] = None,                # required for absorbing_state
    vocab_size: int,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 3e-4,
    noise_sched = LogLinearNoise(),                   # can plug your own
    pad_val: Union[int, None] = None,                 # if you need to ignore PAD in CE
    device: Union[str, torch.device] = "cpu",
    print_every: int = 1,
    print_name: Union[str, None] = None,  # for logging purposes, e.g. "Denoiser"
):
    """
    Trains `denoiser` to predict p(x₀ | x_t, σ_t).

    Loss = cross‑entropy between model logits and the *clean* token x₀.
    """

    assert diffusion in {"absorbing_state", "uniform"}
    if diffusion == "absorbing_state":
        assert mask_idx is not None, "Need a dedicated <mask> token id"

    denoiser = denoiser.to(device)
    denoiser.train()
    opt = torch.optim.AdamW(denoiser.parameters(), lr=lr)

    ds     = TensorDataset(train_seqs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    if val_seqs is not None:
        val_loader = DataLoader(TensorDataset(val_seqs),
                                batch_size=batch_size)

    ce = nn.CrossEntropyLoss(
        ignore_index=pad_val if pad_val is not None else -100)

    print_label = f"[Denoiser]" if print_name is None else f"[Denoiser {print_name}]"
    for ep in range(1, epochs+1):
        running = 0.0
        for (x0,) in loader:
            x0 = x0.to(device)

            # 1) sample continuous timestep t ∼ U(0,1)
            t = torch.rand(x0.size(0), device=device)

            # 2) get sigma(t) for conditioning
            sigma_t, sigma_prime = noise_sched(t)

            # 3) corrupt x₀ → x_t
            x_t, move_mask_t = corrupt(x0=x0, sigma=sigma_t,
                                       diffusion=diffusion,
                                       vocab_size=vocab_size,
                                       mask_idx=mask_idx)

            # 4.1) Loss with weight and masking
            logits = denoiser(x_t, sigma_t)
            logits[..., mask_idx] = -1e9  # zero-masking prob
            nll = F.cross_entropy(logits.transpose(1, 2),x0,reduction='none' ) # keep every token
            nll = nll * move_mask_t         # (B,L)  # mask out non-moved tokens
            weight = sigma_prime.unsqueeze(1)  # σ′(t)  ==  α′/(1-α)
            loss1 = (nll * weight).sum() / move_mask_t.sum()

            # 4.2) forward + loss
            logits = denoiser(x_t, sigma_t)            # (B,L,V)
            loss2   = ce(logits.view(-1, vocab_size), x0.view(-1))

            opt.zero_grad()
            loss2.backward()
            opt.step()
            running += loss2.item() * x0.size(0)

        if ep % print_every == 0:
            print(f"{print_label} epoch {ep}/{epochs} "
                  f"trainCE = {running / len(ds):.4f}", end="")
            wandb.log({"Denoiser/train_loss": running / len(ds), "epoch": ep})

            # quick val metrics if provided
            if val_seqs is not None:
                denoiser.eval()
                with torch.no_grad():
                    val_loss, correct_tok, correct_seq, total_tok, total_seq = 0, 0, 0, 0, 0
                    for (x0,) in val_loader:
                        x0 = x0.to(device)  # (B,L)
                        B, L = x0.shape
                        t = torch.rand(B, device=device)
                        sigma_t, _ = noise_sched(t)
                        x_t, move_mask_t = corrupt(x0=x0, sigma=sigma_t,
                                       diffusion=diffusion,
                                       vocab_size=vocab_size,
                                       mask_idx=mask_idx)  # noisy input

                        logits = denoiser(x_t, sigma_t)  # (B,L,V)
                        val_loss += ce(logits.view(-1, vocab_size),
                                       x0.view(-1)).item() * B * L

                        # ------------------------------
                        # accuracy metrics
                        # ------------------------------
                        pred = logits.argmax(dim=-1)  # (B,L)
                        match = pred.eq(x0)  # bool mask

                        correct_tok += match[move_mask_t].sum().item()
                        total_tok += move_mask_t.sum().item()

                        correct_seq += match.all(dim=1).sum().item()
                        total_seq += B

                print(f"    • val CE = {val_loss / total_tok:.4f}"
                      f"    • token acc = {100 * correct_tok / total_tok:.2f}%"
                      f"    • seq acc = {100 * correct_seq / total_seq:.2f}%")
            else:
                print()
    return denoiser



# =====================================================================
#  5. Train the ratio-estimator rφ without time-conditioning on clean data (just for experimentation)
# =====================================================================
def train_ratio_estimator_on_clean_data(
    model: nn.Module,
    domain_classifier: nn.Module,
    source_data: torch.Tensor,
    target_data: torch.Tensor,
    epochs: int      = 10,
    batch_size: int  = 256,
    lr: float        = 1e-4,
    classifier_output_with_sigmoid: bool  = True,          # train on log-ratio instead of ratio
    device: str      = "cpu",
    lambda_lap : float = 0,   # Laplacian penalty
):
    """
    Parameters
    ----------
    model              : RatioNet          (outputs rφ(x) or log rφ(x))
    domain_classifier  : already-trained dω(x) ∈ (0,1) (1 = source)
    source_data        : [N_src, L] long
    target_data        : [N_tgt, L] long
    """
    # freeze the classifier
    domain_classifier.eval().to(device)
    for p in domain_classifier.parameters():
        p.requires_grad_(False)

    # put the ratio network into training mode
    model.train().to(device)
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-4,lr=lr)
    mse_loss  = nn.MSELoss()

    # unlabeled mixture dataset
    seqs = torch.cat([source_data, target_data])      # [N_src+N_tgt, L]
    loader = DataLoader(seqs, batch_size=batch_size,
                        shuffle=True, drop_last=True)
    eps = 1e-8
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch_seqs in loader:                     # batch_seqs: [B,L]
            batch_seqs = batch_seqs.to(device)

            # ----------------------------------------------------------
            # 1.  Compute pseudo-targets with the fixed classifier
            with torch.no_grad():
                c_out = domain_classifier(batch_seqs).squeeze(-1)  # [B]
                if classifier_output_with_sigmoid:
                    ratio_target = (1.0 - c_out) / (c_out + eps)  # avoid /0
                    ratio_target = torch.log(ratio_target + eps)  # [B]
                else:
                    ratio_target = -c_out                            # [B]

            # ----------------------------------------------------------
            # 2.  Forward pass through ratio network
            ratio_pred = model(batch_seqs)            # [B]
            loss = mse_loss(ratio_pred, ratio_target)

            # ----------------------------------------------------------
            # 3. Laplacian penalty
            lap = laplacian_penalty(ratio_pred, batch_seqs, radius=2)
            loss = loss + lambda_lap * lap

            # ----------------------------------------------------------
            # 4.  Optimise
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_seqs.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        print(f"[Ratio] epoch {epoch:2d}/{epochs}, loss = {epoch_loss:.6f}")
        wandb.log({"Ratio/train_loss": epoch_loss, "epoch": epoch})

    # move back to CPU for convenience
    model.to("cpu")
    return model


# ==================================================================================
# 2. Train time‑dependent ratio estimator  r_ψ(x_t , t)
# ==================================================================================
def train_ratio_estimator(
    model: nn.Module,
    domain_classifier: nn.Module,
    source_data: torch.Tensor,
    target_data: torch.Tensor,
    noise_sched: nn.Module,
    *,
    diffusion: str = "absorbing_state",
    mask_idx: Optional[int] = None,
    vocab_size: Optional[int] = None,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-4,
    classifier_output_with_sigmoid: bool = False,
    device: Union[str, torch.device] = "cpu",
    lambda_lap: float = 0.0,
    only_source_domain: bool = False,                 #  training on source domain only - like in the TLDM paper
):
    """Train a *time‑conditioned* density‑ratio network on noisy sequences.

    The network learns  r_ψ(x_t , t) ≈ q(x_0) / p(x_0)  from pairs (x_t , t)
    produced by the discrete forward process.  Pseudo‑targets come from a
    fixed binary domain classifier d_ω(x_0).
    """

    # ───────────────────── 1.  prepare models ────────────────────────────    domain_classifier.eval().to(device)
    for p in domain_classifier.parameters():
        p.requires_grad_(False)

    model.train().to(device)
    domain_classifier.eval().to(device)  # freeze the classifier
    opt = torch.optim.Adam(model.parameters(), weight_decay=1e-4, lr=lr)
    mse = nn.MSELoss()

    # Combined unlabeled pool (x_0 ,) – only x_0 used here.
    seqs = source_data if only_source_domain else torch.cat([source_data, target_data])         # (N, L)
    loader = DataLoader(seqs, batch_size=batch_size, shuffle=True, drop_last=True)

    for ep in range(1, epochs + 1):
        running = 0.0
        for x0 in loader:                                   # x0 : (B, L)
            x0 = x0.to(device)
            B = x0.size(0)

            # ---------------- 2. sample timestep & corrupt ----------------
            t = torch.rand(B, device=device)                # U(0,1)
            sigma_t, _ = noise_sched(t)                     # σ(t)
            x_t, _ = corrupt(
                x0,
                sigma_t,
                diffusion=diffusion,
                vocab_size=vocab_size,
                mask_idx=mask_idx,
            )
            #aug_seqs = augment_hamming1(seqs=x_t, mask_idx=mask_idx, vocab_size=vocab_size)  # (B, L)

            # ---------------- 3. build pseudo targets --------------------
            with torch.no_grad():
                c_out = domain_classifier(x0).squeeze(-1)          # (B,)
                if classifier_output_with_sigmoid:
                    ratio_target = (1.0 - c_out) / (c_out + 1e-8)
                    ratio_target = torch.log(ratio_target + 1e-8)
                else:
                    ratio_target = -c_out                            # (B,)

            # ---------------- 4. forward + loss --------------------------
            ratio_pred = model(x_t, t)                              # (B,)
            loss = mse(ratio_pred, ratio_target)

            # optional Laplacian regulariser for smoothness
            if lambda_lap > 0:
                lap = laplacian_penalty(ratio_pred, x_t, radius=2)
                loss = loss + lambda_lap * lap

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item() * B

        print_statement = "[Ratio‑TD]" if not only_source_domain else "[Ratio‑TD-only-src]"
        print(f"{print_statement} epoch {ep}/{epochs}, loss = {running / len(loader.dataset):.6f}")
        wandb.log({f"{print_statement}/train_loss": running / len(loader.dataset), "epoch": ep})

    model.to("cpu")
    return model




# =====================================================================
# 8. Train the ratio with the Regularization from the TLDM paper (Pseudo-Code 4 in the Appendix)
# =====================================================================
def train_ratio_network_with_regularization_like_tldm_paper(
    model: nn.Module,
    domain_classifier: nn.Module,
    domain_classifier_t: nn.Module,
    denoiser_model: nn.Module,
    source_data: torch.Tensor,
    target_data: torch.Tensor,
    noise_sched: nn.Module,
    *,
    diffusion: str = "absorbing_state",
    mask_idx: Optional[int] = None,
    vocab_size: Optional[int] = None,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-4,
    classifier_output_with_sigmoid: bool = False,  # if True, use sigmoid output
    eta1and2: Tuple[float, float] = (0.1, 0.1),
    device: Union[str, torch.device] = "cpu",
):
    """
    Train a time‐conditioned ratio network r_ψ(x_t, t) with:
      L_ratio + η1 · L_cycle + η2 · L_consistency
    as in Appendix Pseudo-Code 4.
    """
    # 1) Prepare
    device = torch.device(device)
    η1, η2 = eta1and2
    use_cycle = η1 > 0.0
    use_consistency = η2 > 0.0

    # Freeze pre-trained nets
    for net in (domain_classifier, domain_classifier_t, denoiser_model):
        net.eval().to(device)
    model.train().to(device)

    # 2) DataLoaders for source & target x₀
    src_loader = DataLoader(
        TensorDataset(source_data),
        batch_size=min(batch_size, len(source_data)),
        shuffle=True, drop_last=True
    )
    tgt_loader = DataLoader(
        TensorDataset(target_data),
        batch_size=min(batch_size, len(target_data)),
        shuffle=True, drop_last=True
    )
    # cycle target if needed
    if len(src_loader) > len(tgt_loader):
        from itertools import cycle
        tgt_loader = cycle(tgt_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # 3) Train
    for epoch in range(1, epochs + 1):
        for (x0_src,), (x0_tgt,) in zip(src_loader, tgt_loader):
            x0_src = x0_src.to(device)
            x0_tgt = x0_tgt.to(device)

            # —– sample times for src and tgt
            B_src = x0_src.size(0)
            t_src = torch.rand(B_src, device=device)  # U(0,1)
            sigma_src, _ = noise_sched(t_src)
            x_t_src, _ = corrupt(x0_src, sigma_src,
                                 diffusion=diffusion,
                                 mask_idx=mask_idx,
                                 vocab_size=vocab_size)

            # —– static‐classifier pseudo‐target
            with torch.no_grad():
                c_src = domain_classifier(x0_src).squeeze(-1)
                if classifier_output_with_sigmoid:
                    r_src = (1. - c_src) / (c_src + 1e-8)
                    r_src = torch.log(r_src + 1e-8)
                else:
                    r_src = -c_src

            # —– compute L_ratio
            r_pred_src = model(x_t_src, t_src)
            loss_ratio = mse(r_pred_src, r_src)

            # —– cycle loss on target
            if use_cycle:
                B_tgt = x0_tgt.size(0)
                t_tgt = torch.rand(B_tgt, device=device)  # U(0,1)
                sigma_tgt, _ = noise_sched(t_tgt)
                x_t_tgt, _ = corrupt(x0_tgt, sigma_tgt,
                                     diffusion=diffusion,
                                     mask_idx=mask_idx,
                                     vocab_size=vocab_size)

                with torch.no_grad():
                    c_tdep = domain_classifier_t(
                       x_t_tgt, t_tgt).squeeze(-1)
                    if classifier_output_with_sigmoid:
                        r_tdep = (1. - c_tdep) / (c_tdep + 1e-8)
                        r_tdep = torch.log(r_tdep + 1e-8)
                    else:
                        r_tdep = -c_tdep

                r_pred_tgt = model(x_t_tgt, t_tgt)
                loss_cycle = mse(r_pred_tgt, r_tdep)
            else:
                loss_cycle = torch.tensor(0., device=device)

            # —– consistency regulariser
            if False:
                # gradient of log r_ψ
                x_t_tgt.requires_grad_(True)
                log_r = torch.log(model(x_t_tgt, t_tgt) + 1e-20).sum()
                grad_log_r = torch.autograd.grad(log_r, x_t_tgt)[0]
                x_t_tgt.requires_grad_(False)

                # source and target score estimates
                with torch.no_grad():
                    s_src = denoiser_model(x_t_src, t_src)
                    score_src = - s_src / sigma_src.unsqueeze(1)
                    s_tgt = denoiser_model(x_t_tgt, t_tgt)
                    score_tgt = - s_tgt / sigma_tgt.unsqueeze(1)

                # target consistency: ∇log(q_t/p_t) = score_tgt – score_src
                grad_target = score_tgt - score_src[:score_tgt.size(0)]
                loss_consistency = mse(grad_log_r, grad_target)
            else:
                loss_consistency = torch.tensor(0., device=device)

            # —– total
            total = loss_ratio + η1 * loss_cycle + η2 * loss_consistency
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        print(f"[Ratio-Reg] epoch {epoch}/{epochs}  "
              f"L_ratio={loss_ratio.item():.4f}  "
              f"L_cycle={loss_cycle.item():.4f}  "
              f"L_consistency={loss_consistency.item():.4f}")
        wandb.log({
            "Ratio-Reg/train_loss": loss_ratio.item(),
            "Ratio-Reg/L_cycle": loss_cycle.item(),
            "Ratio-Reg/L_consistency": loss_consistency.item(),
            "epoch": epoch,
        })

    model.to("cpu")
    return model







@torch.no_grad()
def build_pseudo_ratio_vector(
    network: nn.Module,     # frozen dω(xₜ , t)
    xt:  torch.Tensor,                  # (B, L)   noisy sequences
    pos: torch.Tensor,                  # (B,)     masked positions
    t:   torch.Tensor,                  # (B,)     timesteps
    network_type: str,                  # "classifier" or "ratio_estimator"
    *,
    vocab_size: int,
    batch_size: int = 4_096,            # mini-batch along (B·V) axis
    classifier_output_with_sigmoid: bool = False,
) -> torch.Tensor:                      # -> (B, V)
    """
    For every example i and every token v ∈ {0..V-1}, form
       xₜ(i,ℓ←v)   →   dω(xₜ(i,ℓ←v), t_i)
    and convert that probability into a (log) density-ratio target.

    The work is streamed in chunks of `batch_size` candidates to avoid
    blowing up GPU RAM.  Complexity: O(B·V) forward passes.
    """
    eps = 1e-8
    device = xt.device
    B, L = xt.shape
    N = B * vocab_size                      # total # of candidate sequences

    # flat indices of size N
    seq_id  = torch.arange(B, device=device).repeat_interleave(vocab_size)   # (N,)
    tok_id  = torch.arange(vocab_size, device=device).repeat(B)              # (N,)
    pos_id  = pos.repeat_interleave(vocab_size)                              # (N,)
    t_flat  = t.repeat_interleave(vocab_size)                                # (N,)

    out = torch.empty(N, device=device)          # holds log-ratios or ratios

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        # slice the candidate indices for this micro-batch
        sid  = seq_id[start:end]
        pid  = pos_id[start:end]
        tok  = tok_id[start:end]
        t_mb = t_flat[start:end]

        # clone and mutate only the targeted slot
        seq_mb = xt[sid].clone()
        seq_mb[torch.arange(seq_mb.size(0), device=device), pid] = tok

        if network_type == "classifier":
            # forward through frozen classifier
            p_src = network(seq_mb, t_mb).squeeze(-1)      # (m,)
            if classifier_output_with_sigmoid:
                ratio = (1.0 - p_src) / (p_src + eps)
                ratio = torch.log(ratio + eps)
            else:
                ratio = -p_src                               # (m,)
        elif network_type == "ratio_estimator":
            # forward through frozen ratio estimator
            ratio = network(seq_mb, t_mb)                     # (m,)
        else:
            raise ValueError(f"Unknown network type: {network_type}")
        out[start:end] = ratio

    return out.view(B, vocab_size)          # (B, V)

def vector_ratio_training(
    model: nn.Module,
    domain_classifier_t: nn.Module,
    pretrained_ratio_net: nn.Module,          # r_φ(xₜ, t)  – frozen
    source_data: torch.Tensor,
    target_data: torch.Tensor,
    noise_sched: nn.Module,
    *,
    diffusion: str = "absorbing_state",
    mask_idx: Optional[int] = None,
    vocab_size: int,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-4,
    classifier_output_with_sigmoid: bool = False,
    device: Union[str, torch.device] = "cuda",
    lambda_lap: float = 0.0,
    eta_clf_and_ratio: Tuple[float, float] = (1, 0),  # weights fo r classifier estimation and pre-trained ratio net
    pseudo_mb: int = 4_096,          # micro-batch size for pseudo targets
    lambda_hinge: float = 0.0,          # optional mask penalty
):
    """
    Train rψ(xₜ, ℓ, t) → (B, V) with *full-vector* supervision produced
    by the frozen time-dependent classifier.  Every token v at the masked
    slot contributes to the loss.
    """
    eps = 1e-8
    delta_base = 2.0
    device = torch.device(device)

    # ── freeze helper net ─────────────────────────────────────────────
    domain_classifier_t.eval().to(device)
    for p in domain_classifier_t.parameters():
        p.requires_grad_(False)
    pretrained_ratio_net.eval().to(device)

    for p in pretrained_ratio_net.parameters():
        p.requires_grad_(False)

    # ── prepare ratio net & optimiser ────────────────────────────────
    model.train().to(device)
    opt  = torch.optim.Adam(model.parameters(), weight_decay=1e-4, lr=lr)
    mse  = nn.MSELoss()

    # unlabeled mixture
    seqs = torch.cat([source_data, target_data])     # (N, L)
    loader = DataLoader(seqs, batch_size=batch_size,
                        shuffle=True, drop_last=True)
    L = seqs.size(1)

    eta_clf, eta_ratio = eta_clf_and_ratio
    loss_clf, loss_ratio = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    # ── main loop ────────────────────────────────────────────────────
    for ep in range(1, epochs + 1):
        running = 0.0

        for x0 in loader:                              # (B, L)
            x0 = x0.to(device)
            B  = x0.size(0)

            # pick one target position per sequence
            pos = torch.randint(0, L, (B,), device=device)

            # mask that slot before corruption
            x0_mask = x0.clone()
            x0_mask[torch.arange(B, device=device), pos] = mask_idx

            # forward-diffuse
            t   = torch.rand(B, device=device)
            σ_t, _ = noise_sched(t)
            x_t, _ = corrupt(x0_mask, σ_t,
                             diffusion=diffusion,
                             mask_idx=mask_idx,
                             vocab_size=vocab_size)
            # model predictions
            pred_vec = model(x_t, pos, t)  # (B, V)

            # pseudo-target vector (B, V)
            if eta_clf > 0.0:
                ratio_vec = build_pseudo_ratio_vector(
                    domain_classifier_t,
                    network_type="classifier",
                    xt=x_t,
                    pos=pos,
                    t=t,
                    vocab_size=vocab_size,
                    batch_size=pseudo_mb*8,
                    classifier_output_with_sigmoid=classifier_output_with_sigmoid,
                ).detach()                                  # no grad!
                loss_clf = mse(pred_vec, ratio_vec)

            if eta_ratio > 0.0:
                ratio_vec = build_pseudo_ratio_vector(
                    pretrained_ratio_net,
                    network_type="ratio_estimator",
                    xt=x_t,
                    pos=pos,
                    t=t,
                    vocab_size=vocab_size,
                    batch_size=pseudo_mb*8,
                    classifier_output_with_sigmoid=classifier_output_with_sigmoid,
                ).detach()                                  # no grad!
                loss_ratio = mse(pred_vec, ratio_vec)

            # combine losses
            loss = eta_clf * loss_clf + eta_ratio * loss_ratio

            # optional Laplacian penalty on the *scalar* ratios of gt token
            if lambda_lap > 0.0:
                gt_tok = x0[torch.arange(B, device=device), pos]
                lap_in = pred_vec[torch.arange(B), gt_tok]    # (B,)
                lap = laplacian_penalty(lap_in, x_t, radius=2)
                loss = loss + lambda_lap * lap
            if lambda_hinge > 0.0:
                # mask penalty on the *scalar* ratios of masked token
                # scalar mask scores and mean real-token score
                mask_val = pred_vec[..., mask_idx]  # (B,)
                mean_real = pred_vec[..., :-1].mean(-1)  # (B,)
                # time-dependent margin     δ = δ₀·(1 − t)
                delta_t = delta_base * (1.0 - t)  # (B,)
                # hinge penalty: 0 if (mask ≤ mean_real − δ)
                penalty = F.relu(mask_val - mean_real + delta_t)  # (B,)
                mask_reg = penalty.mean()

                loss = loss + lambda_hinge * mask_reg

            wandb.log({
                "Vec-Ratio/loss": loss.item(),
                "Vec-Ratio/loss_clf": loss_clf.item(),
                "Vec-Ratio/loss_ratio": loss_ratio.item(),
                "Vec-Ratio/mask_regularizer": mask_reg.item() if lambda_hinge > 0.0 else 0.0,
            })
            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item() * B

        avg = running / len(loader.dataset)
        print(f"[Vec-Ratio clf:{eta_clf} and ratio:{eta_ratio}] epoch {ep}/{epochs}, MSE = {avg:.6f}")
        wandb.log({f"Vec-Ratio/clf:{eta_clf} and ratio:{eta_ratio}/train_loss": avg,"epoch": ep,})

    model.to("cpu")
    return model




# ==================================================================================
# 2-bis.  Train time-dependent ratio estimator  r_ψ(x_t , t)
#        **with mixture sampling & importance weights**
# ==================================================================================
def train_ratio_estimator_on_noisy_data_is(
    model: nn.Module,
    domain_classifier: nn.Module,
    source_data: torch.Tensor,
    target_data: torch.Tensor,
    noise_sched: nn.Module,
    *,
    diffusion: str = "absorbing_state",
    mask_idx: Optional[int] = None,
    vocab_size: Optional[int] = None,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-4,
    classifier_output_with_sigmoid: bool = False,
    alpha: Union[float, None] = None,                 # P(x₀ comes from *source*) in the mixture
    device: Union[str, torch.device] = "cpu",
    lambda_lap: float = 0.0,
):
    r"""Importance-resampling version of the ratio-net trainer.

    Each mini-batch is drawn from the proposal
        m(x₀)=α·p(x₀) + (1-α)·q(x₀).
    Samples get the weight
        ŵ(x₀)= 1 / (α + (1-α) · r̂(x₀)),
    where r̂(x₀) ≈ q(x₀)/p(x₀) comes from the frozen domain classifier.

    The *unweighted* objective remains
        𝔼ₓ₀∼p 𝔼_{z_t|x₀}[ (h_ψ(z_t,t) − r^\*)² ].
    """
    if alpha is None:
        alpha = len(source_data) / (len(source_data) + len(target_data))
        alpha = min(0.9, max(0.1, alpha))  # avoid degenerate cases
    assert 0.0 < alpha < 1.0, "`alpha` must be in (0,1)"
    print(f"[Ratio-TD-IS] using α = {alpha:.2f} for importance sampling")
    source_data = source_data.to(device)
    target_data = target_data.to(device)

    device = torch.device(device)
    eps = 1e-8

    # ───────────────────── 1.  freeze helper nets ──────────────────────
    domain_classifier.eval().to(device)
    for p in domain_classifier.parameters():
        p.requires_grad_(False)

    model.train().to(device)
    optim = torch.optim.Adam(model.parameters(), weight_decay=1e-4, lr=lr)

    N_src, N_tgt = source_data.size(0), target_data.size(0)

    # handy lambdas to sample from tensors on the fly (avoids DataLoader boilerplate)
    def sample_rows(tensor: torch.Tensor, n: int) -> torch.Tensor:
        idx = torch.randint(0, tensor.size(0), (n,), device=device)
        return tensor[idx]

    for ep in range(1, epochs + 1):
        # pick enough iterations so that ≈ every point is seen once/epoch
        iters = math.ceil((N_src + N_tgt) / batch_size)
        running = 0.0

        for _ in range(iters):
            # ---------------- 2. draw x₀ from the mixture m --------------
            n_src = int(round(alpha * batch_size))
            n_tgt = batch_size - n_src
            x0_src = sample_rows(source_data, n_src)
            x0_tgt = sample_rows(target_data, n_tgt)
            x0     = torch.cat([x0_src, x0_tgt], dim=0)    # (B,L)

            # permute so src/tgt are mixed (nice for batch-norm etc.)
            perm = torch.randperm(batch_size, device=device)
            x0   = x0[perm]

            B = x0.size(0)

            # ---------------- 3. corrupt forward -------------------------
            t = torch.rand(B, device=device)                # U(0,1)
            sigma_t, _ = noise_sched(t)
            x_t, _ = corrupt(
                x0, sigma_t,
                diffusion=diffusion,
                vocab_size=vocab_size,
                mask_idx=mask_idx,
            )

            # ---------------- 4. ratio pseudo-target  r̂(x₀) -------------
            with torch.no_grad():
                c_out = domain_classifier(x0).squeeze(-1)      # P(source|x₀)
                if classifier_output_with_sigmoid:
                    r_hat = (1.0 - c_out) / (c_out + eps)  # ≈ q/p
                    ratio_target = torch.log(r_hat + eps)
                else:
                    r_hat = -c_out                            # ≈ q/p
                    ratio_target = r_hat                      # no log here

                # importance weight  ŵ(x₀) = 1 / (α + (1-α)·r̂)
                w_hat = 1.0 / (alpha + (1.0 - alpha) * r_hat)
                w_hat = w_hat.detach()                         # no grad!

            # ---------------- 5. forward + weighted loss -----------------
            ratio_pred = model(x_t, t)                         # (B,)

            sq_err = (ratio_pred - ratio_target) ** 2          # (B,)
            loss = (w_hat * sq_err).mean()                     # importance-weighted MSE

            # optional Laplacian penalty for smoothness
            if lambda_lap > 0.0:
                lap = laplacian_penalty(ratio_pred, x_t, radius=2)
                loss = loss + lambda_lap * lap

            optim.zero_grad()
            loss.backward()
            optim.step()

            running += loss.item() * B

        print(f"[Ratio-TD-IS] epoch {ep}/{epochs}, "
              f"loss = {running / (iters * batch_size):.6f}")
        wandb.log({"Ratio-TD-IS/train_loss": running / (iters * batch_size), "epoch": ep})

    model.to("cpu")
    return model


# =====================================================================
# 3.  Train ratio-estimator rψ(x_t , t)  **using only the time-dependent
#     domain-classifier**  dω(x_t , t)  and data from BOTH domains
# =====================================================================
def ratio_trained_on_time_dependent_classifier(
    model: nn.Module,
    domain_classifier_t: nn.Module,          # d_ω(x_t , t)  – frozen
    domain_classifier: nn.Module,            # d_ω(x_0)      – frozen
    denoiser_model: nn.Module,               # x_θ(x_t , t)  – frozen
    source_data: torch.Tensor,               # (N_src , L)
    target_data: torch.Tensor,               # (N_tgt , L)
    noise_sched: nn.Module,                  # callable σ(t)
    *,
    diffusion: str = "absorbing_state",
    mask_idx: Optional[int] = None,
    vocab_size: Optional[int] = None,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-4,
    classifier_output_with_sigmoid: bool = False,
    device: Union[str, torch.device] = "cpu",
    lambda_lap: float = 0.0,
    lambda_reconstruction: float = 0.0,               # ★ NEW: weight for reconstruction loss
    lambda_clean_ratio: float = 0.0,               # weight for the clean classifier ratio
):
    r"""
    Learn r_ψ so that r_ψ(x_t , t) ≈ q(x₀)/p(x₀) using
      • guidance targets from the time-dependent classifier, and
      • ★ NEW: a reconstruction-consistency target obtained by
        denoising x_t → \hat x_0 and re-scoring with a *clean* classifier.
    ----------------------------------------------------------------------
    reconstruction loss (per sample):
        L_reconstruction = ‖r_ψ(x_t , t)   -   r̂_reconstruction‖²
        where r̂_reconstruction = (1 - d_ω( \hat x_0 )) / d_ω( \hat x_0 )
    ----------------------------------------------------------------------
    Set λ_reconstruction = 0 to disable the new term.
    """
    eps = 1e-8
    device = torch.device(device)

    # ─────────────────────────────────────────  freeze helper nets
    for net in (domain_classifier_t, domain_classifier, denoiser_model):
        net.eval().to(device)
        for p in net.parameters():
            p.requires_grad_(False)

    # ─────────────────────────────────────────  ratio net + optimiser
    model.train().to(device)
    optim = torch.optim.Adam(model.parameters(), weight_decay=1e-4, lr=lr)
    mse   = nn.MSELoss()

    seqs   = torch.cat([source_data, target_data])
    loader = DataLoader(seqs, batch_size=batch_size,
                        shuffle=True, drop_last=True)
    print_statement = "[Ratio on t-clf"
    if lambda_reconstruction > 0.0:
        print_statement += " + reconstruction"
    if lambda_clean_ratio > 0.0:
        print_statement += " + clean ratio"
    print_statement += "]"
    loss_clean_ratio, loss_reconstruction = torch.tensor(0.0), torch.tensor(0.0)
    # ─────────────────────────────────────────  main loop
    for ep in range(1, epochs + 1):
        epoch_loss = 0.0

        for x0 in loader:                           # (B, L)
            x0 = x0.to(device)
            B  = x0.size(0)

            # 1) sample time and corrupt
            t = torch.rand(B, device=device)        # U(0,1)
            σ_t, _ = noise_sched(t)
            x_t, _ = corrupt(x0, σ_t,
                             diffusion=diffusion,
                             mask_idx=mask_idx,
                             vocab_size=vocab_size)

            # 2) guidance pseudo-target from d_ω(x_t , t)
            with torch.no_grad():
                p_src_t = domain_classifier_t(x_t, t).squeeze(-1)
                if classifier_output_with_sigmoid:
                    guide_ratio = (1.0 - p_src_t) / (p_src_t + eps)
                    guide_ratio = torch.log(guide_ratio + eps)
                else:
                    guide_ratio = -p_src_t                      # (B,)

            # 3) ★ NEW: reconstruction pseudo-target via denoise → d_ω(x₀)
            if lambda_reconstruction > 0.0:
                with torch.no_grad():
                    # denoiser may output logits or tokens – handle both
                    x_hat0 = denoiser_model(x_t, t)          # (B, L, |V|) or (B, L)
                    if x_hat0.dtype == torch.float:          # logits → hard tokens
                        x_hat0 = x_hat0.argmax(-1)           # (B, L)
                    p_src_hat = domain_classifier(x_hat0).squeeze(-1)
                    if classifier_output_with_sigmoid:
                        reconstruction_ratio = (1.0 - p_src_hat) / (p_src_hat + eps)
                        reconstruction_ratio = torch.log(reconstruction_ratio + eps)
                    else:
                        reconstruction_ratio = -p_src_hat

            if lambda_clean_ratio > 0.0:
                # compute the ratio from the clean classifier
                with torch.no_grad():
                    c_out = domain_classifier(x0).squeeze(-1)
                    if classifier_output_with_sigmoid:
                        clean_domain_ratio = (1.0 - c_out) / (c_out + eps)
                        clean_domain_ratio = torch.log(clean_domain_ratio + eps)
                    else:
                        clean_domain_ratio = -c_out

            # 4) forward & losses
            ratio_pred = model(x_t, t).view_as(guide_ratio)   # (B,)
            loss = mse(ratio_pred, guide_ratio)               # guidance term
            if lambda_reconstruction > 0.0:
                loss_reconstruction = mse(ratio_pred, reconstruction_ratio)
                loss = loss + lambda_reconstruction * loss_reconstruction
            if lambda_clean_ratio > 0.0:
                loss_clean_ratio = mse(ratio_pred, clean_domain_ratio)
                loss = loss + lambda_clean_ratio * loss_clean_ratio

            # optional Laplacian smoothness
            if lambda_lap > 0.0:
                lap = laplacian_penalty(ratio_pred, x_t, radius=2)
                loss = loss + lambda_lap * lap

            # 5) optimise
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * B

        print(f"{print_statement} epoch {ep:2d}/{epochs}, "
              f"MSE = {epoch_loss/len(loader.dataset):.6f}")
        wandb.log({f"{print_statement}/train_loss": epoch_loss/len(loader.dataset), "reconstruction_loss": loss_reconstruction.item(),
                     "clean_ratio_loss": loss_clean_ratio.item(),
                     "epoch": ep})

    model.to("cpu")
    return model