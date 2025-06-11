#!/usr/bin/env python3
"""
Main entry point: data → training → sampling.
Keeps every original helper function name from the notebook/codebase.
"""

import time
import torch, random, numpy as np
from tqdm.auto import tqdm

import time, types
import torch, random, numpy as np
from tqdm.auto import tqdm
import os
from datetime import datetime
import matplotlib.pyplot as plt
import wandb

# ── editable configuration object ──────────────────────────────────────
config = types.SimpleNamespace(
    # experiment switches
    data_type="gaussian",                      # "markov" | "gaussian" | "discrete" | "point_forms"
    unguided_sampling=False,
    skip_ratio_estimator_on_clean_data=True,
    test_run=False,                  # if True, use smaller data and fewer epochs

    # reproducibility
    seed=100,

    # data settings
    len_seqs=20,
    vocab_size=5,
    number_of_source_samples=3000,
    number_of_target_samples=3000,

    # data_type="gaussian",                         ´
    dimension_gaussian=2,
    num_mixtures=2,
    grid_size=1.5,
    quantization_step=0.01,

    # data_type="point_forms",
    source_form="moons",       # "swissroll" | "2spirals" | "circles" | "moons" | "pinwheel" | "checkerboard" | "8gaussians"
    target_form="2spirals",      # "swissroll" | "2spirals" | "circles" | "moons" | "pinwheel" | "checkerboard" | "8gaussians"
    #grid_size=6,               # grid size for point forms will be overwritten to 6
    #quantization_step=0.04,    # quantization step for point forms will be overwritten to 0.04

    # data_type="discrete",
    source_dist="gaussian", # "uniform" | "gaussian" | "x2"
    target_dist="x2", # "uniform" | "gaussian" | "x2"
    distance=0, # distance between source and target distributions
    grid_size_discrete=10,

    # data_type="markov",
    type_of_T_matrix="diagonal",  # "random" | "diagonal"
    extra_target_tokens=0, # if True, add extra token to target sequences
    dirichlet_alpha=4.0,


    # training
    batch_size=256,
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
    train_classifier_epochs=40,
    train_ratio_estimator_epochs=30,
    train_denoiser_epochs=40,

    # denoiser model dims
    denoiser_dim=256,
    denoiser_heads=4,
    denoiser_layers=4,

    # sampling
    sample_batch_size=512*2,
    sample_gammas=[0,0.5,1,2], # [0.0, .1,.2,.3,.4,.5,0.6,.7,.8,.9, 1.0, 2.0],
    sample_size= 2048,
    denoising_steps=20,
    use_plg=True,
    eta=1.0,   # eta for PLG sampling, only used if `use_plg=True`: log_ratio - self.cfg.eta * torch.log(1 + torch.exp(log_ratio))
    stochastic_sampling_jitter_mask = 0.0,
    stochastic_sampling_eps_noise= 0.1,
)

# Initialize W&B run

wandb.init(
    project="Try out Ratio Network implementation",
    config=vars(config),
)
wandb.config.update(vars(config), allow_val_change=True)


# ── local modules (unchanged function names) ────────────────────────────
from data import (
    build_markov_transfer,      # Markov‑shift
    orth_means_random,          # Gaussian mixture helpers
    build_iid_gmm_data,
    estimate_transition_matrix,
    create_probability_maps,
    discrete_gray_codes_forms,
    sample_from_prob_map,
)
from utils import transition_stats, seed_everything, print_trace
from plot_function import (
    plot_iid_gmm_points,
    plot_ratio_estimator,
    plot_prob_map,
    plot_sample_counts,
)
from models import AdaLNDenoiser, ClassiferNet, RatioNetAdaLN, RatioNetAdaLNVector
from diffusion import LogLinearNoise, DiffusionConfig, Diffusion
from training import *

# =======================================================================
#  Function to load data
# =======================================================================
def load_data():
    """Return (src, tgt, extras) depending on ``config.data_type``."""

    # ----------------------- markov -----------------------
    if config.data_type == "markov":
        src, tgt, P, Q, voc_sz = build_markov_transfer(
            L=config.len_seqs,
            vocab_size=config.vocab_size,
            n_src=config.number_of_source_samples,
            n_tgt=config.number_of_target_samples,
            diag_src = 0.7,
            diag_tgt = 0.4,
            dirichlet_alpha_src=config.dirichlet_alpha,
            dirichlet_alpha_tgt=config.dirichlet_alpha,
            type_of_T_matrix=config.type_of_T_matrix,
            extra_target_tokens=config.extra_target_tokens,
        )
        config.vocab_size = voc_sz
        return src, tgt, {"S": P, "T": Q}

    # ----------------------- gaussian -----------------------
    if config.data_type == "gaussian":
        mu_s, mu_t = orth_means_random(
            dimension=config.dimension_gaussian,)
        if config.num_mixtures == 1 and config.dimension_gaussian == 2:
            mu_s = torch.tensor([config.grid_size-1, config.grid_size-1])
            mu_t = torch.tensor([-(config.grid_size-1), -(config.grid_size-1)])

        src, tgt, vocab_sz, edges = build_iid_gmm_data(
            mu_src=mu_s,
            mu_tgt=mu_t,
            device=config.device,
            n_src=config.number_of_source_samples, n_tgt=config.number_of_target_samples,
            grid_min = -config.grid_size, grid_max=config.grid_size,
            num_mixtures = config.num_mixtures,)
        config.len_seqs = config.dimension_gaussian
        config.vocab_size = vocab_sz
        return src, tgt, {"edges": edges, "mu_s": mu_s, "mu_t": mu_t}

    # ----------------------- discrete -----------------------
    if config.data_type == "discrete":
        src_probs, tgt_probs, vocab_size, seq_len = create_probability_maps(grid_max=config.grid_size_discrete,
                                           source_dist=config.source_dist, target_dist=config.target_dist,distance=config.distance,)
        src = sample_from_prob_map(src_probs, config.number_of_source_samples)
        tgt = sample_from_prob_map(tgt_probs, config.number_of_target_samples)
        config.len_seqs = seq_len
        config.vocab_size = vocab_size
        return src, tgt, {"grid_shape": src_probs.shape, "src_probs": src_probs, "tgt_probs": tgt_probs,
                          "edges": torch.arange(0, vocab_size, 1)}

    if config.data_type == "point_forms":
        config.grid_size = 6
        config.quantization_step = 0.04
        src, vocab_size, edges = discrete_gray_codes_forms(form=config.source_form, grid_min=-config.grid_size, grid_max=config.grid_size,
                                                    discrete_step=config.quantization_step,n_samples=config.number_of_source_samples)
        tgt, _, _ = discrete_gray_codes_forms(form=config.target_form, grid_min=-config.grid_size, grid_max=config.grid_size,
                                           discrete_step=config.quantization_step,n_samples=config.number_of_target_samples)
        config.len_seqs = 2
        config.vocab_size = vocab_size
        return src, tgt, {"edges": edges, "src_form": config.source_form, "tgt_form": config.target_form}

    raise ValueError(f"Unknown data_type: {config.data_type}")


# =======================================================================
#  Function to evaluate raw data
# =======================================================================
def evaluate_raw_data(src, tgt, extras):
    """Quick sanity checks / visualisations before training."""
    if config.data_type == "markov":
        P_est = estimate_transition_matrix(src, config.vocab_size)
        abs_diff, norm, diag_mean, _ = transition_stats(P_est, extras["S"], config)
        if config.type_of_T_matrix == "random":
            P_est_tgt = estimate_transition_matrix(tgt, config.vocab_size)
            abs_diff_tgt, norm_tgt, diag_mean_tgt,_ = transition_stats(P_est_tgt, extras["T"], config)
        abs_diff, norm, diag_mean, _ = transition_stats(P_est, extras["T"], config)
    elif config.data_type == "gaussian" or config.data_type == "point_forms":
        edges = extras["edges"]
        plot_iid_gmm_points(tgt, edges, title=f"Target samples n = {config.number_of_target_samples}")
        plot_iid_gmm_points(src, edges, title=f"Source samples n = {config.number_of_source_samples}")

    elif config.data_type == "discrete":
        plot_prob_map(extras["src_probs"], title="Source distribution")
        plot_prob_map(extras["tgt_probs"], title="Target distribution (shifted)")
        #plot_sample_counts(src,grid_shape=extras["grid_shape"],title="Observed sample counts")
        #plot_sample_counts(tgt,grid_shape=extras["grid_shape"], title="Observed sample counts")



# =======================================================================
# Main function
# =======================================================================
def main() -> None:
    seed_everything(config.seed)

    # --------------------------------------------------------------------------
    # 1 ▸ Data -----------------------------------------------------------------
    src, tgt, extras = load_data()
    evaluate_raw_data(src, tgt, extras)

    wandb.log({
        "data/num_src": len(src),
        "data/num_tgt": len(tgt),
    })

    if config.test_run:
        config.train_classifier_epochs = 1
        config.train_ratio_estimator_epochs = 1
        config.train_denoiser_epochs = 1
    if config.unguided_sampling:
        config.sample_gammas = [0.0]

    # --------------------------------------------------------------------------
    # 2 ▸ Domain classifier ----------------------------------------------------
    if not config.skip_ratio_estimator_on_clean_data or not config.unguided_sampling:
        classifier = ClassiferNet(
                vocab_sz=config.vocab_size, seq_len=config.len_seqs, output_with_sigmoid=False
            )
    
        classifier = train_domain_classifier(
            model=classifier,
            source_data=src,
            target_data=tgt,
            epochs=config.train_classifier_epochs,
            batch_size=config.batch_size,
            device=config.device,
        )
        validate_domain_classifier(
            classifier,
            src,
            tgt,
            batch_size=config.batch_size,
            device=config.device,)


        classifier_t = RatioNetAdaLN(
                vocab_sz=config.vocab_size, seq_len=config.len_seqs, output_with_sigmoid=False)

        classifier_t = train_time_dependent_classifier(
            model=classifier_t,
            noise_sched = LogLinearNoise(),
            source_data=src,
            target_data=tgt,
            mask_idx=config.vocab_size,
            vocab_size=config.vocab_size + 1,
            epochs=config.train_classifier_epochs,
            batch_size=config.batch_size,
            device=config.device,
        )
        validate_domain_classifier(
            classifier_t,
            src,
            tgt,
            batch_size=config.batch_size,
            device=config.device,)



    # --------------------------------------------------------------------------
    # 3 ▸ Denoiser -------------------------------------------------------------
    denoiser = AdaLNDenoiser(
        vocab_size=config.vocab_size + 1,
        seq_len=config.len_seqs,
        d=config.denoiser_dim,
        heads=config.denoiser_heads,
        layers=config.denoiser_layers,
    )
    denoiser_source = train_denoiser(
        denoiser=denoiser,
        train_seqs=src,
        diffusion="absorbing_state",
        mask_idx=config.vocab_size,
        vocab_size=config.vocab_size + 1,
        epochs=config.train_denoiser_epochs,
        batch_size=config.batch_size,
        device=config.device,
    )


    # --------------------------------------------------------------------------
    # 4 ▸ Ratio estimator (noisy) ---------------------------------------------
    if not config.unguided_sampling:
        ratios_epochs_dict = {
            "ratio scaler only on src": 30,
            "ratio scaler guided src and tgt": 30,
            "ratio scaler on t-classifier": 30,
            "ratio scaler on t-classifier + clean": 30,
            "ratio scaler like TLDM": 30,
            "ratio vector on t-classifier": 30,
            "ratio vector on (t-classifier + clean)": 30,
            "ratio vector on pre-trained nrm ratio": 30,
            "ratio vector on pre-trained ratio + t-clas": 30,
        }

        # --- ratio estimator on only source domain ---
        ratio_net_guided_only_on_source = RatioNetAdaLN(vocab_sz=config.vocab_size + 1, seq_len=config.len_seqs)
        ratio_net_guided_only_on_source = train_ratio_estimator(
            model=ratio_net_guided_only_on_source,
            domain_classifier=classifier,
            source_data=src,
            target_data=tgt,
            noise_sched=LogLinearNoise(),
            diffusion="absorbing_state",
            mask_idx=config.vocab_size,
            vocab_size=config.vocab_size + 1,
            epochs=ratios_epochs_dict["ratio scaler only on src"],
            batch_size=config.batch_size,
            device=config.device,
            only_source_domain=True,
        )
        
        # --- ratio estimator on only target and source domain ---
        ratio_net_guided = RatioNetAdaLN(vocab_sz=config.vocab_size + 1, seq_len=config.len_seqs)
        ratio_net_guided = train_ratio_estimator(
            model=ratio_net_guided,
            domain_classifier=classifier,
            source_data=src,
            target_data=tgt,
            noise_sched=LogLinearNoise(),
            diffusion="absorbing_state",
            mask_idx=config.vocab_size,
            vocab_size=config.vocab_size + 1,
            epochs=ratios_epochs_dict["ratio scaler guided src and tgt"],
            batch_size=config.batch_size,
            device=config.device,
        )

        ratio_net_guided_only_time = RatioNetAdaLN(vocab_sz=config.vocab_size + 1, seq_len=config.len_seqs)
        ratio_net_guided_only_time = ratio_trained_on_time_dependent_classifier(
            model=ratio_net_guided_only_time,
            domain_classifier_t=classifier_t,
            domain_classifier=classifier,
            denoiser_model=denoiser_source,
            source_data=src,
            target_data=tgt,
            noise_sched=LogLinearNoise(),
            diffusion="absorbing_state",
            mask_idx=config.vocab_size,
            vocab_size=config.vocab_size + 1,
            epochs=ratios_epochs_dict["ratio scaler on t-classifier"],
            batch_size=config.batch_size,
            device=config.device,
            lambda_reconstruction=0,
            lambda_clean_ratio=0,  # regularization term for ratio net
        )


        # --- ratio estimator on time dependent classifier with denoiser reconstruction loss ---
        ratio_net_guided_only_time_clean = RatioNetAdaLN(vocab_sz=config.vocab_size + 1, seq_len=config.len_seqs)
        ratio_net_guided_only_time_clean = ratio_trained_on_time_dependent_classifier(
            model=ratio_net_guided_only_time_clean,
            domain_classifier_t=classifier_t,
            domain_classifier=classifier,
            denoiser_model=denoiser_source,
            source_data=src,
            target_data=tgt,
            noise_sched=LogLinearNoise(),
            diffusion="absorbing_state",
            mask_idx=config.vocab_size,
            vocab_size=config.vocab_size + 1,
            epochs=ratios_epochs_dict["ratio scaler on t-classifier + clean"],
            batch_size=config.batch_size,
            device=config.device,
            lambda_reconstruction=0.0,
            lambda_clean_ratio=0.1,  # regularization term for ratio net
        )

        # --- ratio estimator with regularization like TLDM paper ---
        ratio_net_guided_tldm = RatioNetAdaLN(vocab_sz=config.vocab_size + 1, seq_len=config.len_seqs)
        ratio_net_guided_tldm = train_ratio_network_with_regularization_like_tldm_paper(
            model = ratio_net_guided_tldm,
            domain_classifier = classifier,
            domain_classifier_t = classifier_t,
            denoiser_model = denoiser_source,
            eta1and2 = [0.1, 0],
            source_data = src,
            target_data = tgt,
            noise_sched = LogLinearNoise(),
            diffusion = "absorbing_state",
            mask_idx = config.vocab_size,
            vocab_size = config.vocab_size + 1,
            epochs = ratios_epochs_dict["ratio scaler like TLDM"],
            batch_size = config.batch_size,
            device = config.device,
            )
       
        # ------------ ratio estimator in vector ---------------
        t1 = time.time()

        ratio_net_guided_vector_clf = RatioNetAdaLNVector(vocab_sz=config.vocab_size + 1, seq_len=config.len_seqs)

        ratio_net_guided_vector_clf = vector_ratio_training(
            model=ratio_net_guided_vector_clf,
            pretrained_ratio_net=ratio_net_guided_only_time_clean,
            domain_classifier_t=classifier_t,
            source_data=src,
            target_data=tgt,
            noise_sched=LogLinearNoise(),
            diffusion="absorbing_state",
            mask_idx=config.vocab_size,
            vocab_size=config.vocab_size + 1,
            epochs=ratios_epochs_dict["ratio vector on (t-classifier + clean)"],
            batch_size=config.batch_size,
            device=config.device,
            eta_clf_and_ratio=(0,1),
            lambda_hinge=0.1,
        )
        t2 = time.time()
        print(f"Time taken to train ratio net vector: {t2-t1:.2f} seconds")


        # ------------ ratio estimator in vector ---------------
        t1 = time.time()
        ratio_net_guided_vector_clf_time = RatioNetAdaLNVector(vocab_sz=config.vocab_size + 1, seq_len=config.len_seqs)

        ratio_net_guided_vector_clf_time = vector_ratio_training(
            model=ratio_net_guided_vector_clf_time,
            pretrained_ratio_net=ratio_net_guided_only_time_clean,
            domain_classifier_t=classifier_t,
            source_data=src,
            target_data=tgt,
            noise_sched=LogLinearNoise(),
            diffusion="absorbing_state",
            mask_idx=config.vocab_size,
            vocab_size=config.vocab_size + 1,
            epochs=ratios_epochs_dict["ratio vector on t-classifier"],
            batch_size=config.batch_size,
            device=config.device,
            eta_clf_and_ratio=(1,0),
            lambda_hinge=0.1,
        )
        t2 = time.time()
        print(f"Time taken to train ratio net vector: {t2-t1:.2f} seconds")

        # ── save trained ratio networks ──
        # --- ratio estimator in vector ---
        t1 = time.time()
        ratio_net_guided_vector_ratio = RatioNetAdaLNVector(vocab_sz=config.vocab_size + 1, seq_len=config.len_seqs)

        ratio_net_guided_vector_ratio = vector_ratio_training(
            model=ratio_net_guided_vector_ratio,
            pretrained_ratio_net=ratio_net_guided,
            domain_classifier_t=classifier_t,
            source_data=src,
            target_data=tgt,
            noise_sched=LogLinearNoise(),
            diffusion="absorbing_state",
            mask_idx=config.vocab_size,
            vocab_size=config.vocab_size + 1,
            epochs=ratios_epochs_dict["ratio vector on pre-trained nrm ratio"],
            batch_size=config.batch_size,
            device=config.device,
            eta_clf_and_ratio=(0,1),
            lambda_hinge=0.1,
        )
        t2 = time.time()
        print(f"Time taken to train ratio net vector: {t2-t1:.2f} seconds")


        t1 = time.time()
        ratio_net_guided_vector_both = RatioNetAdaLNVector(vocab_sz=config.vocab_size + 1, seq_len=config.len_seqs)
        ratio_net_guided_vector_both = vector_ratio_training(
            model=ratio_net_guided_vector_both,
            pretrained_ratio_net=ratio_net_guided_tldm,
            domain_classifier_t=classifier_t,
            source_data=src,
            target_data=tgt,
            noise_sched=LogLinearNoise(),
            diffusion="absorbing_state",
            mask_idx=config.vocab_size,
            vocab_size=config.vocab_size + 1,
            epochs=ratios_epochs_dict["ratio vector on pre-trained ratio + t-clas"],
            batch_size=config.batch_size,
            device=config.device,
            eta_clf_and_ratio=(.25, 1),
            lambda_hinge=0.1,
        )
        t2 = time.time()
        print(f"Time taken to train ratio net vector: {t2-t1:.2f} seconds")


        ratios_dict = {
            "ratio scaler only on src": ratio_net_guided_only_on_source,
            "ratio scaler guided src and tgt": ratio_net_guided,
            "ratio scaler on t-classifier": ratio_net_guided_only_time,
            "ratio scaler on t-classifier + clean": ratio_net_guided_only_time_clean,
            "ratio scaler like TLDM": ratio_net_guided_tldm,
            "ratio vector on (t-classifier + clean)": ratio_net_guided_vector_clf,
            "ratio vector on t-classifier": ratio_net_guided_vector_clf_time,
            "ratio vector on pre-trained nrm ratio": ratio_net_guided_vector_ratio,
            "ratio vector on pre-trained ratio + t-clas": ratio_net_guided_vector_both,
        }
        for ratio_net_name, ratio_net in ratios_dict.items():
            if ratios_epochs_dict[ratio_net_name] == 1:
                continue
            print(f"[Ratio estimator] {ratio_net_name} trained")
            ratio_net.to(config.device)
            ratio_net.eval()
            plot_ratio_estimator(source_samples=src, target_samples=tgt,
                ratio_net=ratio_net,extras=extras,config=config,
                                 extra_title=f"+ {ratio_net_name}")
    else:
        ratio_net_guided = RatioNetAdaLN(vocab_sz=config.vocab_size + 1, seq_len=config.len_seqs)
        ratios_dict = {
            "no ratio used": ratio_net_guided,
        }



    # --------------------------------------------------------------------------
    # 6 ▸ Sampling -------------------------------------------------------------
    cfg = DiffusionConfig()
    cfg.vocab_size = config.vocab_size + 1
    cfg.seq_len = config.len_seqs
    cfg.mask_idx = config.vocab_size
    cfg.batch_size = config.sample_batch_size
    cfg.T_sampling = config.denoising_steps
    cfg.stochastic_sampling_jitter_mask = config.stochastic_sampling_jitter_mask
    cfg.stochastic_sampling_eps_noise = config.stochastic_sampling_eps_noise
    cfg.use_plg = False
    type = "LRG"
    cfg.k_best_sampling = -1
    gamma = 1
    gamma_0_done = False
    for ratio_net_name, ratio_net in ratios_dict.items():
            if ratios_epochs_dict[ratio_net_name] == 1:
                continue
            ratio_net_name_buffer = ratio_net_name
            for gamma in [0,1]:
                if not gamma_0_done and gamma == 0:
                    ratio_net_name = "no ratio used"
                    gamma_0_done = True
                elif gamma == 0:
                    continue
                cfg.gamma = gamma
                for kk in [-1]:#[-1, 200, 100, 50, 20, 10, 5, 3]:
                    cfg.k_best_sampling = kk
                    #type = f"k={kk}"
                    print(f"\n[Sampling] gamma={gamma}, type={type}, ratio_net={ratio_net_name}")
                    sampler = Diffusion(denoiser_source, ratio_net, cfg).to(config.device)
                    with torch.inference_mode():
                        sample_trace = sampler.sample_trace(num_samples=8, device=config.device, )
                    #print_trace(sample_trace, sample_idx=0, vocab_size=config.vocab_size + 1, mask_idx=config.vocab_size)

                    tic = time.time()
                    samples = sampler.sample(num_of_samples=config.sample_size, device=config.device)
                    dt = time.time() - tic

                    # ── evaluation / visualisation ──
                    if config.data_type == "markov":
                        print(f"Sampled sequences (γ={gamma}), type={type}, ratio_net={ratio_net_name}")
                        P_est = estimate_transition_matrix(samples.cpu(), config.vocab_size)
                        P_true = extras["S"] if gamma == 0 else extras["T"]
                        abs_diff, mean_diff, mean_diag, mean_extra = transition_stats(P_est, extras["S"], config)
                        abs_diff, mean_diff, mean_diag, mean_extra = transition_stats(P_est, P_true, config)
                    elif config.data_type == "gaussian" or config.data_type == "point_forms":
                        plot_iid_gmm_points(samples.cpu(), extras["edges"], title=f"Generated γ={gamma}, type={type}, {ratio_net_name}")
                    elif config.data_type == "discrete":
                        plot_sample_counts(samples.cpu(),
                                           grid_shape=extras["grid_shape"],
                                           title=f"Generated samples γ={gamma}, type={type}, {ratio_net_name}",
                                           )
                    ratio_net_name = ratio_net_name_buffer

if __name__ == "__main__":
    main()
