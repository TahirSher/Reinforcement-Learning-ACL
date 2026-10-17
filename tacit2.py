#!/usr/bin/env python3
"""TACIT-2 -- Targeted, Anchored, Closed-loop Iterated Tilting with a certified line search.

Implicit-feedback alignment of a customer-service LLM for customer satisfaction under hard constraints
on task information, hygiene and length, with a certified non-degradation test.  One self-contained
script: data, models, judge, training, ablations, baselines, compute-matched evaluation, statistics,
human-study tooling, a claims audit, a certification-rate analysis, and `diagnose-v1`, which turns
TACIT-1's publication tables into the evidence below.

OBJECTIVE (unchanged from TACIT-1)
    Raise the customer's implicit affect (next-turn emotion of a validated simulated customer) without
    explicit ratings, without losing task information, hygiene or length, and never return a policy worse
    than SFT on the measured objectives, with probability at least 1 - alpha.

WHAT TACIT-1 SHOWED (publication_tables/; reproduce with `diagnose-v1 --v1-dir publication_tables`)
    E1  Every certification failure of TACIT-1 was content non-inferiority, missed narrowly: lower bounds
        -0.0508 and -0.0519 (seeds 44, 42) and -0.0519, -0.0590, -0.0544 (tacit_no_predict) against
        the -0.05 margin, slack -0.0008 to -0.009.  The test compared ONE deployment reply per policy and
        context, so decoder randomness was part of every per-dialogue difference.  These failures came
        from too little power.  They are not large losses of information.
    E2  The information controller could not see the loss.  The deployment probe ran on TRAINING contexts,
        whose gold turns SFT had been fitted on (sft.max_train = 0 = every training turn).  Paired by
        policy, the probe saw a content drift of +0.003 on average where the same policies lost -0.020 on
        held-out selection contexts.  The two signs disagreed in 69% of 45 round pairs.  Length does not
        depend on the gold turn, and there the two agreed (-0.060 vs -0.056).  So the information
        constraint was enforced on a quantity that SFT may have memorised (`validate` now measures it).
    E3  Best-of-n-sized steps hurt, so H3b was REVERSED.  tacit_small_step (0.5 nats a round) certified
        3/3 seeds and beat TACIT by +0.073 outcome, CI95 [0.063, 0.085].  On selection contexts its
        content loss was -0.012, against -0.031 with best-of-n steps (1.2 nats).  Every M-step stopped
        at its KL cap (60/60 rounds of the five MPO arms), so the deployed policy never reached the
        E-step target, and that gap was uncontrolled.  This matches proxy over-optimisation as a function of KL from the
        initial policy (Gao et al., ICML 2023).
    E4  Two TACIT-1 verdicts are artefacts.  H3c "supported" compares TACIT with an ablation that
        returned SFT in all 3 seeds, so it measured the certification fallback and not the
        predicted-pass rule.  H1/H4c effect sizes average certified policies with SFT fallbacks.  Three
        seeds cannot separate certification rates of 3/3 and 1/3 (one-sided Fisher exact p = 0.20).
    E5  Amortisation is reachable.  tacit_small_step at N = 1 reached outcome 0.377 against sft_bon at
        N = 4 with 0.366 (training judge), and 0.275 against 0.245 (external evaluator), with every
        constraint met.

THE METHOD (TACIT-2), per round t; the judge, the E-step algebra and the certificate are TACIT-1's
    0. Held-out RL contexts (fixes E2).  The training dialogues are split once, by a hash of the dialogue
       id, into an SFT part and an RL part (`rl_holdout_frac`).  SFT and the simulator never see the RL
       part.  The E-step, the probe and the line search use only RL contexts, so information is
       measured against gold turns the policy has never been fitted on.
    1. Monte Carlo deployment probe (fixes E2).  K_p deployment-decoder replies from pi_t and from SFT
       per probe context, with common random numbers.  This gives per-context paired drifts of content,
       log length and hygiene, with standard errors over dialogues (one context per dialogue).
    2. Closed-loop E-step (TACIT-1).  q*(y|x) is proportional to 1[feasible] exp((A + lambda I + mu L)/eta_x),
       with moment targets that cancel the measured cumulative drift (integral feedback, Stooke et al.,
       ICML 2020), on the radius eps_t of step 5.
    3. M-step (TACIT-1): weighted MLE of exact pi_t samples, with a k3 KL stop at eps_t.
    4. Certified backtracking line search (new; fixes E3).  theta(a) = theta_t + a (theta_M - theta_t), with
       a in `line_alphas` = (1, 1/2, 1/4).  The largest a is accepted whose probe drifts satisfy the
       TIGHTENED constraints at one-sided confidence `line_conf`:
           content > -tau m_info,  hygiene > -tau m_hyg,  |log length| < tau d_len    (tau = line_tol)
       If no a passes, theta_t is kept.  This is the backtracking line search of TRPO (Schulman et al.,
       ICML 2015) with a deployed-constraint acceptance test in place of the surrogate/KL test.
       Tightening the constraint keeps a buffer for the certificate's own noise, as in constraint
       tightening for safe RL (Chow et al., NeurIPS 2018; Achiam et al., ICML 2017).
    5. Adaptive trust region (new; fixes E3).  eps_{t+1} = min(eps_up eps_t, cap) when a = 1, else
       max(eps_min, eps_down eps_t).  The cap is the best-of-n KL bound log n - (n-1)/n (Beirami et al.,
       ICML 2025).  This is the classical trust-region radius update (Conn, Gould & Toint, 2000).
       TACIT-1's fixed radii are the ablations tacit2_fixed_large and tacit2_fixed_small.
    6. Monte Carlo certification (new; fixes E1).  The selection and safety tests average K = `mc_k`
       deployment replies per context for the judge-free objectives (content, hygiene, length), and
       K_j = `mc_k_judge` for affect and the monitor.  The estimand E_x E_{y~pi}[f] does not change.
       The per-dialogue variance falls from s_b^2 + s_w^2 to s_b^2 + s_w^2 / K (between and within
       dialogue).  The same test with K = 1 runs on the first replies of the same data, which gives a
       PAIRED power comparison (exact McNemar test over runs).
    7. Predicted-pass selection (TACIT-1; Thomas et al., Science 2019), then ONE intersection-union test
       on the dialogue-disjoint safety split.  The selected candidate is saved apart from the returned
       policy, so the optimiser (per-protocol) is evaluated separately from the fallback (E4).

GUARANTEE (unchanged; conditions stated)
    The selected checkpoint depends only on the RL contexts, their pools, the probe and the selection
    split.  The safety split S (G dialogues, one context each) is independent of it.  Averaging K
    replies within a dialogue keeps the dialogue means independent across dialogues, so the bounds keep
    their level.  The line search and the radius use RL data only, so they are part of training and do
    not touch the guarantee.  For affect (superiority), monitor, information and hygiene
    (non-inferiority at the pre-registered margins) and log length (TOST):
        P(theta returned and some null true) <= alpha (+ o(1) for the "t" bound),
    by the intersection-union argument (Berger, 1982).  NOT claimed: (i) human-level effects (the Delta_k
    are properties of the judges); (ii) certification in every seed (a test has power below 1).  The
    margins are IDENTICAL to TACIT-1, CAMPO and CASPI.

ARMS
    tacit2                  full method (steps 0-7)
    tacit2_no_linesearch    a = 1 always                                               -> H3a
    tacit2_fixed_small      fixed radius 0.5 (TACIT-1's best ablation, tacit_small_step) -> H3b
    tacit2_fixed_large      fixed best-of-n radius (TACIT-1's step, ~1.2 nats)         -> H3b
    tacit2_open_loop        moment targets relative to pi_t's exact samples            -> H3d
    tacit1                  TACIT-1's optimiser and tests (best-of-n steps, no line search, one-reply
                            certification) on the shared step-0 data and probe            -> H4c
    campo                   CAMPO: open loop, small steps, pessimism, K = 1            -> H4c
    online_dpo, offline_dpo iterative / one-shot DPO on the same samples and judge     -> H4d
    sft, sft_bon            SFT; best-of-N SFT samples chosen by the judge on an independent pool
    tacit2_bon              best-of-N on the TACIT-2 policy (compute-matched to sft_bon)  -> H4b
    <arm>_cand              the arm's selected candidate before the certification fallback (per-protocol)
    Every arm, baselines included, trains on the same held-out RL contexts (step 0 is a measurement fix,
    not a component).  The power ablation of step 6 is paired within each run and needs no extra arm.

PRE-REGISTERED HYPOTHESES (`claims`; margins in preregistration.json).  TACIT-1 is the pilot that
generated E1-E5.  Report it as such, and test the hypotheses below on fresh seeds only.
    H1   tacit2 > sft on simulated affect (test split)
    H2   tacit2 meets every constraint vs sft (information, hygiene, monitor non-inferior; length TOST)
    H3a  the line search contributes: higher certification rate than tacit2_no_linesearch (Fisher exact),
         or the ablation's candidate violates a constraint that tacit2's meets
    H3b  the adaptive radius contributes: tacit2 > tacit2_fixed_small on outcome with constraints met,
         and contributes over the fixed best-of-n radius (tacit2_fixed_large; same rule as H3a)
    H3c  Monte Carlo certification narrows the content bound (half-width ratio K vs 1 < 1 across runs)
         and raises power (exact McNemar test, paired within runs)
    H3d  the closed loop contributes (vs tacit2_open_loop; same rule as H3a)
    H3e  SFT shows a memorisation gap in content on seen vs held-out training dialogues (motivates step 0)
    H4a  amortisation: tacit2 (N = 1, no judge at inference) is non-inferior to sft_bon (N = 4)
    H4b  compute-matched dominance: tacit2_bon > sft_bon at the same N
    H4c  tacit2 > tacit1 and tacit2 > campo on outcome, constraints met
    H4d  Pareto vs online_dpo / offline_dpo
    H5   external evaluator, judge validity, human study
    H6   no returned policy significantly worse than SFT on any objective
    H7   certification rate over seeds, with a Clopper-Pearson interval (reported, not assumed)
    Five seeds by default: 5/5 vs 1/5 certifications give one-sided Fisher p = 0.024, while 3 seeds can
    never reach p < 0.05.

REFERENCES (peer-reviewed venues; the method's components and the tests it uses)
    Abdolmaleki et al. 2018 ICLR (MPO); Achiam et al. 2017 ICML (CPO); Beirami et al. 2025 ICML (best-of-n
    KL bound); Berger 1982 Technometrics (intersection-union tests); Chow et al. 2018 NeurIPS (Lyapunov
    safe RL); Conn, Gould & Toint 2000 SIAM (trust-region methods); Coste et al. 2024 ICLR (reward-model
    over-optimisation); Dai et al. 2024 ICLR (Safe RLHF); Feng et al. 2022 LREC (EmoWOZ); Gao, Schulman &
    Hilton 2023 ICML (over-optimisation scaling); Gui, Garbacea & Veitch 2024 NeurIPS (BoNBoN: amortising
    best-of-n); Laroche et al. 2019 ICML (SPIBB); Liang & Zeger 1986 Biometrika; Maurer & Pontil 2009 COLT;
    Moskovitz et al. 2024 ICLR (constrained RLHF); Owen 2013 (Monte Carlo); Rafailov et al. 2023 NeurIPS
    (DPO); Schulman et al. 2015 ICML (TRPO); Stooke, Achiam & Abbeel 2020 ICML (PID Lagrangian); Tessler et
    al. 2019 ICLR (RCPO); Thomas et al. 2015 ICML (high-confidence policy improvement); Thomas et al. 2019
    Science (Seldonian); Xiong et al. 2024 ICML (iterative KL-constrained RLHF).

Usage
    python tacit2.py unittest                                  # numpy checks (statistics, E-step, line search, tests)
    python tacit2.py selftest                                  # every stage and arm on tiny random local models (CPU)
    python tacit2.py diagnose-v1 --v1-dir publication_tables   # E1-E5 from TACIT-1's publication tables
    python tacit2.py all --download --models-dir /path/to/hf_cache --out tacit2_run
    TACIT-1's `sft_policy/` and `simulator/` adapters are NOT reusable: they were fitted on the RL dialogues.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
import urllib.request
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

VERSION = "tacit-2"
EPS = 1e-12

ZENODO_FILES = {
    "emowoz-multiwoz.json": "https://zenodo.org/records/6506504/files/emowoz-multiwoz.json?download=1",
    "emowoz-dialmage.json": "https://zenodo.org/records/6506504/files/emowoz-dialmage.json?download=1",
    "data-split.json": "https://zenodo.org/records/6506504/files/data-split.json?download=1",
}
SPLIT_ALIASES = {"train": "train", "dev": "valid", "valid": "valid", "validation": "valid", "test": "test"}
# Ordinal valence of the EmoWOZ labels (neutral, fearful, dissatisfied, apologetic, abusive, excited,
# satisfied), used only to anchor the judge against human labels.
EMOTION_ORDINAL = {4: -2.0, 2: -1.5, 1: -1.0, 3: -0.5, 0: 0.0, 5: 1.0, 6: 1.5}

# Prompt formats are identical to CARO/PACE/CASPI/TACIT-1.
AGENT_SYSTEM = ("You are a helpful customer service agent for a travel and hospitality assistant. "
                "Reply to the customer in one short, concrete, polite turn.")
CUSTOMER_SYSTEM = ("You are the CUSTOMER in a task-oriented conversation. Write only the customer's next turn, "
                   "reacting naturally to what the agent just said.")

ROLE_LEAK_RE = re.compile(r"(?im)^\s*(customer|user|agent|system|assistant)\s*:")
_STOPWORDS = frozenset(
    "a an the and or but if then so to of in on at for from by with about as is are was were be been being "
    "it its this that these those there here i you we they he she me my your our their his her them us "
    "do does did doing have has had can could would should will shall may might must not no yes please "
    "thank thanks sure okay ok would like just also any some all what which who whom when where why how "
    "am pm let know need help anything something nothing everything else another other "
    # courtesy and affect words are not information: counting them would confound information
    # retention with the affect being optimised
    "great glad happy sorry unfortunately welcome afraid apologise apologize apologies certainly "
    "absolutely course problem wonderful perfect lovely hope enjoy hello goodbye good nice have "
    "day evening morning afternoon".split())
SLOT_PATTERNS = {
    "time": re.compile(r"\b\d{1,2}[:.]\d{2}\s*(?:am|pm)?\b|\b\d{1,2}\s*(?:am|pm)\b", re.I),
    "date": re.compile(r"\b(?:mon|tue|wed|thu|fri|sat|sun)\w*\b"
                       r"|\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\b", re.I),
    "price": re.compile(r"[£$€]\s?\d+(?:\.\d{1,2})?|\b\d+\s?(?:pounds?|dollars?|euros?)\b", re.I),
    "ref": re.compile(r"\b(?:ref(?:erence)?|code|number|booking)\s*[:#]?\s*[A-Z0-9]{4,}\b|\b[A-Z]{2,}\d{3,}\b"),
    "postcode": re.compile(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b", re.I),
    "phone": re.compile(r"\b(?:\+?\d[\d\s\-]{7,}\d)\b"),
    "venue": re.compile(r"\b(?:hotel|restaurant|museum|college|hospital|park|station|airport|cafe|pub|"
                        r"theatre|theater|cinema|gallery|club|inn|lodge)\b", re.I),
}

HUMAN_QUESTIONS = {
    "q_satisfaction": "Imagine you are this customer. After which reply would you feel better about how the "
                      "conversation is going?",
    "q_information": "Which reply gives the customer more of the information they need to get their task done?",
    "q_appropriate": "Which reply responds more appropriately to how the customer seems to feel (acknowledges it "
                     "where needed, without sounding insincere or over-the-top)?",
}
CATCH_RESPONSES = ("That is not my problem, figure it out yourself.", "I do not care what you want. Stop asking me.",
                   "Whatever. Ask someone else.")

# Component switches of every trained arm.
#   reward: "sim" (shared-pool simulated customer affect) | "sentiment" (agent-reply sentiment)
#   optimizer: "mpo" (constrained E-step + weighted M-step) | "dpo" (best vs worst baseline)
#   closed_loop: moment targets from the deployment probe | relative to pi_t's exact samples (CAMPO)
#   step: "adaptive" (trust-region radius, step 5) | "bon" (log n - (n-1)/n, TACIT-1) | "small" (0.5, CAMPO)
#   line_search: certified backtracking on the probe (step 4)
#   mc: K-reply Monte Carlo selection and certification (step 6); False = one reply (TACIT-1)
#   predict: predicted-pass candidate selection (Thomas et al., 2019) | plain selection-split test
_MPO = {"reward": "sim", "optimizer": "mpo", "closed_loop": True, "step": "adaptive", "line_search": True,
        "mc": True, "predict": True, "pessimism": False, "select": True, "offline": False}
_DPO = {"reward": "sim", "optimizer": "dpo", "closed_loop": False, "step": "small", "line_search": False,
        "mc": False, "predict": False, "pessimism": False, "select": False, "offline": False}
ARM_FLAGS: Dict[str, Dict[str, Any]] = {
    "tacit2":               dict(_MPO),
    "tacit2_no_linesearch": {**_MPO, "line_search": False},
    "tacit2_fixed_small":   {**_MPO, "step": "small"},
    "tacit2_fixed_large":   {**_MPO, "step": "bon"},
    "tacit2_open_loop":     {**_MPO, "closed_loop": False},
    "tacit1":               {**_MPO, "step": "bon", "line_search": False, "mc": False},
    "campo":                {**_MPO, "closed_loop": False, "step": "small", "line_search": False, "mc": False,
                             "predict": False, "pessimism": True},
    "online_dpo":           dict(_DPO),
    "offline_dpo":          {**_DPO, "offline": True},
}
PRIMARY = "tacit2"
ABLATIONS = ("tacit2_no_linesearch", "tacit2_fixed_small", "tacit2_fixed_large", "tacit2_open_loop")
# Evaluation-only arms.  "<base>_bon" = best-of-N over samples of <base>, chosen by the judge on an
# independent pool (sft_bon is the selection baseline; tacit2_bon is its compute-matched counterpart).
# "<base>_cand" = the candidate <base> selected before the certification fallback (per-protocol analysis).
CAND_ARMS = tuple(f"{a}_cand" for a in (PRIMARY,) + ABLATIONS + ("tacit1",))
EVAL_ONLY_ARMS = ("sft", "sft_bon", f"{PRIMARY}_bon") + CAND_ARMS
KNOWN_ARMS = EVAL_ONLY_ARMS + tuple(ARM_FLAGS)


# ==========================================================================================================
# configuration
# ==========================================================================================================

@dataclass
class GenConfig:
    max_new_tokens: int = 64
    min_new_tokens: int = 4
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0                      # 0 disables top-k (never inherit a model's default top-k)
    repetition_penalty: float = 1.0


# Exact sampling from the policy: the E-step and the M-step KL estimate need samples whose density is the
# model density, so temperature 1 and every truncation disabled.
EXACT_GEN = GenConfig(min_new_tokens=0, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0)


@dataclass
class SFTConfig:
    lr: float = 5e-5
    weight_decay: float = 0.01
    epochs: int = 3
    batch_size: int = 8
    grad_accum: int = 2
    warmup_frac: float = 0.03
    evals_per_epoch: int = 4
    patience: int = 4
    dev_examples: int = 1000
    max_train: int = 0                  # 0 = all SFT-part training turns (the RL part is never used)


@dataclass
class TACITConfig:
    rounds: int = 6
    contexts_per_round: int = 192
    n_samples: int = 8                  # exact samples per context (the E-step support)
    pool_size: int = 32                 # simulated replies per training context (every sample is a proposal)
    kl_holdout: int = 16                # contexts per round used only to estimate KL(pi_t || pi_t+1)
    probe_contexts: int = 256           # held-out RL contexts of the deployment probe (one per dialogue)
    probe_k: int = 4                    # deployment replies per probe context and policy (step 1)
    loop_gain: float = 1.0              # controller gain g of the closed-loop moment targets
    loop_clip_len: float = 0.10         # |per-round length correction| cap (log words)
    loop_clip_info: float = 0.05        # per-round information correction cap (content units)
    eps_init: float = 0.5               # step "adaptive": initial radius (TACIT-1's best fixed radius)
    eps_min: float = 0.1
    eps_up: float = 1.5                 # radius growth after a full step (a = 1)
    eps_down: float = 0.5               # radius shrinkage after a backtracked or rejected step
    epsilon_cap: float = 1.5            # cap of every radius per context (nats)
    small_epsilon: float = 0.5          # step "small": E-step radius and M-step KL stop
    line_alphas: Tuple[float, ...] = (1.0, 0.5, 0.25)   # backtracking fractions (step 4)
    line_tol: float = 0.5               # tau: the line search allows tau x each certification margin
    line_conf: float = 0.8              # one-sided confidence of the line-search bounds
    mc_k: int = 8                       # deployment replies per context for judge-free objectives (step 6)
    mc_k_judge: int = 1                 # ... for affect and the monitor (each costs judge pool scoring)
    kappa: float = 1.0                  # pessimism weight (only arms with pessimism=True use it; TACIT-1 G3)
    lambda_max: float = 10.0            # cap of the information multiplier
    mu_max: float = 5.0                 # cap of |length multiplier|
    min_weight: float = 1e-3            # samples below this E-step weight are not used in the M-step
    lr: float = 2e-5
    epochs: int = 2
    batch: int = 8
    kl_check_every: int = 8             # M-step steps between KL checks
    dpo_beta: float = 0.1               # DPO baselines (CASPI's settings)
    dpo_lr: float = 5e-6
    dpo_nll_coef: float = 0.05
    select_contexts: int = 400          # selection split (heuristic; not part of the guarantee)
    select_alpha: float = 0.10
    predict_inflation: float = 2.0      # predicted-pass rule: CI half-width inflation (Thomas et al., 2019)
    safety_contexts: int = 600          # certification split (one context per dialogue)
    alpha: float = 0.05
    bound: str = "t"                    # "t" (asymptotic) | "bernstein" (finite-sample, conservative)
    guard_pool_size: int = 24
    # Pre-registered margins: IDENTICAL to TACIT-1, CASPI and CAMPO.  Never loosened after seeing results.
    margin_monitor: float = 0.012
    margin_info: float = 0.05
    margin_hygiene: float = 0.02
    length_equiv: float = 0.15
    min_ess_frac: float = 0.2


@dataclass
class Config:
    data_dir: str = "emowoz_data"
    out: str = "tacit2_run"
    device: str = ""                    # "" = cuda if available, else cpu
    models_dir: str = ""
    download: bool = False
    seed: int = 42
    seeds: Tuple[int, ...] = (42, 43, 44, 45, 46)
    rl_holdout_frac: float = 0.15       # share of training dialogues held out of SFT for RL (step 0)
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    monitor_model: str = "SamLowe/roberta-base-go_emotions"
    external_model: str = "microsoft/Phi-3.5-mini-instruct"
    external_emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    load_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    max_len: int = 448
    gen_batch: int = 32
    score_batch: int = 64
    sft: SFTConfig = field(default_factory=SFTConfig)
    sim_sft: SFTConfig = field(default_factory=SFTConfig)
    sim_turns: int = 20000
    sim_max_new_tokens: int = 40
    tacit: TACITConfig = field(default_factory=TACITConfig)
    arms: Tuple[str, ...] = KNOWN_ARMS
    validate_contexts: int = 600
    memorisation_contexts: int = 400    # per group (seen / held-out training dialogues), H3e
    eval_turns: int = 800
    eval_pool_size: int = 48
    bon_n: int = 4                     # CASPI and TACIT-1 used 4: kept for comparability
    external_pool_size: int = 24
    cross_eval_contexts: int = 300
    cross_eval_k: int = 4
    equiv_margin_outcome: float = 0.012
    n_boot: int = 2000
    n_signflip: int = 20000
    human_contexts: int = 150
    human_validity_pairs: int = 200
    human_overlap: float = 0.3
    human_comparisons: Tuple[str, ...] = ("tacit2:sft", "tacit2:sft_bon", "tacit2:tacit1")
    v1_dir: str = "publication_tables"  # diagnose-v1: TACIT-1's publication tables (CSV)

    @property
    def out_dir(self) -> Path:
        return Path(self.out)
# ==========================================================================================================
# generic utilities
# ==========================================================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_logger(out_dir: Path, tag: str) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(f"tacit.{tag}")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    lg.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / f"{tag}.log", encoding="utf-8")):
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable_hash(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def n_words(s: str) -> int:
    return len(str(s).split())


def release_gpu() -> None:
    """Collect garbage and release cached GPU memory (the caller must drop its references first)."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ==========================================================================================================
# statistics
# ==========================================================================================================

def rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks for ties."""
    a = np.asarray(a, float)
    if a.size == 0:
        return a
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(a.size, int)
    inv[sorter] = np.arange(a.size)
    s = a[sorter]
    obs = np.r_[True, s[1:] != s[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], a.size]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    a, b = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    d = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / d) if d > EPS else float("nan")


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return pearson(rankdata(x[ok]), rankdata(y[ok])) if ok.sum() >= 3 else float("nan")


def partial_spearman(x, y, z) -> float:
    """Rank correlation of x and y after removing the rank-linear effect of z from both."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 10:
        return float("nan")
    Z = np.column_stack([np.ones(ok.sum()), rankdata(z[ok])])

    def resid(v):
        return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return pearson(resid(rankdata(x[ok])), resid(rankdata(y[ok])))


def holm(pvals: Sequence[float]) -> List[float]:
    p = list(pvals)
    order = sorted(range(len(p)), key=lambda i: p[i])
    out, prev = [0.0] * len(p), 0.0
    for k, i in enumerate(order):
        prev = max(prev, min(1.0, (len(p) - k) * p[i]))
        out[i] = prev
    return out


def normal_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, relative error < 1.2e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError("normal_ppf needs 0 < p < 1")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    lo = 0.02425
    if p < lo:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > 1.0 - lo:
        return -normal_ppf(1.0 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def t_ppf(p: float, df: float) -> float:
    """Student-t quantile by the Cornish-Fisher expansion in 1/df (Hill, 1970); error < 1e-3 for df >= 10."""
    z = normal_ppf(p)
    g1 = (z ** 3 + z) / 4.0
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96.0
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384.0
    g4 = (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / 92160.0
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3 + g4 / df ** 4


def cluster_means(d: np.ndarray, cluster: Sequence[Any]) -> np.ndarray:
    """Mean of d within each cluster (dialogue): the i.i.d. units of every confidence bound."""
    d = np.asarray(d, float)
    cl = np.asarray(list(cluster), dtype=object)
    ok = np.isfinite(d)
    if not ok.any():
        return np.zeros(0)
    _, inv = np.unique(cl[ok], return_inverse=True)
    return np.bincount(inv, weights=d[ok]) / np.bincount(inv)


def confidence_bound(d: np.ndarray, cluster: Sequence[Any], alpha: float, side: str, method: str,
                     value_range: float) -> float:
    """One-sided (1 - alpha) bound on the mean of cluster means.  side: "lower" | "upper".

    method "t":         mean -/+ t_{1-alpha, G-1} sd / sqrt(G)   (asymptotic coverage)
    method "bernstein": mean -/+ [sqrt(2 V ln(2/alpha) / G) + 7 R ln(2/alpha) / (3 (G - 1))]
                        (Maurer & Pontil, 2009; finite-sample coverage for values in a range of width R)"""
    u = cluster_means(d, cluster)
    G = u.size
    if G < 10:
        return float("nan")
    mu, var = float(u.mean()), float(u.var(ddof=1))
    if method == "t":
        half = t_ppf(1.0 - alpha, G - 1) * math.sqrt(var / G)
    elif method == "bernstein":
        lg = math.log(2.0 / alpha)
        half = math.sqrt(2.0 * var * lg / G) + 7.0 * value_range * lg / (3.0 * (G - 1))
    else:
        raise ValueError(f"unknown bound {method}")
    return mu - half if side == "lower" else mu + half


def cluster_bootstrap_ci(stat_fn: Callable[[np.ndarray], float], cluster: Sequence[Any], n_boot: int, seed: int,
                         conf: float = 0.95) -> Tuple[float, float, float]:
    """Percentile bootstrap that resamples whole clusters (dialogues)."""
    cl = np.asarray(list(cluster), dtype=object)
    uniq, inv = np.unique(cl, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(uniq.size)]
    point = float(stat_fn(np.arange(cl.size)))
    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = np.concatenate([members[j] for j in rng.integers(0, uniq.size, uniq.size)])
        with np.errstate(all="ignore"):
            draws[b] = stat_fn(idx)
    v = draws[np.isfinite(draws)]
    if v.size < max(20, n_boot // 5):
        return point, float("nan"), float("nan")
    a = (1.0 - conf) / 2.0
    return point, float(np.quantile(v, a)), float(np.quantile(v, 1 - a))


def sign_flip_test(d: np.ndarray, cluster: Sequence[Any], n_mc: int, seed: int, exact_max: int = 16) -> Dict[str, Any]:
    """Two-sided cluster sign-flip test of E[d] = 0; the statistic |sum_g s_g S_g| / N targets the row mean.
    Exact enumeration for <= exact_max clusters, otherwise n_mc draws (p >= 1/(n_mc+1))."""
    d = np.asarray(d, float)
    _, inv = np.unique(np.asarray(list(cluster), dtype=object), return_inverse=True)
    S = np.bincount(inv, weights=d)
    N = float(d.size)
    obs = abs(float(S.sum())) / N
    G = S.size
    if G <= exact_max:
        signs = 1.0 - 2.0 * ((np.arange(2 ** G)[:, None] >> np.arange(G)[None, :]) & 1)
        p = float(np.mean(np.abs(signs @ S) / N >= obs - 1e-15))
        return {"p": p, "exact": True, "text": f"{p:.4f}"}
    rng = np.random.default_rng(seed)
    cnt = 0
    for s0 in range(0, n_mc, 2000):
        sg = rng.choice(np.array([-1.0, 1.0]), size=(min(2000, n_mc - s0), G))
        cnt += int(np.sum(np.abs(sg @ S) / N >= obs - 1e-15))
    p = float((cnt + 1) / (n_mc + 1))
    return {"p": p, "exact": False, "text": (f"< {1 / (n_mc + 1):.2g}" if cnt == 0 else f"{p:.4f}")}


def cluster_robust_se(d: np.ndarray, cluster: Sequence[Any]) -> float:
    """CR1 cluster-robust standard error of the row mean (Liang & Zeger, 1986)."""
    d = np.asarray(d, float)
    _, inv = np.unique(np.asarray(list(cluster), dtype=object), return_inverse=True)
    S, n = np.bincount(inv, weights=d), np.bincount(inv).astype(float)
    G = S.size
    if G < 3:
        return float("nan")
    u = S - n * float(S.sum() / n.sum())
    return float(math.sqrt(G / (G - 1.0) * float(np.sum(u ** 2))) / n.sum())


def krippendorff_alpha_interval(units: Dict[str, List[float]]) -> float:
    """Krippendorff's alpha, interval metric, over units with >= 2 values."""
    vals = [np.asarray(v, float) for v in units.values() if len(v) >= 2]
    if not vals:
        return float("nan")
    n = float(sum(v.size for v in vals))
    do = sum(float(np.sum((v[:, None] - v[None, :]) ** 2)) / (v.size - 1) for v in vals) / n
    allv = np.concatenate(vals)
    de = float(np.sum((allv[:, None] - allv[None, :]) ** 2)) / (n * (n - 1))
    return float(1.0 - do / de) if de > EPS else float("nan")


def binomial_n(p1: float, p0: float = 0.5, alpha: float = 0.05, power: float = 0.80) -> int:
    """Non-tie judgements needed for a two-sided test of a win rate p1 against p0."""
    za, zb = normal_ppf(1 - alpha / 2), normal_ppf(power)
    return int(math.ceil(((za * math.sqrt(p0 * (1 - p0)) + zb * math.sqrt(p1 * (1 - p1))) / (p1 - p0)) ** 2))


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), by exact summation."""
    return float(sum(math.comb(n, j) * p ** j * (1.0 - p) ** (n - j) for j in range(max(k, 0), n + 1)))


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> Tuple[float, float]:
    """Exact two-sided interval for a binomial proportion (Clopper & Pearson, 1934), by bisection on the
    binomial tail (no scipy): lo solves P(X >= k | lo) = a/2, hi solves P(X <= k | hi) = a/2."""
    if n <= 0:
        return float("nan"), float("nan")
    a = (1.0 - conf) / 2.0

    def solve(f, target):
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if f(mid) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    low = 0.0 if k == 0 else solve(lambda p: binom_sf(k, n, p), a)
    high = 1.0 if k == n else solve(lambda p: -(1.0 - binom_sf(k + 1, n, p)), -a)
    return low, high


def fisher_exact_greater(k1: int, n1: int, k2: int, n2: int) -> float:
    """One-sided Fisher exact p-value of H1: rate 1 > rate 2 (hypergeometric upper tail of k1)."""
    K, N = k1 + k2, n1 + n2
    den = math.comb(N, n1)
    return float(sum(math.comb(K, j) * math.comb(N - K, n1 - j) for j in range(k1, min(K, n1) + 1)) / den)


def mcnemar_exact(b: int, c: int) -> float:
    """One-sided exact McNemar p-value of H1: b > c, where b and c are the two discordant counts of
    paired binary outcomes (b = pass only under the new rule, c = pass only under the old)."""
    return binom_sf(b, b + c, 0.5) if b + c > 0 else 1.0


def two_sample_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int, conf: float = 0.95
                            ) -> Tuple[float, float, float]:
    """Percentile bootstrap of mean(a) - mean(b) for two independent samples of independent units."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    d = np.asarray([a[rng.integers(0, a.size, a.size)].mean() - b[rng.integers(0, b.size, b.size)].mean()
                    for _ in range(n_boot)])
    q = (1.0 - conf) / 2.0
    return float(a.mean() - b.mean()), float(np.quantile(d, q)), float(np.quantile(d, 1 - q))



# ==========================================================================================================
# the E-step: constrained I-projection on the empirical support of exact policy samples
# ==========================================================================================================

def kl_to_uniform(w: np.ndarray) -> float:
    """KL(w || uniform) over the n entries of w = log n - H(w)."""
    w = np.asarray(w, float)
    nz = w[w > 0]
    return float(math.log(w.size) + np.sum(nz * np.log(nz)))


def _masked_softmax(S: np.ndarray, F: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Row-wise softmax of S / eta over the feasible entries F (rows without feasible entries are zero)."""
    Z = np.where(F, S, -np.inf) / eta[:, None]
    mx = np.max(np.where(F, Z, -np.inf), 1, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    E = np.where(F, np.exp(np.where(F, Z - mx, 0.0)), 0.0)
    den = E.sum(1, keepdims=True)
    return np.divide(E, den, out=np.zeros_like(E), where=den > 0)


def bon_kl_budget(n: Any) -> Any:
    """log n - (n - 1) / n: the upper bound on KL(best-of-n || base) (Beirami et al., ICML 2025), 0 for n <= 1.
    TACIT uses it as the per-round step budget: one round moves as far as one best-of-n selection may."""
    n = np.asarray(n, float)
    out = np.where(n > 1, np.log(np.maximum(n, 1.0)) - (n - 1.0) / np.maximum(n, 1.0), 0.0)
    return float(out) if out.ndim == 0 else out


def solve_temperatures(S: np.ndarray, F: np.ndarray, epsilon: Any, iters: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Per row c: q_c proportional to exp(S_c / eta_c) on F_c with KL(q_c || uniform on F_c) = epsilon.

    This is the closed form of  max_q E_q[S]  s.t.  KL(q || u_F) <= epsilon  (the MPO E-step on an
    empirical support).  KL(q || u) falls monotonically in eta, from log n_c (eta -> 0, the argmax) to 0
    (eta -> inf), so eta_c is found by vectorised bisection on log eta; epsilon may be a scalar or one
    radius per row.  If log n_c <= epsilon_c the
    constraint cannot bind and q_c is the argmax (ties shared); a single feasible sample gets weight 1;
    a row whose feasible scores are all equal stays uniform.  Returns (Q, eta); eta = 0 encodes the argmax."""
    S, F = np.asarray(S, float), np.asarray(F, bool)
    C = S.shape[0]
    n = F.sum(1)
    eps = np.broadcast_to(np.asarray(epsilon, float), (C,))
    Sm = np.where(F, S, np.nan)
    with warnings.catch_warnings():                # rows without feasible samples are all-NaN by design
        warnings.simplefilter("ignore", RuntimeWarning)
        spread = np.nan_to_num(np.nanmax(Sm, 1) - np.nanmin(Sm, 1), nan=0.0)
    flat = spread < 1e-12
    hard = (np.log(np.maximum(n, 1)) <= eps) & ~flat
    soft = (n >= 2) & ~flat & ~hard
    eta = np.full(C, np.inf)
    lo = np.log(np.maximum(spread, 1e-12) * 1e-6)
    hi = np.log(np.maximum(spread, 1e-12) * 1e6)
    if soft.any():
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            Q = _masked_softmax(S, F, np.exp(mid))
            with np.errstate(divide="ignore", invalid="ignore"):
                kl = np.log(np.maximum(n, 1)) + np.sum(np.where(Q > 0, Q * np.log(np.where(Q > 0, Q, 1.0)), 0.0), 1)
            too_peaked = kl > eps
            lo = np.where(soft & too_peaked, mid, lo)
            hi = np.where(soft & ~too_peaked, mid, hi)
        eta = np.where(soft, np.exp(hi), eta)       # hi always satisfies the constraint
    Q = np.zeros_like(S)
    if soft.any():
        Q[soft] = _masked_softmax(S[soft], F[soft], eta[soft])
    for c in np.flatnonzero(~soft & (n >= 1)):
        if flat[c] or n[c] == 1:
            Q[c] = F[c] / n[c]
        else:                                       # hard: argmax with ties shared
            top = F[c] & (S[c] >= np.nanmax(Sm[c]) - 1e-12)
            Q[c] = top / top.sum()
            eta[c] = 0.0
    return Q, eta


def estep(R: np.ndarray, I: np.ndarray, L: np.ndarray, F: np.ndarray, epsilon: Any, moments: bool,
          lambda_max: float, mu_max: float, fixed_eta: Optional[float] = None, iters: int = 30,
          len_target: float = 0.0, info_target: float = 0.0) -> Dict[str, Any]:
    """Constrained target over the N samples of each of C contexts (arrays C x N).

    R: pessimistic reward; I: task information vs the real agent turn (0 where the turn carries none);
    L: log word count; F: feasibility (hygiene and a sampled EOS).  The target is the I-projection
        q*_c(k) proportional to F_ck exp((R_ck + lambda I_ck + mu L_ck) / eta_c),
    with eta_c solving the per-context KL radius (scalar or per row; or fixed_eta), mu solving
        g_L(mu) = mean_c [ E_q[L_c] - mean_k L_ck ] = len_target        (length moment; E_{pi_t} = sample mean)
    and lambda the smallest value >= 0 with
        g_I(lambda) = mean_c [ E_q[I_c] - mean_k I_ck ] >= info_target  (information moment).
    Open loop (CAMPO): both targets 0.  Closed loop (TACIT): the targets cancel the drift that the
    deployment probe measured against SFT (see train_arm).
    g_L is increasing in mu and g_I increasing in lambda (a larger multiplier tilts toward longer or more
    informative samples), so both are solved by bisection, mu nested inside lambda.  The means run over
    contexts with at least one feasible sample.  If a constraint cannot be met inside its bracket the
    multiplier stays at the bound and the status records it.  With moments=False both multipliers are 0."""
    R, I, L = (np.asarray(x, float) for x in (R, I, L))
    F = np.asarray(F, bool)
    rows = F.any(1)
    if not rows.any():
        return {"weights": np.zeros_like(R), "lambda": 0.0, "mu": 0.0, "status": "no feasible sample",
                "eta": np.full(R.shape[0], np.nan), "g_info": 0.0, "g_len": 0.0, "info_target": info_target,
                "len_target": len_target, "info_slack": 0.0, "len_slack": 0.0}

    def solve(lam: float, mu: float):
        S = R + lam * I + mu * L
        if fixed_eta:
            eta = np.full(R.shape[0], float(fixed_eta))
            Q = _masked_softmax(S, F, eta)
        else:
            Q, eta = solve_temperatures(S, F, epsilon)
        gI = float(np.mean((Q * I).sum(1)[rows] - I.mean(1)[rows])) - info_target
        gL = float(np.mean((Q * L).sum(1)[rows] - L.mean(1)[rows])) - len_target
        return Q, eta, gI, gL

    def best_mu(lam: float):
        lo, hi = -mu_max, mu_max
        if solve(lam, lo)[3] > 0:
            return lo, "length above target at -mu_max"
        if solve(lam, hi)[3] < 0:
            return hi, "length below target at +mu_max"
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if solve(lam, mid)[3] < 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi), "ok"

    lam, mu, status = 0.0, 0.0, "moments off"
    if moments:
        mu, ls = best_mu(0.0)
        if solve(0.0, mu)[2] >= 0:
            status = f"info inactive; length {ls}"
        else:
            hi = 0.05
            while hi < lambda_max and solve(hi, best_mu(hi)[0])[2] < 0:
                hi *= 2.0
            hi = min(hi, lambda_max)
            if solve(hi, best_mu(hi)[0])[2] < 0:
                lam = lambda_max
                status = "info infeasible at lambda_max"
            else:
                lo = 0.0
                for _ in range(iters):
                    mid = 0.5 * (lo + hi)
                    if solve(mid, best_mu(mid)[0])[2] >= 0:
                        hi = mid
                    else:
                        lo = mid
                lam = hi
                status = "info active"
            mu, ls = best_mu(lam)
            status += f"; length {ls}"
    Q, eta, gI, gL = solve(lam, mu)
    # g_info / g_len are reported as achieved moment shifts (target added back), slack = achieved - target
    return {"weights": Q, "lambda": lam, "mu": mu, "status": status, "eta": eta, "g_info": gI + info_target,
            "g_len": gL + len_target, "info_target": info_target, "len_target": len_target,
            "info_slack": gI, "len_slack": gL}


def estep_summary(est: Dict[str, Any], R: np.ndarray, F: np.ndarray) -> Dict[str, Any]:
    """Diagnostics of an E-step: how far the target moved and what it bought under the reward."""
    Q, F = est["weights"], np.asarray(F, bool)
    rows = F.any(1)
    multi = F.sum(1) > 1
    kls = [kl_to_uniform(Q[c][F[c]]) for c in np.flatnonzero(multi)]
    fin = est["eta"][np.isfinite(est["eta"])]
    gain = [(Q[c] * R[c]).sum() - R[c][F[c]].mean() for c in np.flatnonzero(rows)]
    return {"lambda": est["lambda"], "mu": est["mu"], "status": est["status"], "g_info": est["g_info"],
            "g_len": est["g_len"], "info_target": est["info_target"], "len_target": est["len_target"], "feasible_share": float(F.mean()), "contexts_with_signal": int(multi.sum()),
            "kl_median": float(np.median(kls)) if kls else float("nan"),
            "eta_median": float(np.median(fin)) if fin.size else float("nan"),
            "reward_gain_vs_feasible_mean": float(np.mean(gain)) if gain else float("nan")}


def vector_test(diffs: Dict[str, np.ndarray], cluster: Sequence[Any], alpha: float, pc: "TACITConfig",
                ess_median: float, pool_size: int) -> Dict[str, Any]:
    """Intersection-union test of improvement over SFT (see the module docstring for the guarantee).

    diffs: per-context paired differences new - SFT for affect, monitor, content and hygiene, and the log
    word ratio for length.  Every component is one-sided at level alpha; the policy passes only if all
    pass.  Unlike CASPI there is no "beat the best round" rule: selection compares lower bounds instead.
    power_at_zero_difference is the chance of passing a non-inferiority test when the true difference is
    0, given the observed spread; dialogues_for_80pct_power plans the NEXT pre-registration."""
    m = pc.bound
    lb = {k: confidence_bound(diffs[k], cluster, alpha, "lower", m, r) for k, r in
          (("affect", 4.0), ("monitor", 4.0), ("content", 2.0), ("hygiene", 2.0), ("length", 2.0))}
    ub_len = confidence_bound(diffs["length"], cluster, alpha, "upper", m, 2.0)
    mean = {k: float(np.nanmean(v)) if np.isfinite(v).any() else float("nan") for k, v in diffs.items()}
    margins = {"monitor": pc.margin_monitor, "content": pc.margin_info, "hygiene": pc.margin_hygiene,
               "length": pc.length_equiv}
    checks = {"affect_superior": bool(math.isfinite(lb["affect"]) and lb["affect"] > 0),
              "monitor_noninferior": bool(math.isfinite(lb["monitor"]) and lb["monitor"] > -pc.margin_monitor),
              # content is NaN where the real agent turn carries no task information; the test uses the rest
              "content_noninferior": bool(math.isfinite(lb["content"]) and lb["content"] > -pc.margin_info),
              "hygiene_noninferior": bool(math.isfinite(lb["hygiene"]) and lb["hygiene"] > -pc.margin_hygiene),
              "length_equivalent": bool(math.isfinite(lb["length"]) and lb["length"] > -pc.length_equiv
                                        and ub_len < pc.length_equiv),
              "ess_floor": bool(ess_median >= pc.min_ess_frac * pool_size)}
    za, zb = normal_ppf(1 - alpha), normal_ppf(0.8)
    power, n80 = {}, {}
    for k, mg in margins.items():
        u = cluster_means(diffs[k], cluster)
        sd = float(np.std(u, ddof=1)) if u.size > 1 else float("nan")
        se = sd / math.sqrt(max(1, u.size))
        ok = math.isfinite(se) and se > 0
        power[k] = normal_cdf(mg / se - za) if ok else float("nan")
        n80[k] = int(math.ceil(((za + zb) * sd / mg) ** 2)) if ok else None
    return {"accept": all(checks.values()), "checks": checks, "lower_bounds": lb, "length_upper": ub_len,
            "means": mean, "margins": margins, "power_at_zero_difference": power, "dialogues_for_80pct_power": n80,
            "content_vacuous_share": float(np.mean(~np.isfinite(diffs["content"]))), "ess_median": ess_median,
            "alpha": alpha, "bound": m}


def predict_pass(diffs: Dict[str, np.ndarray], cluster: Sequence[Any], n_safe: int, alpha: float, pc: "TACITConfig",
                 inflation: float) -> Dict[str, Any]:
    """Predict, from selection-split data, whether the SAFETY-split vector test will pass.

    For each objective the safety-split bound is predicted as  mean -/+ inflation * t_{1-alpha, n-1} sd / sqrt(n)
    with n = n_safe dialogues and the selection-split mean and sd of the dialogue means: the candidate-
    selection rule of Thomas et al. (Science 2019), who inflate the predicted half-width by 2 because the
    selection estimate is itself noisy and was chosen as a maximum.  A candidate whose predicted bounds clear
    every threshold is likely to certify; one that only scrapes through on the selection split is not
    (CAMPO seed 43, G4).  Returns the decision, the predicted bounds and the slack of each check."""
    tq = t_ppf(1.0 - alpha, max(n_safe - 1, 10))
    pred, slack = {}, {}
    for k in ("affect", "monitor", "content", "hygiene", "length"):
        u = cluster_means(diffs[k], cluster)
        if u.size < 10:
            return {"pass": False, "predicted": {}, "slack": {}, "reason": f"too few dialogues for {k}"}
        h = inflation * tq * float(np.std(u, ddof=1)) / math.sqrt(n_safe)
        pred[k] = (float(u.mean()) - h, float(u.mean()) + h)
    slack["affect"] = pred["affect"][0]
    slack["monitor"] = pred["monitor"][0] + pc.margin_monitor
    slack["content"] = pred["content"][0] + pc.margin_info
    slack["hygiene"] = pred["hygiene"][0] + pc.margin_hygiene
    slack["length"] = min(pred["length"][0] + pc.length_equiv, pc.length_equiv - pred["length"][1])
    return {"pass": all(v > 0 for v in slack.values()), "predicted": pred, "slack": slack,
            "predicted_affect_lb": pred["affect"][0]}


# ==========================================================================================================
# information, hygiene and text helpers
# ==========================================================================================================

def salient_tokens(text: str) -> set:
    """Task-critical tokens: anything with a digit, capitalised non-initial words that are not stop words,
    and other non-stop words of >= 4 letters.  A transparent lexical proxy, reported as such."""
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-:/.]*[A-Za-z0-9]|[A-Za-z0-9]", norm_text(text))
    out, prev_end = set(), True
    for raw in toks:
        w = raw.lower()
        if any(ch.isdigit() for ch in raw):
            out.add(w)
        elif raw[0].isupper() and not prev_end and w not in _STOPWORDS:
            out.add(w)
        elif len(w) >= 4 and w not in _STOPWORDS:
            out.add(w)
        prev_end = raw.endswith((".", "!", "?"))
    return out


def info_recall(response: str, reference: str) -> float:
    """Share of the reference's salient tokens carried by the response; NaN if the reference has none."""
    ref = salient_tokens(reference)
    return float(len(ref & salient_tokens(response)) / len(ref)) if ref else float("nan")


def slot_prf(response: str, reference: str) -> Tuple[float, float, float]:
    """Precision, recall and F1 of regex task slots (times, dates, prices, references, ...)."""
    def slots(t):
        t = norm_text(t)
        return set().union(*[set(p.findall(t)) for p in SLOT_PATTERNS.values()])
    gold, pred = slots(reference), slots(response)
    if not gold:
        return float("nan"), float("nan"), float("nan")
    hit = len(gold & pred)
    p, r = hit / max(len(pred), 1), hit / len(gold)
    return float(p), float(r), float(2 * p * r / max(p + r, 1e-9))


def content_score(response: str, gold: str) -> float:
    """Task information relative to the real agent turn: mean of salient-token recall and slot recall."""
    v = [x for x in (info_recall(response, gold), slot_prf(response, gold)[1]) if math.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def word_jaccard(a: str, b: str) -> float:
    sa, sb = set(norm_text(a).lower().split()), set(norm_text(b).lower().split())
    return len(sa & sb) / max(1, len(sa | sb))


def trim_to_sentence(text: str) -> str:
    s = ROLE_LEAK_RE.split(norm_text(text))[0].strip()
    m = list(re.finditer(r"[.!?](\s|$)", s))
    return s[: m[-1].end()].strip() if m else s


def hygiene_ok(text: str, min_words: int = 3, max_words: int = 120) -> bool:
    """Well-formedness: length, no role leakage, no word or trigram loops, terminated sentence."""
    s = norm_text(text)
    w = s.split()
    if not (min_words <= len(w) <= max_words) or ROLE_LEAK_RE.search(s) or not re.search(r"[.!?]\s*$", s):
        return False
    run = best = 1
    for a, b in zip(w, w[1:]):
        run = run + 1 if a.lower() == b.lower() else 1
        best = max(best, run)
    tri = [tuple(x.lower() for x in w[i:i + 3]) for i in range(max(0, len(w) - 2))]
    return best < 5 and not (tri and 1.0 - len(set(tri)) / len(tri) > 0.5)


def log_word_ratio(a: Sequence[str], b: Sequence[str]) -> np.ndarray:
    """log((words(a)+1)/(words(b)+1)), clipped to [-1, 1] so that the length test has bounded support."""
    return np.clip(np.log((np.asarray([n_words(x) for x in a], float) + 1.0)
                          / (np.asarray([n_words(x) for x in b], float) + 1.0)), -1.0, 1.0)
# ==========================================================================================================
# data
# ==========================================================================================================

@dataclass
class Turn:
    dialogue_id: str
    uid: str
    split: str
    source: str
    turn_index: int
    history: str
    user_text: str
    gold_response: str
    next_user_text: Optional[str]
    user_emotion: int
    next_emotion: Optional[int]

    @property
    def human_valence(self) -> Optional[float]:
        return None if self.next_emotion is None or self.next_emotion < 0 else EMOTION_ORDINAL.get(int(self.next_emotion))

    @property
    def current_valence(self) -> Optional[float]:
        return None if self.user_emotion is None or self.user_emotion < 0 else EMOTION_ORDINAL.get(int(self.user_emotion))


def agent_prompt(t: Turn) -> str:
    return f"{AGENT_SYSTEM}\n{t.history}\nCustomer: {t.user_text}\nAgent:"


def customer_prompt(t: Turn, response: str) -> str:
    return f"{CUSTOMER_SYSTEM}\n{t.history}\nCustomer: {t.user_text}\nAgent: {norm_text(response)}\nCustomer:"


def context_text(t: Turn) -> str:
    return (t.history + "\n" if t.history else "") + f"Customer: {t.user_text}"


def _annotator_emotion(raw: Any) -> int:
    """EmoWOZ stores a list of annotations per customer turn; index 3 is the aggregated label."""
    if isinstance(raw, int):
        return raw
    if isinstance(raw, list) and raw:
        pick = raw[3] if len(raw) > 3 else raw[-1]
        return int(pick.get("emotion", -1)) if isinstance(pick, dict) else (pick if isinstance(pick, int) else -1)
    if isinstance(raw, dict):
        return int(raw.get("emotion", -1))
    return -1


def download_emowoz(data_dir: Path, logger: logging.Logger) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in ZENODO_FILES.items():
        dst = data_dir / name
        if dst.exists() and dst.stat().st_size > 1000:
            continue
        logger.info("downloading %s", name)
        with urllib.request.urlopen(url, timeout=600) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f)


def load_emowoz(data_dir: Path, logger: logging.Logger, max_history: int = 6) -> List[Turn]:
    paths = {"multiwoz": data_dir / "emowoz-multiwoz.json", "dialmage": data_dir / "emowoz-dialmage.json"}
    sp = data_dir / "data-split.json"
    for p in list(paths.values()) + [sp]:
        if not p.exists():
            raise FileNotFoundError(f"missing {p}; run with --download or place the Zenodo files in {data_dir}")
    split_of: Dict[str, str] = {}
    for key, node in load_json(sp).items():
        canon = SPLIT_ALIASES.get(str(key).lower())
        if canon is not None:
            for i in ([x for sub in node.values() for x in sub] if isinstance(node, dict) else list(node)):
                split_of[str(i)] = canon
    turns: List[Turn] = []
    for src, path in paths.items():
        for did, d in load_json(path).items():
            split = split_of.get(str(did))
            log = d.get("log") if isinstance(d, dict) else None
            if split is None or not isinstance(log, list) or len(log) < 2:
                continue
            texts = [norm_text(u.get("text", "")) for u in log]
            emos = [_annotator_emotion(u.get("emotion")) if i % 2 == 0 else -1 for i, u in enumerate(log)]
            for i in range(0, len(log) - 1, 2):
                if not texts[i] or not texts[i + 1]:
                    continue
                hist = [f"{'Customer' if j % 2 == 0 else 'Agent'}: {texts[j]}"
                        for j in range(max(0, i - max_history), i) if texts[j]]
                nxt = texts[i + 2] if i + 2 < len(log) and texts[i + 2] else None
                nemo = emos[i + 2] if i + 2 < len(log) else None
                turns.append(Turn(str(did), f"{did}#{i}", split, src, i, "\n".join(hist), texts[i], texts[i + 1],
                                  nxt, int(emos[i]), int(nemo) if nemo is not None else None))
    if not turns:
        raise RuntimeError("no turns parsed; check the EmoWOZ JSON schema")
    logger.info("EmoWOZ | %d turns from %d dialogues", len(turns), len({t.dialogue_id for t in turns}))
    return turns


def select_turns(turns: Sequence[Turn], split: str, limit: int, seed: int, require_next: bool = False,
                 one_per_dialogue: bool = False, exclude_dialogues: Optional[set] = None) -> List[Turn]:
    """Turns of a split; whole dialogues in a seeded order until `limit`.  one_per_dialogue keeps one
    random context per dialogue, so every context is an independent unit for the confidence bounds."""
    rng = random.Random(seed)
    by: Dict[str, List[Turn]] = {}
    for t in turns:
        if t.split == split and (not require_next or t.next_user_text) and \
                (exclude_dialogues is None or t.dialogue_id not in exclude_dialogues):
            by.setdefault(t.dialogue_id, []).append(t)
    keys = sorted(by)
    rng.shuffle(keys)
    out: List[Turn] = []
    for k in keys:
        if limit and len(out) >= limit:
            break
        out.extend([rng.choice(by[k])] if one_per_dialogue else by[k])
    return out[:limit] if limit else out


def assign_rl_split(turns: Sequence[Turn], frac: float) -> List[Turn]:
    """Step 0: move a fixed share of the TRAINING dialogues into split "rl".  The assignment hashes the
    dialogue id with a constant salt, so it is identical for every seed, stage and arm, and whole dialogues
    move together.  SFT and the simulator train on "train" (now without "rl"); RL uses only "rl", whose gold
    turns no model has been fitted on (TACIT-1, E2)."""
    if frac <= 0:
        return list(turns)
    cut = int(round(frac * 10000))
    return [dataclasses.replace(t, split="rl") if t.split == "train"
            and stable_hash("tacit2-rl-split", t.dialogue_id) % 10000 < cut else t for t in turns]
# ==========================================================================================================
# models
# ==========================================================================================================

def resolve_local_model(name: str, models_dir: str) -> str:
    """Hugging Face cache layout (models--org--name/snapshots/<hash>) or the name itself."""
    if models_dir:
        root = Path(models_dir) / ("models--" + name.replace("/", "--")) / "snapshots"
        if root.exists():
            snaps = sorted(p for p in root.iterdir() if p.is_dir())
            if snaps:
                return str(snaps[-1])
    return name


class Policy:
    """A causal LM with an optional LoRA adapter.  Used for the agent policy, the customer simulator (whose
    base model, adapter disabled, is the monitor customer) and the out-of-family external customer."""

    def __init__(self, name: str, cfg: Config, logger: logging.Logger, lora: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.cfg, self.logger, self.device = cfg, logger, cfg.device
        path = resolve_local_model(name, cfg.models_dir)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        cuda = self.device.startswith("cuda")
        quant = cfg.load_4bit and cuda
        kw: Dict[str, Any] = {"dtype": torch.bfloat16 if cuda else torch.float32}
        if quant:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16,
                                                           bnb_4bit_use_double_quant=True)
            kw["device_map"] = {"": 0}
        model = AutoModelForCausalLM.from_pretrained(path, **kw)
        if not quant:
            model = model.to(self.device)
        self.lora = lora
        if lora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            if quant:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                        gradient_checkpointing_kwargs={"use_reentrant": False})
            elif cuda:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model.enable_input_require_grads()
            # LoRA dropout 0: training-mode and eval-mode forwards are the same function, so the DPO policy and
            # its frozen reference are scored by the same network.
            model = get_peft_model(model, LoraConfig(
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        self.model = model
        eos = {self.tok.eos_token_id}
        g = getattr(model.generation_config, "eos_token_id", None)
        eos |= set(g) if isinstance(g, (list, tuple)) else ({g} if g is not None else set())
        self.eos_ids = sorted(e for e in eos if e is not None)
        self._prefix_cache: Dict[str, List[int]] = {}
        self._tail_kw: Optional[str] = None
        logger.info("model %s | LoRA=%s | 4bit=%s | eos ids %s", name, lora, quant, self.eos_ids)

    # ---- adapters ------------------------------------------------------------------------------------
    def save_adapter(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))

    def load_adapter(self, path: Path) -> None:
        """Write saved LoRA tensors into the live adapter and verify that every tensor matched."""
        from peft import set_peft_model_state_dict
        path = Path(path)
        st = path / "adapter_model.safetensors"
        if st.exists():
            from safetensors.torch import load_file
            sd = load_file(str(st))
        else:
            import torch
            sd = torch.load(str(path / "adapter_model.bin"), map_location="cpu")
        res = set_peft_model_state_dict(self.model, sd)
        missing = [k for k in getattr(res, "missing_keys", []) if "lora_" in k]
        if missing or getattr(res, "unexpected_keys", []):
            raise ValueError(f"adapter {path} does not match the model ({len(missing)} LoRA tensors missing)")
        self.logger.info("loaded adapter %s", path)

    def snapshot(self) -> Dict[str, Any]:
        return {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def restore(self, snap: Dict[str, Any]) -> None:
        import torch
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in snap:
                    p.copy_(snap[n])

    def interpolate(self, start: Dict[str, Any], end: Dict[str, Any], alpha: float) -> None:
        """Set the trainable tensors to start + alpha (end - start): the line-search path (step 4).  For
        LoRA this is linear in the factors, a smooth path from theta_t (alpha = 0) to theta_M (alpha = 1)."""
        import torch
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in start:
                    p.copy_(start[n] + alpha * (end[n] - start[n]))

    @contextlib.contextmanager
    def frozen(self, snap: Dict[str, Any]):
        """Temporarily swap the trainable tensors for a snapshot (reference-policy scoring)."""
        keep = {}
        for n, p in self.model.named_parameters():
            if n in snap:
                keep[n], p.data = p.data, snap[n]
        try:
            yield
        finally:
            for n, p in self.model.named_parameters():
                if n in keep:
                    p.data = keep[n]

    def _adapter_ctx(self, adapter: bool):
        return self.model.disable_adapter() if (self.lora and not adapter) else contextlib.nullcontext()

    # ---- tokenisation --------------------------------------------------------------------------------
    def prefix_ids(self, text: str) -> List[int]:
        ids = self._prefix_cache.get(text)
        if ids is None:
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            self._prefix_cache[text] = ids
        return ids

    def text_ids(self, text: str) -> List[int]:
        """Token ids of a reply as a continuation: " " + text + EOS (the SFT target format)."""
        return self.tok(" " + norm_text(text), add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]

    # ---- generation ----------------------------------------------------------------------------------
    def generate_ids(self, prompts: Sequence[str], gen: GenConfig, seed: int, adapter: bool = True) -> List[List[int]]:
        """Sampled continuation ids, cut after the first EOS (kept, because it was sampled)."""
        import torch
        self.model.eval()
        out: List[List[int]] = []
        with self._adapter_ctx(adapter):
            for s in range(0, len(prompts), self.cfg.gen_batch):
                ids = [self.prefix_ids(p)[-self.cfg.max_len:] for p in prompts[s:s + self.cfg.gen_batch]]
                n = max(len(x) for x in ids)
                X = torch.full((len(ids), n), self.tok.pad_token_id, dtype=torch.long)
                M = torch.zeros((len(ids), n), dtype=torch.long)
                for i, x in enumerate(ids):
                    X[i, n - len(x):] = torch.tensor(x)
                    M[i, n - len(x):] = 1
                torch.manual_seed(seed + s)
                with torch.no_grad():
                    y = self.model.generate(input_ids=X.to(self.device), attention_mask=M.to(self.device),
                                            do_sample=True, temperature=gen.temperature, top_p=gen.top_p,
                                            top_k=gen.top_k, repetition_penalty=gen.repetition_penalty,
                                            max_new_tokens=gen.max_new_tokens, min_new_tokens=gen.min_new_tokens,
                                            pad_token_id=self.tok.pad_token_id, eos_token_id=self.eos_ids,
                                            use_cache=True)
                for row in y[:, n:].cpu().tolist():
                    cut = next((j for j, tk in enumerate(row) if tk in self.eos_ids), None)
                    out.append(row[:cut + 1] if cut is not None else row)
        return out

    def decode(self, ids: Sequence[int]) -> str:
        return norm_text(self.tok.decode(list(ids), skip_special_tokens=True))

    def generate_text(self, prompts: Sequence[str], gen: GenConfig, seed: int, adapter: bool = True) -> List[str]:
        return [trim_to_sentence(self.decode(x)) for x in self.generate_ids(prompts, gen, seed, adapter)]

    # ---- scoring -------------------------------------------------------------------------------------
    def _forward_tail(self, X, M, pos, keep: int):
        """Forward pass that materialises logits only for the last `keep` positions."""
        if self._tail_kw is None:
            for kw in ("logits_to_keep", "num_logits_to_keep", ""):
                try:
                    extra = {kw: keep} if kw else {}
                    lg = self.model(input_ids=X, attention_mask=M, position_ids=pos, use_cache=False, **extra).logits
                    if lg.shape[1] < keep:
                        raise TypeError("short logits")
                    self._tail_kw = kw
                    return lg[:, -keep:]
                except TypeError:
                    continue
            raise RuntimeError("model forward rejected every logits-slicing convention")
        extra = {self._tail_kw: keep} if self._tail_kw else {}
        return self.model(input_ids=X, attention_mask=M, position_ids=pos, use_cache=False, **extra).logits[:, -keep:]

    def _score(self, prefixes: Sequence[str], conts: Sequence[Sequence[int]]):
        """Summed log p(continuation | prefix) and continuation token counts, as tensors (graph kept).
        The prefix is truncated exactly as in generation (its last max_len tokens) and the continuation is kept
        whole, so a sampled reply is scored under the very context it was sampled from.  Left padding puts
        every continuation at the end of its row, so only the tail logits are needed."""
        import torch
        import torch.nn.functional as F
        seqs, nts = [], []
        for p, c in zip(prefixes, conts):
            seqs.append(self.prefix_ids(p)[-self.cfg.max_len:] + list(c))
            nts.append(len(c))
        n, T = max(len(s) for s in seqs), max(1, max(nts))
        X = torch.full((len(seqs), n), self.tok.pad_token_id, dtype=torch.long)
        M = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            X[i, n - len(s):] = torch.tensor(s)
            M[i, n - len(s):] = 1
        pos = (M.cumsum(1) - 1).clamp_min(0)
        X, M, pos = X.to(self.device), M.to(self.device), pos.to(self.device)
        lg = self._forward_tail(X, M, pos, T + 1)[:, :-1]
        tgt = X[:, n - T:]
        lp = -F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").view(tgt.shape)
        nt = torch.tensor(nts, device=lp.device)
        sel = (torch.arange(T, device=lp.device)[None, :] >= (T - nt)[:, None]).float()
        return (lp * sel).sum(1), nt.float()

    def score(self, prefixes: Sequence[str], conts: Sequence[Sequence[int]], adapter: bool = True) -> np.ndarray:
        """No-grad summed log-probabilities, length-bucketed for throughput."""
        import torch
        self.model.eval()
        out = np.zeros(len(prefixes))
        order = sorted(range(len(prefixes)), key=lambda i: len(self.prefix_ids(prefixes[i])) + len(conts[i]))
        b = self.cfg.score_batch
        with torch.no_grad(), self._adapter_ctx(adapter):
            for s in range(0, len(order), b):
                ix = order[s:s + b]
                lp, _ = self._score([prefixes[i] for i in ix], [conts[i] for i in ix])
                out[ix] = lp.float().cpu().numpy()
        return out

    # ---- training ------------------------------------------------------------------------------------
    def fit_sft(self, train: Sequence[Tuple[str, str]], dev: Sequence[Tuple[str, str]], sc: SFTConfig,
                best_dir: Path, tag: str) -> Dict[str, Any]:
        """Token-level NLL on (prompt, target) pairs, cosine schedule, early stopping on dev NLL."""
        import torch
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import LambdaLR
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = AdamW(params, lr=sc.lr, weight_decay=sc.weight_decay)
        steps_per_epoch = max(1, len(train) // (sc.batch_size * sc.grad_accum))
        total, warm = steps_per_epoch * sc.epochs, max(1, int(sc.warmup_frac * steps_per_epoch * sc.epochs))
        sched = LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                         max(0.05, 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm))))))
        every = max(1, steps_per_epoch // max(1, sc.evals_per_epoch))
        dev_ids = [(p, self.text_ids(t)) for p, t in dev]

        def dev_nll() -> float:
            lp = self.score([p for p, _ in dev_ids], [c for _, c in dev_ids])
            return float(-lp.sum() / max(1, sum(len(c) for _, c in dev_ids)))

        rng = random.Random(1234)
        best, best_step, stale, step, micro = float("inf"), -1, 0, 0, 0
        for _ in range(sc.epochs):
            order = list(range(len(train)))
            rng.shuffle(order)
            self.model.train()
            for s in range(0, len(order) - sc.batch_size + 1, sc.batch_size):
                batch = [train[i] for i in order[s:s + sc.batch_size]]
                lp, nt = self._score([p for p, _ in batch], [self.text_ids(t) for _, t in batch])
                (-lp.sum() / nt.sum().clamp_min(1.0) / sc.grad_accum).backward()
                micro += 1
                if micro % sc.grad_accum:
                    continue
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % every == 0 or step == total:
                    d = dev_nll()
                    if d < best - 1e-4:
                        best, best_step, stale = d, step, 0
                        self.save_adapter(best_dir)
                    else:
                        stale += 1
                    self.logger.info("%s | step %d/%d | dev NLL %.4f (best %.4f) | stale %d/%d", tag, step, total, d,
                                     best, stale, sc.patience)
                    self.model.train()
                    if stale >= sc.patience:
                        break
            if stale >= sc.patience:
                break
        if best_dir.exists():
            self.load_adapter(best_dir)
        return {"best_dev_nll": best, "best_step": best_step, "steps": step}

    def dpo_update(self, pairs: Sequence[Tuple[str, str, str]], weights: Sequence[float], ref: Dict[str, Any],
                   beta: float, lr: float, epochs: int, batch: int, nll_coef: float, seed: int) -> Dict[str, Any]:
        """Weighted DPO against the frozen reference (summed log-probabilities; Rafailov et al., 2023) plus a
        length-normalised NLL anchor on the chosen reply.  Returns the change of the mean chosen and rejected
        log-likelihoods over the update: a negative chosen change is likelihood displacement."""
        import torch
        import torch.nn.functional as F
        from torch.optim import AdamW
        prompts = [p for p, _, _ in pairs]
        ch = [self.text_ids(c) for _, c, _ in pairs]
        rj = [self.text_ids(r) for _, _, r in pairs]
        with self.frozen(ref):
            ref_c, ref_r = self.score(prompts, ch), self.score(prompts, rj)
        pre_c, pre_r = self.score(prompts, ch), self.score(prompts, rj)
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = AdamW(params, lr=lr, weight_decay=0.0)
        rng = random.Random(seed)
        idx = list(range(len(pairs)))
        losses, accs = [], []
        self.model.train()
        for _ in range(max(1, epochs)):
            rng.shuffle(idx)
            for s in range(0, len(idx), batch):
                b = idx[s:s + batch]
                lc, nc = self._score([prompts[i] for i in b], [ch[i] for i in b])
                lr_, _ = self._score([prompts[i] for i in b], [rj[i] for i in b])
                rc = torch.tensor(ref_c[b], device=lc.device, dtype=lc.dtype)
                rr = torch.tensor(ref_r[b], device=lc.device, dtype=lc.dtype)
                w = torch.tensor([float(weights[i]) for i in b], device=lc.device)
                m = (lc - rc) - (lr_ - rr)
                loss = (w * -F.logsigmoid(beta * m)).sum() / w.sum().clamp_min(1e-6)
                loss = loss + nll_coef * (-(lc / nc.clamp_min(1.0))).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                losses.append(float(loss.detach()))
                accs.append(float((m.detach() > 0).float().mean()))
        opt.zero_grad(set_to_none=True)
        post_c, post_r = self.score(prompts, ch), self.score(prompts, rj)
        return {"loss": float(np.mean(losses)), "pref_acc": float(np.mean(accs)), "steps": len(losses),
                "d_chosen_logp": float(np.mean(post_c - pre_c)), "d_rejected_logp": float(np.mean(post_r - pre_r))}

    def fit_weighted(self, prompts: Sequence[str], conts: Sequence[Sequence[int]], weights: Sequence[float],
                     probe_prompts: Sequence[str], probe_conts: Sequence[Sequence[int]], lr: float, epochs: int,
                     batch: int, kl_max: Optional[float], check_every: int, seed: int) -> Dict[str, Any]:
        """M-step: weighted maximum likelihood of exact pi_t samples (forward KL to the E-step target q).

        Loss = - sum_i w_i log pi(y_i | x_i) / (sum_i w_i * mean target length): the sequence-level forward
        KL up to a constant, scaled to token units (no per-sample length normalisation, which would bias the
        objective toward short replies).  probe_* are exact pi_t samples of HELD-OUT contexts (never trained
        on).  With d = log pi_new - log pi_t on them, mean(exp(d) - 1 - d) is an unbiased estimate of
        KL(pi_t || pi_new) whose every term is >= 0 (the "k3" estimator; E[exp(d)] = 1 under pi_t), so,
        unlike mean(-d), sampling noise cannot make it negative and hide a large step.
        Every check_every steps the estimate is compared with kl_max; past it, the weights roll back to the
        last checked state inside the region and the M-step stops (the MPO M-step trust region)."""
        import torch
        from torch.optim import AdamW
        lp_probe0 = self.score(probe_prompts, probe_conts) if len(probe_prompts) else np.zeros(0)

        def probe_kl() -> float:
            if not len(probe_prompts):
                return 0.0
            d = np.minimum(self.score(probe_prompts, probe_conts) - lp_probe0, 30.0)
            return float(np.mean(np.expm1(d) - d))
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = AdamW(params, lr=lr, weight_decay=0.0)
        w_all = np.asarray(weights, float)
        tok_mean = float(np.sum(w_all * np.asarray([len(c) for c in conts])) / max(w_all.sum(), EPS))
        lp_tgt0 = self.score(prompts, conts)
        rng = random.Random(seed)
        idx = list(range(len(prompts)))
        last_ok, kl, steps, stopped, losses, trace = self.snapshot(), 0.0, 0, False, [], []
        for _ in range(max(1, epochs)):
            rng.shuffle(idx)
            self.model.train()
            for s in range(0, len(idx), batch):
                b = idx[s:s + batch]
                lp, _ = self._score([prompts[i] for i in b], [conts[i] for i in b])
                w = torch.tensor(w_all[b], device=lp.device, dtype=lp.dtype)
                loss = -(w * lp).sum() / (w.sum().clamp_min(1e-8) * max(tok_mean, 1.0))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                steps += 1
                losses.append(float(loss.detach()))
                if kl_max is not None and steps % max(1, check_every) == 0:
                    kl = probe_kl()
                    trace.append(round(kl, 4))
                    if kl > kl_max:
                        self.restore(last_ok)
                        stopped = True
                        break
                    last_ok = self.snapshot()
                    self.model.train()
            if stopped:
                break
        opt.zero_grad(set_to_none=True)
        kl = probe_kl()
        lp_tgt1 = self.score(prompts, conts)
        return {"loss": float(np.mean(losses)) if losses else float("nan"), "steps": steps, "kl_stop": stopped,
                "kl_estimate": kl, "kl_trace": trace,
                # weighted mean change of the targets' log-likelihood: must be >= 0 for a working M-step
                "d_target_logp": float(np.sum(w_all * (lp_tgt1 - lp_tgt0)) / max(w_all.sum(), EPS))}
class ValenceClassifier:
    """Valence in [-1, 1] from a sentiment or emotion classifier: P(positive labels) - P(negative labels).
    Labels are mapped by name, so sentiment (positive/negative), GoEmotions and Ekman heads all work."""
    POS = frozenset("positive joy love optimism admiration gratitude approval relief amusement excitement caring "
                    "pride satisfied happiness".split())
    NEG = frozenset("negative anger disgust fear sadness annoyance disappointment disapproval embarrassment grief "
                    "nervousness remorse dissatisfied".split())

    def __init__(self, name: str, cfg: Config, logger: logging.Logger, batch: int = 64, max_len: int = 128):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = resolve_local_model(name, cfg.models_dir)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(cfg.device).eval()
        self.device, self.batch, self.max_len = cfg.device, batch, max_len
        lab = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.sign = np.asarray([1.0 if lab[i] in self.POS else (-1.0 if lab[i] in self.NEG else 0.0)
                                for i in range(len(lab))])
        if not (self.sign > 0).any() or not (self.sign < 0).any():
            raise ValueError(f"{name}: no recognisable positive/negative labels in {lab}")
        self.multilabel = getattr(self.model.config, "problem_type", "") == "multi_label_classification"
        logger.info("valence classifier %s | %d labels | multilabel=%s", name, len(lab), self.multilabel)

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        texts = [norm_text(t) or "." for t in texts]
        out = np.zeros(len(texts))
        with torch.no_grad():
            for s in range(0, len(texts), self.batch):
                enc = self.tok(texts[s:s + self.batch], return_tensors="pt", padding=True, truncation=True,
                               max_length=self.max_len)
                lg = self.model(input_ids=enc["input_ids"].to(self.device),
                                attention_mask=enc["attention_mask"].to(self.device)).logits.float()
                p = (torch.sigmoid(lg) if self.multilabel else torch.softmax(lg, -1)).cpu().numpy()
                v = p @ self.sign
                out[s:s + len(v)] = v / np.maximum(p.sum(1), 1.0) if self.multilabel else v
        return out


def chat_customer_prompt(model: Policy, t: Turn, response: str) -> str:
    """Customer prompt for an instruct model used zero-shot (monitor and external customers)."""
    conv = f"{t.history}\nCustomer: {t.user_text}\nAgent: {norm_text(response)}".strip()
    if not getattr(model.tok, "chat_template", None):
        return f"{CUSTOMER_SYSTEM}\n{conv}\nCustomer:"
    return model.tok.apply_chat_template(
        [{"role": "system", "content": CUSTOMER_SYSTEM},
         {"role": "user", "content": f"Conversation so far:\n{conv}\n\nWrite only the customer's next turn, in one "
                                     f"or two sentences."}], tokenize=False, add_generation_prompt=True)


def simulator_prompt(model: Policy, t: Turn, response: str) -> str:
    return customer_prompt(t, response)
# ==========================================================================================================
# the judge: shared-pool multiple importance sampling over a customer model
# ==========================================================================================================

def snis_weights(log_p: np.ndarray, log_q: np.ndarray) -> Tuple[np.ndarray, float]:
    """Self-normalised importance weights p/q, truncated at mean(w) sqrt(R) (Ionides, 2008); returns (w, ESS)."""
    lw = np.asarray(log_p, float) - np.asarray(log_q, float)
    w = np.exp(lw - np.max(lw))
    w = np.minimum(w, float(np.mean(w)) * math.sqrt(w.size))
    w = w / max(float(w.sum()), EPS)
    return w, float(1.0 / max(float(np.sum(w ** 2)), EPS))


def snis_se(w: np.ndarray, labels: np.ndarray) -> float:
    """Delta-method standard error of a self-normalised IS estimate sum_j w_j l_j (weights summing to 1):
    sqrt(sum_j w_j^2 (l_j - estimate)^2) (Owen, 2013, eq. 9.9)."""
    w, labels = np.asarray(w, float), np.asarray(labels, float)
    return float(math.sqrt(float(np.sum(w ** 2 * (labels - w @ labels) ** 2))))


def balance_log_q(L: np.ndarray) -> np.ndarray:
    """log q(c_j) = log mean_m p(c_j | x, r_m) for a (proposals x replies) matrix of log-probabilities."""
    mx = L.max(0)
    return mx + np.log(np.mean(np.exp(L - mx), 0))


class Judge:
    """E[label(next customer turn) | context, reply] on per-context reply pools.

    A pool for (namespace, context) is drawn once per Judge: `pool_size` replies split evenly over the proposal
    agent turns, sampled at temperature 1 with every truncation disabled, so that the sampling density is the
    model density; the sampled reply token ids are scored exactly.  Each labeller reads every pool reply once.
    Pools are never re-used across runs (a pool built for other proposals would extrapolate); the store file
    is an audit record of the pools of this run.  `adapter=False` turns the simulator into the un-fine-tuned
    base customer (the monitor)."""

    POOL_GEN = GenConfig(min_new_tokens=0, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0)

    def __init__(self, model: Policy, labellers: Dict[str, Callable[[Sequence[str]], np.ndarray]],
                 prompt_fn: Callable[[Policy, Turn, str], str], store: Path, pool_size: int, adapter: bool,
                 max_new_tokens: int, logger: logging.Logger, name: str):
        self.model, self.labellers, self.prompt_fn = model, labellers, prompt_fn
        self.store, self.pool_size, self.adapter, self.logger, self.name = Path(store), pool_size, adapter, logger, name
        self.gen = dataclasses.replace(self.POOL_GEN, max_new_tokens=max_new_tokens)
        self.pools: Dict[str, Dict[str, Any]] = {}
        self._lp: Dict[Tuple[str, Tuple[int, ...]], float] = {}
        if self.store.exists():
            self.store.unlink()

    def _logp(self, prefixes: Sequence[str], conts: Sequence[Sequence[int]]) -> np.ndarray:
        keys = [(hashlib.sha1(p.encode()).hexdigest(), tuple(c)) for p, c in zip(prefixes, conts)]
        todo = {k: (p, c) for k, p, c in zip(keys, prefixes, conts) if k not in self._lp}
        if todo:
            vals = self.model.score([p for p, _ in todo.values()], [c for _, c in todo.values()], adapter=self.adapter)
            self._lp.update(zip(todo.keys(), vals.tolist()))
        return np.asarray([self._lp[k] for k in keys])

    def build(self, ns: str, turns: Sequence[Turn], proposals: Dict[str, List[str]]) -> int:
        """Create the missing pools of namespace `ns`.  proposals[uid] lists the proposal agent turns; the gold
        turn is always added.  Returns the number of pools created."""
        need = []
        for t in {t.uid: t for t in turns}.values():
            key = f"{ns}|{t.uid}"
            if key not in self.pools:
                props = list(dict.fromkeys([norm_text(t.gold_response)] + [norm_text(p) for p in proposals.get(t.uid, [])]))
                need.append((key, t, props))
        if not need:
            return 0
        prompts, owner = [], []
        for ci, (_, t, props) in enumerate(need):
            R = max(self.pool_size, len(props))
            for m, pr in enumerate(props):
                for _ in range(R // len(props) + (1 if m < R % len(props) else 0)):
                    prompts.append(self.prompt_fn(self.model, t, pr))
                    owner.append(ci)
        ids = self.model.generate_ids(prompts, self.gen, seed=stable_hash(ns, need[0][0]), adapter=self.adapter)
        ids = [x if x else [self.model.tok.eos_token_id] for x in ids]
        texts = [self.model.decode(x) or "." for x in ids]
        labels = {k: np.asarray(fn(texts), float) for k, fn in self.labellers.items()}
        owner_a = np.asarray(owner)
        entries = []
        for ci, (key, t, props) in enumerate(need):
            ix = np.flatnonzero(owner_a == ci)
            rep = [ids[i] for i in ix]
            L = np.stack([self._logp([self.prompt_fn(self.model, t, pr)] * len(rep), rep) for pr in props], 0)
            entries.append({"key": key, "uid": t.uid, "proposals": props, "reply_ids": rep,
                            "replies": [texts[i] for i in ix], "log_q": balance_log_q(L).tolist(),
                            "labels": {k: v[ix].tolist() for k, v in labels.items()}})
        for e in entries:
            self.pools[e["key"]] = e
        self.store.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store, "a", encoding="utf-8") as fh:
            for e in entries:
                fh.write(json.dumps(e) + "\n")
        return len(entries)

    def expect(self, ns: str, turns: Sequence[Turn], responses: Sequence[str],
               proposals: Optional[Dict[str, List[str]]] = None) -> Dict[str, np.ndarray]:
        """SNIS expectations of every labeller for each (turn, response); builds missing pools first (with the
        given proposals, or with the responses of this call as proposals)."""
        turns, responses = list(turns), [norm_text(r) for r in responses]
        if proposals is None:
            proposals = {}
            for t, r in zip(turns, responses):
                proposals.setdefault(t.uid, []).append(r)
        self.build(ns, turns, proposals)
        pre, conts, row = [], [], []
        pools = [self.pools[f"{ns}|{t.uid}"] for t in turns]
        for i, (t, r, pl) in enumerate(zip(turns, responses, pools)):
            pre.extend([self.prompt_fn(self.model, t, r)] * len(pl["reply_ids"]))
            conts.extend(pl["reply_ids"])
            row.extend([i] * len(pl["reply_ids"]))
        lp, row = self._logp(pre, conts), np.asarray(row)
        out = {k: np.zeros(len(turns)) for k in self.labellers}
        out.update({f"se_{k}": np.zeros(len(turns)) for k in self.labellers})
        ess = np.zeros(len(turns))
        for i, pl in enumerate(pools):
            w, ess[i] = snis_weights(lp[row == i], np.asarray(pl["log_q"]))
            for k in self.labellers:
                lab = np.asarray(pl["labels"][k])
                out[k][i] = float(w @ lab)
                out[f"se_{k}"][i] = snis_se(w, lab)
        out["ess"] = ess
        return out
# ==========================================================================================================
# experiment scaffolding
# ==========================================================================================================

class Experiment:
    """Holds the configuration, the logger, the turns and lazily loaded auxiliary models of one stage."""

    def __init__(self, cfg: Config, tag: str):
        self.cfg = cfg
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = make_logger(cfg.out_dir, tag)
        seed_everything(cfg.seed)
        self.logger.info("TACIT %s | stage %s | device %s | out %s", VERSION, tag, cfg.device, cfg.out)
        cache = cfg.out_dir / "turns.json"
        if cache.exists():
            self.turns = [Turn(**d) for d in load_json(cache)]
        else:
            if cfg.download:
                download_emowoz(Path(cfg.data_dir), self.logger)
            self.turns = load_emowoz(Path(cfg.data_dir), self.logger)
            dump_json([asdict(t) for t in self.turns], cache)
        self.turns = assign_rl_split(self.turns, cfg.rl_holdout_frac)
        self._cache: Dict[str, Any] = {}

    def _lazy(self, key: str, make: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    @property
    def sentiment(self) -> ValenceClassifier:
        return self._lazy("sent", lambda: ValenceClassifier(self.cfg.sentiment_model, self.cfg, self.logger))

    @property
    def monitor_labeller(self) -> ValenceClassifier:
        return self._lazy("mon", lambda: ValenceClassifier(self.cfg.monitor_model, self.cfg, self.logger))

    def policy(self, adapter: Optional[str]) -> Policy:
        p = Policy(self.cfg.base_model, self.cfg, self.logger)
        if adapter:
            p.load_adapter(self.cfg.out_dir / adapter)
        return p

    def judges(self, sim: Policy, store_tag: str, pool_size: int) -> Tuple[Judge, Judge]:
        """The training judge (fine-tuned customer + sentiment engine) and the monitor (the same weights with
        the adapter disabled, i.e. the base instruct model as customer, read by an independent labeller)."""
        out = self.cfg.out_dir / "pools"
        j = Judge(sim, {"affect": self.sentiment}, simulator_prompt, out / f"{store_tag}_sim.jsonl", pool_size,
                  True, self.cfg.sim_max_new_tokens, self.logger, "simulator")
        m = Judge(sim, {"monitor": self.monitor_labeller}, chat_customer_prompt, out / f"{store_tag}_monitor.jsonl",
                  pool_size, False, self.cfg.sim_max_new_tokens, self.logger, "monitor")
        return j, m


def reply_metrics(texts: Sequence[str], turns: Sequence[Turn]) -> Dict[str, np.ndarray]:
    """Task information, hygiene and length of agent replies (judge-free)."""
    return {"content": np.asarray([content_score(r, t.gold_response) for r, t in zip(texts, turns)]),
            "info_recall": np.asarray([info_recall(r, t.gold_response) for r, t in zip(texts, turns)]),
            "slot_f1": np.asarray([slot_prf(r, t.gold_response)[2] for r, t in zip(texts, turns)]),
            "hygiene": np.asarray([float(hygiene_ok(r)) for r in texts]),
            "words": np.asarray([float(n_words(r)) for r in texts]),
            # the length test's scale: differences of log(words + 1) are the paired log word ratio
            "log_words": np.log(np.asarray([float(n_words(r)) + 1.0 for r in texts]))}


# ==========================================================================================================
# stages 1-3: SFT, simulator, judge validity
# ==========================================================================================================

def stage_data(cfg: Config) -> Dict[str, Any]:
    ex = Experiment(cfg, "0_data")
    per: Dict[str, int] = {}
    for t in ex.turns:
        per[t.split] = per.get(t.split, 0) + 1
    ex.logger.info("turns per split %s", per)
    return per


def stage_sft(cfg: Config) -> Dict[str, Any]:
    ex = Experiment(cfg, "1_sft")
    tr = select_turns(ex.turns, "train", cfg.sft.max_train, cfg.seed)
    dv = select_turns(ex.turns, "valid", cfg.sft.dev_examples, cfg.seed)
    pol = ex.policy(None)
    res = pol.fit_sft([(agent_prompt(t), t.gold_response) for t in tr],
                      [(agent_prompt(t), t.gold_response) for t in dv], cfg.sft, cfg.out_dir / "sft_best", "SFT")
    pol.save_adapter(cfg.out_dir / "sft_policy")
    dump_json(res, cfg.out_dir / "sft.json")
    return res


def stage_simulator(cfg: Config) -> Dict[str, Any]:
    ex = Experiment(cfg, "2_simulator")
    tr = select_turns(ex.turns, "train", cfg.sim_turns, cfg.seed, require_next=True)
    dv = select_turns(ex.turns, "valid", cfg.sim_sft.dev_examples, cfg.seed, require_next=True)
    pol = ex.policy(None)
    res = pol.fit_sft([(customer_prompt(t, t.gold_response), t.next_user_text) for t in tr],
                      [(customer_prompt(t, t.gold_response), t.next_user_text) for t in dv], cfg.sim_sft,
                      cfg.out_dir / "sim_best", "SIM")
    pol.save_adapter(cfg.out_dir / "simulator")
    dump_json(res, cfg.out_dir / "simulator.json")
    return res


def stage_validate(cfg: Config) -> Dict[str, Any]:
    """Construct validity and noise of the judge, on labelled VALID contexts (no policy is involved).

    * between-context anchor: judge(gold reply) vs the real customer's next-turn emotion label, and the
      PARTIAL anchor given the current turn's label (emotional carry-over removed);
    * replication: the same replies on an independent pool (ICC of the context-level estimates);
    * within-context noise: sigma of the gold-vs-SFT affect difference across the two pools versus the
      spread of that difference (the signal a preference has to resolve), and the flip rate."""
    ex = Experiment(cfg, "3_validate")
    turns = [t for t in select_turns(ex.turns, "valid", cfg.validate_contexts, cfg.seed + 3, require_next=True,
                                     one_per_dialogue=True) if t.human_valence is not None]
    pol = ex.policy("sft_policy")
    sft = pol.generate_text([agent_prompt(t) for t in turns], GenConfig(), seed=cfg.seed)
    # H3e: does SFT carry more task information on training dialogues it was fitted on than on held-out
    # training dialogues?  The two groups are random dialogue partitions of one split, so a gap is caused
    # by fitting (memorisation), and it means an information signal measured on seen contexts (TACIT-1's
    # probe) does not transfer.  Units are dialogues (one context each), so the groups are independent.
    mem_seen = select_turns(ex.turns, "train", cfg.memorisation_contexts, cfg.seed + 5, one_per_dialogue=True)
    mem_rl = select_turns(ex.turns, "rl", cfg.memorisation_contexts, cfg.seed + 5, one_per_dialogue=True)
    y_seen = pol.generate_text([agent_prompt(t) for t in mem_seen], GenConfig(), seed=cfg.seed + 55)
    y_rl = pol.generate_text([agent_prompt(t) for t in mem_rl], GenConfig(), seed=cfg.seed + 55)
    del pol
    release_gpu()
    c_seen = reply_metrics(y_seen, mem_seen)["content"]
    c_rl = reply_metrics(y_rl, mem_rl)["content"]
    copy_seen = np.asarray([float(word_jaccard(y, t.gold_response) >= 0.8) for y, t in zip(y_seen, mem_seen)])
    copy_rl = np.asarray([float(word_jaccard(y, t.gold_response) >= 0.8) for y, t in zip(y_rl, mem_rl)])
    memorisation = {"n_seen": len(mem_seen), "n_heldout": len(mem_rl),
                    "content_seen": float(np.nanmean(c_seen)) if c_seen.size else float("nan"),
                    "content_heldout": float(np.nanmean(c_rl)) if c_rl.size else float("nan"),
                    "content_gap": two_sample_bootstrap_ci(c_seen, c_rl, cfg.n_boot, cfg.seed + 9),
                    "gold_copy_gap": two_sample_bootstrap_ci(copy_seen, copy_rl, cfg.n_boot, cfg.seed + 10)}
    ex.logger.info("memorisation (H3e) | SFT content on seen %.4f vs held-out %.4f training dialogues | gap %+.4f "
                   "CI[%+.4f,%+.4f] | near-copy of the gold turn: gap %+.4f CI[%+.4f,%+.4f]",
                   memorisation["content_seen"], memorisation["content_heldout"], *memorisation["content_gap"],
                   *memorisation["gold_copy_gap"])
    sim = ex.policy("simulator")
    judge, _ = ex.judges(sim, "validate", cfg.tacit.pool_size)
    prop = {t.uid: [t.gold_response, r] for t, r in zip(turns, sft)}
    both_t, both_r = turns + turns, [t.gold_response for t in turns] + sft
    e0 = judge.expect("v0", both_t, both_r, prop)["affect"]
    e1 = judge.expect("v1", both_t, both_r, prop)["affect"]
    n = len(turns)
    hv = np.asarray([t.human_valence for t in turns], float)
    cv = np.asarray([t.current_valence if t.current_valence is not None else np.nan for t in turns], float)
    cl = np.asarray([t.dialogue_id for t in turns], dtype=object)
    gold0 = e0[:n]
    d0, d1 = e0[:n] - e0[n:], e1[:n] - e1[n:]
    sigma = float(np.std(d0 - d1, ddof=1) / math.sqrt(2.0))
    res = {"n": n,
           "anchor": cluster_bootstrap_ci(lambda ix: spearman(gold0[ix], hv[ix]), cl, cfg.n_boot, cfg.seed),
           "partial_anchor": cluster_bootstrap_ci(lambda ix: partial_spearman(gold0[ix], hv[ix], cv[ix]), cl,
                                                  cfg.n_boot, cfg.seed + 1),
           "pool_replication_pearson": pearson(e0, e1),
           "within_context_noise_sd": sigma,
           "within_context_signal_sd": float(np.std(0.5 * (d0 + d1), ddof=1)),
           "replication_flip_rate": float(np.mean(np.sign(d0) != np.sign(d1))), "memorisation": memorisation}
    ex.logger.info("judge validity | anchor rho %+.3f CI[%+.3f,%+.3f] | partial (given current emotion) %+.3f "
                   "CI[%+.3f,%+.3f] | pool replication r %.3f | within-context noise sd %.4f vs signal sd %.4f | "
                   "flip rate %.3f", *res["anchor"], *res["partial_anchor"], res["pool_replication_pearson"], sigma,
                   res["within_context_signal_sd"], res["replication_flip_rate"])
    if not res["partial_anchor"][1] > 0:
        ex.logger.warning("the partial anchor does not exclude zero: the judge's response-level human signal is "
                          "unestablished; only the human study can validate within-context rankings")
    dump_json(res, cfg.out_dir / "judge_validity.json")
    return res


# ==========================================================================================================
# training core: Monte Carlo paired checks, probe, line search, adaptive trust region
# ==========================================================================================================

def deploy_replies(pol: Policy, turns: Sequence[Turn], k: int, seed: int, snap: Optional[Dict[str, Any]] = None
                   ) -> List[List[str]]:
    """k deployment-decoder replies per context, as k lists over the contexts.  One generation call with a
    fixed layout, so two policies called with the same seed share their random numbers (CRN)."""
    prompts = [agent_prompt(t) for t in turns]
    with (pol.frozen(snap) if snap is not None else contextlib.nullcontext()):
        flat = pol.generate_text([p for _ in range(k) for p in prompts], GenConfig(), seed)
    n = len(prompts)
    return [flat[j * n:(j + 1) * n] for j in range(k)]


def mc_metrics(reps: List[List[str]], turns: Sequence[Turn]) -> Dict[str, np.ndarray]:
    """k x n matrices of the judge-free objectives of k replies per context."""
    per = [reply_metrics(r, turns) for r in reps]
    return {m: np.stack([p[m] for p in per], 0) for m in ("content", "hygiene", "log_words", "words")}


def paired_dev_check(pol: Policy, ref: Dict[str, Any], judge: Judge, monitor: Judge, turns: Sequence[Turn],
                     ns: str, seed: int, k: int = 1, k_judge: int = 1
                     ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], float, Dict[str, float]]:
    """SFT and the current policy answer the same contexts k times each (CRN), step 6.

    Judge-free objectives (content, hygiene, log length) average all k replies of a context; affect and
    the monitor average the first k_judge, which with both policies' replies and the gold turn are the
    proposals of one FRESH pool per context.  Returns (diffs, diffs_k1, ess, levels): diffs are the
    per-context Monte Carlo means of the paired differences (the estimand E_y[f(pi)] - E_y[f(SFT)] is the
    same for every k), diffs_k1 the same differences from the first reply only (TACIT-1's test, for the
    paired power comparison), and levels carry the variance decomposition."""
    k, kj = max(1, k), max(1, min(k_judge, k))
    y0 = deploy_replies(pol, turns, k, seed, ref)
    y1 = deploy_replies(pol, turns, k, seed)
    n = len(turns)
    props = {t.uid: [y0[j][i] for j in range(kj)] + [y1[j][i] for j in range(kj)] for i, t in enumerate(turns)}
    tt = [t for _ in range(kj) for _ in range(2) for t in turns]
    yy = [y for j in range(kj) for y in y0[j] + y1[j]]
    ja = judge.expect(ns, tt, yy, props)
    jm = monitor.expect(ns, tt, yy, props)
    A = ja["affect"].reshape(kj, 2, n)
    Mo = jm["monitor"].reshape(kj, 2, n)
    m0, m1 = mc_metrics(y0, turns), mc_metrics(y1, turns)
    D = {"affect": A[:, 1] - A[:, 0], "monitor": Mo[:, 1] - Mo[:, 0],
         "content": m1["content"] - m0["content"], "hygiene": m1["hygiene"] - m0["hygiene"],
         "length": np.stack([log_word_ratio(y1[j], y0[j]) for j in range(k)], 0)}
    # content is NaN exactly where the gold turn carries no information, for every reply of that context
    diffs = {key: v.mean(0) for key, v in D.items()}
    diffs_k1 = {key: v[0] for key, v in D.items()}
    ess = float(np.median(np.concatenate([ja["ess"], jm["ess"]])))
    level = {"affect_sft": float(np.mean(A[:, 0])), "monitor_sft": float(np.mean(Mo[:, 0])),
             "words_new": float(np.mean(m1["words"])), "words_sft": float(np.mean(m0["words"])), "k": k,
             "k_judge": kj}
    # variance decomposition: sd of the per-dialogue difference with k replies vs with one
    for key in ("content", "hygiene", "length"):
        a, b = diffs[key][np.isfinite(diffs[key])], diffs_k1[key][np.isfinite(diffs_k1[key])]
        if a.size > 2 and b.size > 2 and np.std(b) > 0:
            level[f"sd_ratio_{key}"] = float(np.std(a, ddof=1) / np.std(b, ddof=1))
    return diffs, diffs_k1, ess, level


def sample_exact(pol: Policy, ctx: Sequence[Turn], n: int, seed: int) -> Tuple[List[List[List[int]]], List[List[str]]]:
    """N exact samples per context: token ids exactly as sampled (EOS kept if sampled) and their decoding."""
    flat = pol.generate_ids([agent_prompt(t) for t in ctx for _ in range(n)], EXACT_GEN, seed)
    flat = [x if x else [pol.tok.eos_token_id] for x in flat]
    ids = [flat[c * n:(c + 1) * n] for c in range(len(ctx))]
    return ids, [[pol.decode(x) for x in row] for row in ids]


def sample_features(pol: Policy, ctx: Sequence[Turn], ids: List[List[List[int]]], texts: List[List[str]]
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """C x N matrices: information against the real agent turn (0 where it carries none: the constraint is
    vacuous there), log word count, and feasibility (well-formed AND terminated by a sampled EOS: a reply
    cut by the token limit is not a reply the deployed policy would give)."""
    eos = set(pol.eos_ids)
    I = np.asarray([[content_score(y, t.gold_response) for y in row] for t, row in zip(ctx, texts)], float)
    I = np.nan_to_num(I, nan=0.0)
    L = np.log(np.asarray([[n_words(y) + 1.0 for y in row] for row in texts]))
    F = np.asarray([[hygiene_ok(y) and bool(x) and x[-1] in eos for y, x in zip(row, xr)]
                    for row, xr in zip(texts, ids)], bool)
    return I, L, F


def judge_samples(texts: List[List[str]], ctx: Sequence[Turn], judge: Optional[Judge], ex: "Experiment", ns: str,
                  reward: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Affect, its SNIS standard error and ESS (C x N).  Every sample is a proposal of its context's pool."""
    C, N = len(texts), len(texts[0]) if texts else 0
    flat_t = [t for t in ctx for _ in range(N)]
    flat_y = [y for row in texts for y in row]
    if reward == "sentiment":
        a = np.asarray(ex.sentiment(flat_y), float)
        return a.reshape(C, N), np.zeros((C, N)), np.full((C, N), np.inf)
    res = judge.expect(ns, flat_t, flat_y, {t.uid: row for t, row in zip(ctx, texts)})
    return res["affect"].reshape(C, N), res["se_affect"].reshape(C, N), res["ess"].reshape(C, N)


def dpo_pairs(A: np.ndarray, ctx: Sequence[Turn], texts: List[List[str]], dup_jaccard: float = 0.9
              ) -> List[Tuple[str, str, str]]:
    """CASPI's baseline pairs on the same samples: best vs worst affect per context, near-duplicates dropped."""
    out = []
    for c, t in enumerate(ctx):
        i, j = int(np.argmax(A[c])), int(np.argmin(A[c]))
        if A[c, i] > A[c, j] and word_jaccard(texts[c][i], texts[c][j]) < dup_jaccard:
            out.append((agent_prompt(t), texts[c][i], texts[c][j]))
    return out


def deploy_probe(pol: Policy, turns: Sequence[Turn], k: int, seed: int, snap: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, np.ndarray]:
    """Step 1: per-context Monte Carlo means of the deployment-decoder objectives (the decoder the
    certification test uses) over k replies, on held-out RL probe contexts.  snap = frozen weights (SFT).
    The same seed for every policy gives common random numbers, so paired drifts have low variance."""
    m = mc_metrics(deploy_replies(pol, turns, k, seed, snap), turns)
    return {key: v.mean(0) for key, v in m.items()}


def probe_summary(p: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {k: float(np.nanmean(v)) if np.isfinite(v).any() else 0.0 for k, v in p.items()}


def loop_targets(drift: Dict[str, float], pc: TACITConfig) -> Tuple[float, float]:
    """Closed-loop moment targets from the cumulative deployment drift against SFT (integral feedback):
    length cancels the drift, information recovers any loss (and never asks for a loss).  Both are capped
    so that one noisy probe cannot demand an infeasible tilt."""
    len_t = float(np.clip(-pc.loop_gain * drift["log_words"], -pc.loop_clip_len, pc.loop_clip_len))
    info_t = float(np.clip(-pc.loop_gain * drift["content"], 0.0, pc.loop_clip_info))
    return len_t, info_t


def drift_stats(now: Dict[str, np.ndarray], sft: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float]]:
    """Paired per-context drift (pi - SFT) on the probe: (mean, standard error) per objective.  Probe
    contexts are one per dialogue, so contexts are the independent units."""
    out = {}
    for k in ("content", "log_words", "hygiene"):
        d = np.asarray(now[k], float) - np.asarray(sft[k], float)
        d = d[np.isfinite(d)]
        out[k] = (float(d.mean()), float(d.std(ddof=1) / math.sqrt(d.size))) if d.size >= 2 else (0.0, float("inf"))
    return out


def constraint_tube(stats: Dict[str, Tuple[float, float]], pc: TACITConfig) -> Tuple[bool, Dict[str, float]]:
    """Acceptance test of the line search (step 4): every cumulative drift must lie inside the TIGHTENED
    constraint set (tau x each certification margin) with one-sided confidence line_conf.  Returns the
    decision and the slack of each check (positive = inside)."""
    z, tau = normal_ppf(pc.line_conf), pc.line_tol
    (mc, sc), (ml, sl), (mh, sh) = stats["content"], stats["log_words"], stats["hygiene"]
    slack = {"content": (mc - z * sc) + tau * pc.margin_info,
             "hygiene": (mh - z * sh) + tau * pc.margin_hygiene,
             "length": tau * pc.length_equiv - (abs(ml) + z * sl)}
    return all(v > 0 for v in slack.values()), slack


def next_radius(eps: float, alpha: float, pc: TACITConfig, cap: float) -> float:
    """Trust-region radius update (Conn, Gould & Toint, 2000): grow after a full step, shrink after a
    backtracked or rejected one."""
    if alpha >= 1.0:
        return float(min(eps * pc.eps_up, cap))
    return float(max(pc.eps_min, eps * pc.eps_down))


def step_radii(flags: Dict[str, Any], F: np.ndarray, eps_t: float, pc: TACITConfig) -> Tuple[Any, float]:
    """Per-context E-step radii and the M-step KL stop of one round.
    adaptive: min(eps_t, best-of-n bound of the feasible count, cap); bon: TACIT-1; small: CAMPO.  The
    M-step stop is the median E-step radius, so the M-step never moves further than its target asks."""
    n = F.sum(1)
    if flags["step"] == "adaptive":
        eps = np.minimum(np.minimum(bon_kl_budget(n), pc.epsilon_cap), eps_t)
        return eps, (float(np.median(eps[n > 1])) if (n > 1).any() else float(eps_t))
    if flags["step"] == "bon":
        eps = np.minimum(bon_kl_budget(n), pc.epsilon_cap)
        return eps, (float(np.median(eps[n > 1])) if (n > 1).any() else pc.small_epsilon)
    return pc.small_epsilon, pc.small_epsilon


def train_arm(cfg: Config, arm: str, seed: int) -> Dict[str, Any]:
    """Train one arm (see ARM_FLAGS); save the returned policy as policy_<arm>_s<seed> and, for arms with
    selection, the selected candidate as cand_<arm>_s<seed> (SFT when there is none)."""
    ex = Experiment(cfg, f"4_train_{arm}_{seed}")
    lg, pc, flags = ex.logger, cfg.tacit, ARM_FLAGS[arm]
    tag = f"{arm}_s{seed}"
    seed_everything(seed)
    pol = ex.policy("sft_policy")
    ref = pol.snapshot()                             # SFT: the anchor of every test and the DPO reference
    sim = ex.policy("simulator")
    judge, _ = ex.judges(sim, tag, pc.pool_size)
    g_judge, g_monitor = ex.judges(sim, f"{tag}_guard", pc.guard_pool_size)
    per_round = pc.contexts_per_round + pc.kl_holdout
    # step 0: every training context comes from the held-out RL dialogues; the probe owns its dialogues
    probe_ctx = select_turns(ex.turns, "rl", pc.probe_contexts, seed + 17, one_per_dialogue=True)
    train_ctx = select_turns(ex.turns, "rl", pc.rounds * per_round, seed,
                             exclude_dialogues={t.dialogue_id for t in probe_ctx})
    sel = select_turns(ex.turns, "valid", pc.select_contexts, cfg.seed + 101, require_next=True, one_per_dialogue=True)
    safe = select_turns(ex.turns, "valid", pc.safety_contexts, cfg.seed + 202, require_next=True,
                        one_per_dialogue=True, exclude_dialogues={t.dialogue_id for t in sel})
    k_mc, k_j = (pc.mc_k, pc.mc_k_judge) if flags["mc"] else (1, 1)
    mpo = flags["optimizer"] == "mpo"
    lg.info("%s | flags %s | %d RL training contexts | probe %d x %d replies | selection %d, safety %d "
            "(dialogue-disjoint) | Monte Carlo replies per test context %d (judge %d)", tag, flags, len(train_ctx),
            len(probe_ctx), pc.probe_k, len(sel), len(safe), k_mc, k_j)
    probe_seed = seed * 31 + 7
    probe_sft = deploy_probe(pol, probe_ctx, pc.probe_k, probe_seed, ref) if mpo else {}
    probe_now = dict(probe_sft)
    eps_t = pc.eps_init
    hist: List[Dict[str, Any]] = []
    checkpoints: Dict[int, Dict[str, Any]] = {}
    offline_pairs: List[Tuple[str, str, str]] = []
    for rd in range(1, pc.rounds + 1):
        block = train_ctx[(rd - 1) * per_round: rd * per_round]
        if len(block) < pc.kl_holdout + 8:
            lg.warning("%s | round %d skipped: only %d RL contexts left", tag, rd, len(block))
            break
        probe, ctx = block[:pc.kl_holdout], block[pc.kl_holdout:]
        rseed = seed * 1000 + 10 * rd
        if flags["offline"]:
            with pol.frozen(ref):
                ids, texts = sample_exact(pol, ctx, pc.n_samples, rseed)
        else:
            ids, texts = sample_exact(pol, ctx, pc.n_samples, rseed)
        A, SE, ESS = judge_samples(texts, ctx, judge, ex, f"r{rd}", flags["reward"])
        I, L, F = sample_features(pol, ctx, ids, texts)
        rec: Dict[str, Any] = {"round": rd, "contexts": len(ctx), "samples": int(A.size),
                               "affect_sample_mean": float(A.mean()), "se_median": float(np.median(SE)),
                               "se_within_context_sd_median": float(np.median(SE.std(1))),
                               "ess_median": float(np.median(ESS[np.isfinite(ESS)])) if np.isfinite(ESS).any()
                               else float("nan"), "hygiene_eos_share": float(F.mean()),
                               "words_sample_mean": float(np.exp(L).mean() - 1.0)}
        changed = True
        if flags["optimizer"] == "dpo":
            pairs = dpo_pairs(A, ctx, texts)
            rec["pairs"] = len(pairs)
            if flags["offline"]:
                offline_pairs += pairs
                hist.append(rec)
                lg.info("%s | round %d | %d pairs collected under SFT (offline)", tag, rd, len(pairs))
                continue
            upd = pol.dpo_update(pairs, [1.0] * len(pairs), ref, pc.dpo_beta, pc.dpo_lr, pc.epochs, 4,
                                 pc.dpo_nll_coef, seed * 100 + rd) if pairs else {}
            rec["update"] = upd
            lg.info("%s | round %d | %d pairs | %s", tag, rd, len(pairs),
                    {k: round(v, 3) for k, v in upd.items() if isinstance(v, float)})
        else:
            # 1. cumulative deployment drift of the current policy against SFT (held-out probe, k replies)
            stats = drift_stats(probe_now, probe_sft)
            drift = {k: v[0] for k, v in stats.items()}
            len_t, info_t = loop_targets(drift, pc) if flags["closed_loop"] else (0.0, 0.0)
            # 2-3. closed-loop I-projection and weighted M-step on the round's radius
            Rw = A - pc.kappa * SE if flags["pessimism"] else A
            eps, kl_max = step_radii(flags, F, eps_t, pc)
            est = estep(Rw, I, L, F, eps, True, pc.lambda_max, pc.mu_max, len_target=len_t, info_target=info_t)
            es = estep_summary(est, Rw, F)
            Q = est["weights"].copy()
            Q[Q < pc.min_weight] = 0.0
            Q = np.divide(Q, Q.sum(1, keepdims=True), out=np.zeros_like(Q), where=Q.sum(1, keepdims=True) > 0)
            prompts, conts, w = [], [], []
            for c, t in enumerate(ctx):
                for k in np.flatnonzero(Q[c] > 0):
                    prompts.append(agent_prompt(t))
                    conts.append(ids[c][k])
                    w.append(float(Q[c, k]))
            es.update({"target_affect": float(np.sum(Q * A) / max(Q.sum(), EPS)),
                       "target_words": float(np.sum(Q * (np.exp(L) - 1.0)) / max(Q.sum(), EPS)),
                       "m_step_targets": len(w), "kl_budget_median": float(np.median(np.atleast_1d(eps))),
                       "kl_max": kl_max})
            before = pol.snapshot()
            p_ids, _ = sample_exact(pol, probe, pc.n_samples, rseed + 5)
            upd = pol.fit_weighted(prompts, conts, w, [agent_prompt(t) for t in probe for _ in range(pc.n_samples)],
                                   [x for row in p_ids for x in row], pc.lr, pc.epochs, pc.batch, kl_max,
                                   pc.kl_check_every, seed * 100 + rd) if w else {"steps": 0}
            # 4. certified backtracking line search on the deployed constraints (tightened by line_tol)
            tried: List[Dict[str, Any]] = []
            alpha = 1.0 if w else 0.0
            if flags["line_search"] and w:
                after = pol.snapshot()
                alpha = 0.0
                for a in pc.line_alphas:
                    pol.interpolate(before, after, a)
                    P = deploy_probe(pol, probe_ctx, pc.probe_k, probe_seed)
                    ok, slack = constraint_tube(drift_stats(P, probe_sft), pc)
                    tried.append({"alpha": a, "accept": ok, "slack": {k: round(v, 4) for k, v in slack.items()}})
                    if ok:
                        alpha, probe_now = a, P
                        break
                if alpha == 0.0:
                    pol.restore(before)               # keep theta_t; probe_now still describes it
                del after
            elif w:
                probe_now = deploy_probe(pol, probe_ctx, pc.probe_k, probe_seed)
            changed = alpha > 0.0
            del before
            # 5. trust-region radius for the next round
            eps_next = next_radius(eps_t, alpha, pc, pc.epsilon_cap) if flags["step"] == "adaptive" else eps_t
            after_stats = drift_stats(probe_now, probe_sft)
            rec.update({"estep": es, "update": upd, "drift": drift, "drift_se": {k: v[1] for k, v in stats.items()},
                        "probe": probe_summary(probe_now), "probe_sft": probe_summary(probe_sft),
                        "drift_after": {k: v[0] for k, v in after_stats.items()},
                        "line_search": {"alpha": alpha, "tried": tried}, "radius": eps_t, "radius_next": eps_next})
            lg.info("%s | round %d | probe drift vs SFT: log-len %+.3f content %+.3f hygiene %+.3f -> targets len %+.3f "
                    "info %+.3f | E-step: feasible %.2f, radius %.2f (median %.2f), lambda %.3f, mu %+.3f (%s), "
                    "achieved len %+.3f info %+.3f, reward gain %+.4f | M-step: %d steps, KL %.3f%s | line search "
                    "alpha %.2f %s | drift after: log-len %+.3f content %+.3f | next radius %.2f", tag, rd,
                    drift["log_words"], drift["content"], drift["hygiene"], len_t, info_t, es["feasible_share"], eps_t,
                    es["kl_budget_median"], es["lambda"], es["mu"], es["status"], es["g_len"], es["g_info"],
                    es["reward_gain_vs_feasible_mean"], upd.get("steps", 0), upd.get("kl_estimate", float("nan")),
                    " (stopped at kl_max)" if upd.get("kl_stop") else "", alpha,
                    [(x["alpha"], x["accept"]) for x in tried], rec["drift_after"]["log_words"],
                    rec["drift_after"]["content"], eps_next)
            eps_t = eps_next
        if flags["select"] and changed:
            diffs, _, ess, level = paired_dev_check(pol, ref, g_judge, g_monitor, sel, f"sel_r{rd}", seed * 7 + 1,
                                                    k_mc, k_j)
            cl = [t.dialogue_id for t in sel]
            dec = vector_test(diffs, cl, pc.select_alpha, pc, ess, pc.guard_pool_size)
            pp = predict_pass(diffs, cl, len(safe), pc.alpha, pc, pc.predict_inflation)
            ok = dec["accept"] and (pp["pass"] or not flags["predict"])
            rec.update({"selection": dec, "selection_levels": level, "predicted_pass": pp, "candidate": ok})
            if ok:
                score = pp["predicted_affect_lb"] if flags["predict"] else dec["lower_bounds"]["affect"]
                checkpoints[rd] = {"snap": pol.snapshot(), "score": score}
            lg.info("%s | round %d selection | affect %+.4f (LB %+.4f) monitor %+.4f content %+.3f (LB %+.3f) hygiene "
                    "%+.3f log-length %+.3f | ESS %.1f | predicted safety slack %s | %s", tag, rd,
                    dec["means"]["affect"], dec["lower_bounds"]["affect"], dec["means"]["monitor"],
                    dec["means"]["content"], dec["lower_bounds"]["content"], dec["means"]["hygiene"],
                    dec["means"]["length"], ess, {k: round(v, 4) for k, v in pp.get("slack", {}).items()},
                    "candidate" if ok else "not a candidate: " + str([k for k, v in dec["checks"].items() if not v]
                                                                     + ([] if pp["pass"] else ["predicted_pass"])))
        elif flags["select"]:
            rec["selection_skipped"] = "line search rejected the step: the policy is unchanged"
        hist.append(rec)
    if flags["optimizer"] == "dpo" and flags["offline"] and offline_pairs:
        hist.append({"offline_update": pol.dpo_update(offline_pairs, [1.0] * len(offline_pairs), ref, pc.dpo_beta,
                                                      pc.dpo_lr, pc.epochs, 4, pc.dpo_nll_coef, seed * 100)})
    cert: Dict[str, Any] = {"run": False}
    chosen, returned_sft = None, False
    if flags["select"]:
        chosen = max(checkpoints, key=lambda r: checkpoints[r]["score"]) if checkpoints else None
        pol.restore(checkpoints[chosen]["snap"] if chosen is not None else ref)
        pol.save_adapter(cfg.out_dir / f"cand_{tag}")      # per-protocol candidate (SFT if there is none)
        if chosen is not None:
            diffs, diffs1, ess, level = paired_dev_check(pol, ref, g_judge, g_monitor, safe, "certify", seed * 7 + 2,
                                                         k_mc, k_j)
            cl = [t.dialogue_id for t in safe]
            dec = vector_test(diffs, cl, pc.alpha, pc, ess, pc.guard_pool_size)
            dec1 = vector_test(diffs1, cl, pc.alpha, pc, ess, pc.guard_pool_size)
            cert = {"run": True, **dec, "levels": level, "n": len(safe), "round": chosen, "k": k_mc,
                    "k1": {key: dec1[key] for key in ("accept", "checks", "lower_bounds", "length_upper", "means")}}
            lg.info("%s | CERTIFICATION of round %d (alpha %.3f, %s bound, %d contexts x %d replies) | %s | failed: %s | "
                    "content LB %+.4f (one reply: %+.4f, %s) | sd ratio k vs 1 %s | power at zero difference %s | "
                    "dialogues for 80%% power %s", tag, chosen, pc.alpha, pc.bound, len(safe), k_mc,
                    "PASS" if dec["accept"] else "FAIL -> returning SFT", [k for k, v in dec["checks"].items() if not v],
                    dec["lower_bounds"]["content"], dec1["lower_bounds"]["content"],
                    "pass" if dec1["accept"] else "fail",
                    {k: round(v, 3) for k, v in level.items() if k.startswith("sd_ratio")},
                    {k: round(v, 2) for k, v in dec["power_at_zero_difference"].items()},
                    dec["dialogues_for_80pct_power"])
            returned_sft = not dec["accept"]
        else:
            returned_sft = True
        if returned_sft:
            pol.restore(ref)
            lg.warning("%s | the returned policy IS the SFT policy (no selection candidate, or certification failed); "
                       "this is the algorithm working as designed, report it", tag)
    pol.save_adapter(cfg.out_dir / f"policy_{tag}")
    res = {"arm": arm, "seed": seed, "flags": flags, "rounds": hist, "selected_round": chosen,
           "accepted_rounds": len(checkpoints), "certification": cert, "returned_sft": returned_sft,
           "mc_k": k_mc, "config": asdict(pc), "version": VERSION}
    dump_json(res, cfg.out_dir / f"train_{tag}.json")
    return res


def stage_preregister(cfg: Config) -> Dict[str, Any]:
    """Write the analysis plan before any policy is trained.  Every margin, level and split seed of the tests
    is fixed here; the claims stage reads the margins from this file, never from data."""
    ex = Experiment(cfg, "3_preregister")
    f = cfg.out_dir / "preregistration.json"
    plan = {"version": VERSION, "tacit": asdict(cfg.tacit), "equiv_margin_outcome": cfg.equiv_margin_outcome,
            "seeds": list(cfg.seeds), "arms": list(cfg.arms), "primary": f"{PRIMARY}_vs_sft:outcome",
            "co_primary": "H2 constraints vs sft (information, hygiene, monitor non-inferior; length TOST)",
            "h3": "H3a line search, H3b adaptive radius (vs fixed small and fixed best-of-n), H3c Monte Carlo "
                  "certification (bound width, paired McNemar), H3d closed loop, H3e memorisation gap; contribution "
                  "= significantly better outcome with constraints met, OR a higher certification rate (one-sided "
                  "Fisher exact, alpha 0.05), OR the ablation's CANDIDATE violates a constraint that tacit2's meets; "
                  "an ablation that returned SFT in every seed is compared through its candidate",
            "h4": "H4a tacit2 non-inferior to sft_bon (amortisation); H4b tacit2_bon > sft_bon at the same N "
                  "(compute-matched); H4c tacit2 > tacit1 and > campo; H4d Pareto vs DPO baselines",
            "bon_n": cfg.bon_n, "rl_holdout_frac": cfg.rl_holdout_frac,
            "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    if f.exists():
        old = load_json(f)
        changed = [k for k in ("tacit", "equiv_margin_outcome", "primary", "bon_n", "rl_holdout_frac", "seeds")
                   if old.get(k) != plan.get(k)]
        if changed:
            raise RuntimeError(f"preregistration.json exists and differs in {changed}: delete it deliberately and "
                               "report the change, or restore the registered values")
        return old
    dump_json(plan, f)
    ex.logger.info("preregistration written to %s", f)
    return plan
# ==========================================================================================================
# evaluation
# ==========================================================================================================

def bon_base(arm: str) -> Optional[str]:
    """'<base>_bon' -> '<base>' (best-of-N over samples of the base policy), else None."""
    return arm[:-4] if arm.endswith("_bon") and arm not in ARM_FLAGS else None


def adapter_of(cfg: Config, arm: str, sd: int) -> Path:
    """Adapter behind an evaluated arm: SFT, a returned policy, or a selected candidate ("<base>_cand")."""
    if arm == "sft":
        return cfg.out_dir / "sft_policy"
    if arm.endswith("_cand"):
        return cfg.out_dir / f"cand_{arm[:-5]}_s{sd}"
    return cfg.out_dir / f"policy_{arm}_s{sd}"


def stage_eval(cfg: Config) -> Dict[str, Any]:
    """Test-split evaluation of every arm under every seed.  Generation uses the same seeds for all arms
    (common random numbers).  Scoring is symmetric: per (seed, context) every arm's reply is a proposal of
    ONE shared pool, under both the training judge and the monitor.  "<base>_bon" arms draw bon_n samples of
    the base policy and keep the one the judge prefers on an INDEPENDENT pool (no winner's curse), so
    sft_bon and tacit2_bon are compute-matched; their test-time judge cost is logged.  "<base>_cand" arms
    are selected candidates before the certification fallback (per-protocol analysis, TACIT-1 E4)."""
    ex = Experiment(cfg, "5_eval")
    te = select_turns(ex.turns, "test", cfg.eval_turns, cfg.seed, require_next=True)
    prompts = [agent_prompt(t) for t in te]
    gen = GenConfig()
    pol = ex.policy("sft_policy")
    replies: Dict[int, Dict[str, List[str]]] = {}
    bon_samples: Dict[Tuple[int, str], List[List[str]]] = {}

    for sd in cfg.seeds:
        replies[sd] = {}
        for arm in cfg.arms:
            base = bon_base(arm) or arm
            path = adapter_of(cfg, base, sd)
            if not path.exists():
                ex.logger.warning("eval: %s seed %d skipped, %s missing", arm, sd, path)
                continue
            pol.load_adapter(path)
            if bon_base(arm):
                flat = pol.generate_text([p for p in prompts for _ in range(cfg.bon_n)], gen, 9500 + sd)
                bon_samples[(sd, arm)] = [flat[k * cfg.bon_n:(k + 1) * cfg.bon_n] for k in range(len(te))]
            else:
                replies[sd][arm] = pol.generate_text(prompts, gen, 9000 + sd)
    del pol
    release_gpu()
    sim = ex.policy("simulator")
    judge, monitor = ex.judges(sim, "eval", cfg.eval_pool_size)
    for (sd, arm), cands in bon_samples.items():
        flat_t = [t for t in te for _ in range(cfg.bon_n)]
        a = judge.expect(f"{arm}_select_s{sd}", flat_t, [y for c in cands for y in c],
                         {t.uid: c for t, c in zip(te, cands)})["affect"].reshape(len(te), cfg.bon_n)
        ok = np.asarray([[hygiene_ok(y) for y in c] for c in cands])
        a = np.where(ok | ~ok.any(1, keepdims=True), a, -np.inf)
        replies[sd][arm] = [c[int(np.argmax(r))] for c, r in zip(cands, a)]
        ex.logger.info("eval %s seed %d | test-time cost per context: %d policy samples + %d simulated customer "
                       "replies scored by the judge", arm, sd, cfg.bon_n, max(cfg.eval_pool_size, cfg.bon_n + 1))
    for sd in cfg.seeds:
        arms = list(replies[sd])
        if not arms:
            continue
        props = {t.uid: [replies[sd][a][k] for a in arms] for k, t in enumerate(te)}
        for a in arms:
            ja = judge.expect(f"eval_s{sd}", te, replies[sd][a], props)
            jm = monitor.expect(f"eval_s{sd}", te, replies[sd][a], props)
            rm = reply_metrics(replies[sd][a], te)
            agent_sent = ex.sentiment(replies[sd][a])
            rows = [{"uid": t.uid, "dialogue_id": t.dialogue_id, "arm": a, "seed": sd, "response": r,
                     "outcome": float(ja["affect"][k]), "monitor": float(jm["monitor"][k]),
                     "ess": float(ja["ess"][k]), "agent_sentiment": float(agent_sent[k]),
                     "test_time_samples": cfg.bon_n if bon_base(a) else 1,
                     **{m: float(v[k]) for m, v in rm.items()}, "human_valence": t.human_valence}
                    for k, (t, r) in enumerate(zip(te, replies[sd][a]))]
            dump_json(rows, cfg.out_dir / f"eval_{a}_{sd}.json")
            ex.logger.info("eval %-17s seed %d | outcome %.4f monitor %.4f | content %.3f hygiene %.3f words %.1f | "
                           "median ESS %.1f", a, sd, np.mean(ja["affect"]), np.mean(jm["monitor"]),
                           np.nanmean(rm["content"]), rm["hygiene"].mean(), rm["words"].mean(), np.median(ja["ess"]))
    return {"n_contexts": len(te)}


def stage_eval_external(cfg: Config) -> Dict[str, Any]:
    """Re-score every evaluated reply with an evaluator that shares nothing with training: an out-of-family
    customer LLM (no adapter, no EmoWOZ fine-tuning) read by an emotion classifier that no stage uses, with the
    same symmetric shared-pool protocol as `eval`."""
    ex = Experiment(cfg, "6_eval_external")
    rows = load_rows(cfg, "eval")
    turn_of = {t.uid: t for t in ex.turns}
    ext = Policy(cfg.external_model, cfg, ex.logger, lora=False)
    emo = ValenceClassifier(cfg.external_emotion_model, cfg, ex.logger)
    judge = Judge(ext, {"ext": emo}, chat_customer_prompt, cfg.out_dir / "pools" / "external.jsonl",
                  cfg.external_pool_size, True, cfg.sim_max_new_tokens, ex.logger, "external")
    for sd in cfg.seeds:
        arms = [a for a in rows if sd in rows[a]]
        if not arms:
            continue
        uids = [r["uid"] for r in rows[arms[0]][sd]]
        te = [turn_of[u] for u in uids]
        resp = {a: {r["uid"]: r["response"] for r in rows[a][sd]} for a in arms}
        props = {u: [resp[a][u] for a in arms] for u in uids}
        for a in arms:
            res = judge.expect(f"ext_s{sd}", te, [r["response"] for r in rows[a][sd]], props)
            out = [{**r, "outcome": float(v), "sim_outcome": r["outcome"]} for r, v in zip(rows[a][sd], res["ext"])]
            dump_json(out, cfg.out_dir / f"ext_{a}_{sd}.json")
            ex.logger.info("external %-17s seed %d | outcome %.4f | rank agreement with the training judge (rows) "
                           "%+.3f", a, sd, np.mean(res["ext"]), spearman(res["ext"], [r["outcome"] for r in rows[a][sd]]))
    return {"seeds": list(cfg.seeds)}


def stage_cross_eval(cfg: Config) -> Dict[str, Any]:
    """Within-context agreement of the training judge and the external evaluator on K SFT replies per context
    (chance = 0.5), plus between-context anchors on context means.  The samples are kept for the human
    study's validity items.  Automatic evidence only."""
    ex = Experiment(cfg, "6_cross_eval")
    te = select_turns(ex.turns, "test", cfg.cross_eval_contexts, cfg.seed + 7, require_next=True, one_per_dialogue=True)
    K = max(2, cfg.cross_eval_k)
    flat_t = [t for t in te for _ in range(K)]
    pol = ex.policy("sft_policy")
    ys = pol.generate_text([agent_prompt(t) for t in flat_t], GenConfig(), cfg.seed + 77)
    del pol
    release_gpu()
    props = {t.uid: ys[k * K:(k + 1) * K] for k, t in enumerate(te)}
    sim = ex.policy("simulator")
    judge, _ = ex.judges(sim, "cross", cfg.eval_pool_size)
    a_sim = judge.expect("cross", flat_t, ys, props)["affect"].reshape(len(te), K)
    del sim, judge
    release_gpu()
    ext = Policy(cfg.external_model, cfg, ex.logger, lora=False)
    ej = Judge(ext, {"ext": ValenceClassifier(cfg.external_emotion_model, cfg, ex.logger)}, chat_customer_prompt,
               cfg.out_dir / "pools" / "cross_external.jsonl", cfg.external_pool_size, True, cfg.sim_max_new_tokens,
               ex.logger, "external")
    a_ext = ej.expect("cross", flat_t, ys, props)["ext"].reshape(len(te), K)
    iu = np.triu_indices(K, 1)
    per, cl = [], []
    for c, t in enumerate(te):
        x = (a_sim[c][:, None] - a_sim[c][None, :])[iu]
        y = (a_ext[c][:, None] - a_ext[c][None, :])[iu]
        m = np.abs(x) > 1e-9
        if m.any():
            per.append(float(np.mean(x[m] * y[m] > 0)))
            cl.append(t.dialogue_id)
    pa, ca = np.asarray(per), np.asarray(cl, dtype=object)
    ci = cluster_bootstrap_ci(lambda ix: float(np.mean(pa[ix])), ca, cfg.n_boot, cfg.seed)
    sf = sign_flip_test(pa - 0.5, ca, cfg.n_signflip, cfg.seed + 1)
    hv = np.asarray([t.human_valence if t.human_valence is not None else np.nan for t in te], float)
    out = {"n_contexts": len(per), "K": K, "agreement": ci, "p_vs_chance": sf["p"], "p_text": sf["text"],
           "between_context_rho_simulator": spearman(a_sim.mean(1), hv),
           "between_context_rho_external": spearman(a_ext.mean(1), hv)}
    dump_json(out, cfg.out_dir / "cross_eval.json")
    dump_json([{"uid": t.uid, "dialogue_id": t.dialogue_id, "replies": ys[k * K:(k + 1) * K],
                "sim": a_sim[k].tolist()} for k, t in enumerate(te)], cfg.out_dir / "cross_eval_samples.json")
    ex.logger.info("cross-evaluator within-context agreement %.3f CI[%.3f,%.3f] (chance 0.5, p %s) over %d contexts | "
                   "between-context rho: simulator %+.3f, external %+.3f", *ci, sf["text"], len(per),
                   out["between_context_rho_simulator"], out["between_context_rho_external"])
    return out


# ==========================================================================================================
# report, length analysis, claims
# ==========================================================================================================

REPORT_METRICS = ("outcome", "monitor", "content", "info_recall", "slot_f1", "hygiene", "words", "log_words",
                  "agent_sentiment")


def load_rows(cfg: Config, prefix: str) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    rows: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    for arm in cfg.arms:
        for sd in cfg.seeds:
            f = cfg.out_dir / f"{prefix}_{arm}_{sd}.json"
            if f.exists():
                rows.setdefault(arm, {})[sd] = load_json(f)
    if not rows:
        raise FileNotFoundError(f"no {prefix}_*.json rows in {cfg.out}")
    return rows


def paired_contrast(rows_a: Dict[int, List[Dict[str, Any]]], rows_b: Dict[int, List[Dict[str, Any]]], metric: str,
                    seed: int, n_boot: int, n_mc: int, seeds: Optional[Sequence[int]] = None,
                    equiv: Optional[float] = None, noninf: Optional[float] = None) -> Dict[str, Any]:
    """Arm minus comparator, paired within (seed, context), pooled over seeds, DIALOGUES as clusters.
    equiv: two-sided TOST margin (90% CI inside +-equiv); noninf: one-sided margin (lower CI90 > -noninf)."""
    d, cl = [], []
    for sd, ra in rows_a.items():
        if (seeds is not None and sd not in seeds) or sd not in rows_b:
            continue
        ib = {r["uid"]: r for r in rows_b[sd]}
        for r in ra:
            q = ib.get(r["uid"])
            if q is not None and r.get(metric) is not None and q.get(metric) is not None and \
                    math.isfinite(float(r[metric])) and math.isfinite(float(q[metric])):
                d.append(float(r[metric]) - float(q[metric]))
                cl.append(str(r["dialogue_id"]))
    if len(d) < 20:
        return {"n": len(d)}
    d_, cl_ = np.asarray(d), np.asarray(cl, dtype=object)
    pt, lo, hi = cluster_bootstrap_ci(lambda ix: float(np.mean(d_[ix])), cl_, n_boot, seed)
    _, lo90, hi90 = cluster_bootstrap_ci(lambda ix: float(np.mean(d_[ix])), cl_, n_boot, seed + 7, conf=0.90)
    sf = sign_flip_test(d_, cl_, n_mc, seed + 1)
    se = cluster_robust_se(d_, cl_)
    sd_d = float(np.std(d_, ddof=1))
    out = {"n": len(d), "n_clusters": int(np.unique(cl_).size), "delta": pt, "ci95": [lo, hi], "ci90": [lo90, hi90],
           "p_value": sf["p"], "p_text": sf["text"], "se_cluster": se,
           "mde80": float((normal_ppf(0.975) + normal_ppf(0.8)) * se) if math.isfinite(se) else float("nan"),
           "cohens_dz": float(pt / sd_d) if sd_d > 0 else float("nan"),
           "paired_dominance": float(np.mean(d_ > 0) - np.mean(d_ < 0))}
    if equiv is not None:
        out.update({"equiv_margin": equiv, "equivalent": bool(lo90 > -equiv and hi90 < equiv)})
    if noninf is not None:
        out.update({"noninf_margin": noninf, "noninferior": bool(lo90 > -noninf)})
    return out


def comparison_plan(arms: Sequence[str]) -> List[Tuple[str, str]]:
    """Every arm vs SFT; the primary vs every other arm; best-of-N compute-matched (H4b); and the primary's
    candidate vs every ablation's candidate (per-protocol contributions, H3)."""
    plan = [(a, "sft") for a in arms if a != "sft"]
    if PRIMARY in arms:
        plan += [(PRIMARY, a) for a in arms if a not in ("sft", PRIMARY)]
    if f"{PRIMARY}_bon" in arms and "sft_bon" in arms:
        plan.append((f"{PRIMARY}_bon", "sft_bon"))     # H4b: compute-matched
    pc = f"{PRIMARY}_cand"
    if pc in arms:
        plan += [(pc, a) for a in arms if a.endswith("_cand") and a != pc]
    return list(dict.fromkeys(plan))


def stage_report(cfg: Config, prefix: str = "eval") -> Dict[str, Any]:
    """Arm means and every planned contrast; Holm within each metric family.  outcome gets a TOST margin
    (for null claims such as "the guard costs no affect"); content/info/monitor get non-inferiority margins."""
    ex = Experiment(cfg, f"7_report_{prefix}")
    rows = load_rows(cfg, prefix)
    pre = load_json(cfg.out_dir / "preregistration.json") if (cfg.out_dir / "preregistration.json").exists() else None
    pcm = pre["tacit"] if pre else asdict(cfg.tacit)
    eq = pre["equiv_margin_outcome"] if pre else cfg.equiv_margin_outcome
    metrics = [m for m in REPORT_METRICS if any(m in rs[0] for per in rows.values() for rs in per.values() if rs)]
    summary: Dict[str, Any] = {"evaluator": prefix, "arms": {}, "contrasts": {}, "equiv_margin_outcome": eq}
    for arm, per in rows.items():
        allr = [r for rs in per.values() for r in rs]
        summary["arms"][arm] = {m: float(np.nanmean([r[m] for r in allr])) for m in metrics}
        summary["arms"][arm]["n_rows"] = len(allr)
    ni = {"content": pcm["margin_info"], "info_recall": pcm["margin_info"], "monitor": pcm["margin_monitor"],
          "hygiene": pcm["margin_hygiene"]}
    tost = {"outcome": eq, "log_words": pcm["length_equiv"]}
    for a, b in comparison_plan([x for x in cfg.arms if x in rows]):
        for m in metrics:
            c = paired_contrast(rows[a], rows[b], m, cfg.seed, cfg.n_boot, cfg.n_signflip,
                                equiv=tost.get(m), noninf=ni.get(m))
            if c.get("n", 0) >= 20:
                summary["contrasts"][f"{a}_vs_{b}:{m}"] = {**c, "arm": a, "comparator": b, "metric": m}
    for m in metrics:
        keys = [k for k, v in summary["contrasts"].items() if v["metric"] == m]
        for k, p in zip(keys, holm([summary["contrasts"][k]["p_value"] for k in keys])):
            summary["contrasts"][k]["p_holm"] = p
    dump_json(summary, cfg.out_dir / f"report_{prefix}.json")
    ex.logger.info("%-18s %9s %9s %8s %8s %8s", "arm", "outcome", "monitor", "content", "hygiene", "words")
    for arm, v in summary["arms"].items():
        ex.logger.info("%-18s %9.4f %9.4f %8.3f %8.3f %8.1f", arm, v.get("outcome", np.nan), v.get("monitor", np.nan),
                       v.get("content", np.nan), v.get("hygiene", np.nan), v.get("words", np.nan))
    for k, v in summary["contrasts"].items():
        if v["metric"] in ("outcome", "content", "monitor", "hygiene", "log_words"):
            ex.logger.info("%-40s %+.4f CI95[%+.4f,%+.4f] p %s holm %.4f | MDE80 %.4f%s", k, v["delta"], *v["ci95"],
                           v["p_text"], v["p_holm"], v["mde80"],
                           f" | non-inferior: {v['noninferior']}" if "noninferior" in v else
                           (f" | TOST-equivalent: {v['equivalent']}" if "equivalent" in v else ""))
    return summary


def stage_length(cfg: Config) -> Dict[str, Any]:
    """Is the outcome gain over SFT explained by length?  Paired regression d_outcome = a + b d_words (a is the
    gain at equal length; descriptive, since length is post-treatment) and the length-matched stratum."""
    ex = Experiment(cfg, "7_length")
    rows = load_rows(cfg, "eval")
    out: Dict[str, Any] = {}
    for arm in [a for a in rows if a != "sft" and "sft" in rows]:
        dy, dw, cl = [], [], []
        for sd, ra in rows[arm].items():
            ib = {r["uid"]: r for r in rows["sft"].get(sd, [])}
            for r in ra:
                if r["uid"] in ib:
                    dy.append(r["outcome"] - ib[r["uid"]]["outcome"])
                    dw.append(r["words"] - ib[r["uid"]]["words"])
                    cl.append(r["dialogue_id"])
        if len(dy) < 30:
            continue
        dy_, dw_, cl_ = np.asarray(dy), np.asarray(dw), np.asarray(cl, dtype=object)

        def intercept(ix):
            return float(np.linalg.lstsq(np.column_stack([np.ones(ix.size), dw_[ix]]), dy_[ix], rcond=None)[0][0])
        m = np.abs(dw_) <= 2
        res = {"raw": float(dy_.mean()), "mean_d_words": float(dw_.mean()),
               "equal_length": cluster_bootstrap_ci(intercept, cl_, cfg.n_boot, cfg.seed),
               "matched_n": int(m.sum())}
        if m.sum() >= 30:
            dm, cm = dy_[m], cl_[m]
            res["matched"] = cluster_bootstrap_ci(lambda ix: float(np.mean(dm[ix])), cm, cfg.n_boot, cfg.seed + 1)
        out[arm] = res
        ex.logger.info("%-18s raw %+.4f (d words %+.1f) | at equal length %+.4f CI[%+.4f,%+.4f] | |dw|<=2: n=%d %s",
                       arm, res["raw"], res["mean_d_words"], *res["equal_length"], res["matched_n"],
                       "" if "matched" not in res else "%+.4f CI[%+.4f,%+.4f]" % tuple(res["matched"]))
    dump_json(out, cfg.out_dir / "length_analysis.json")
    return out
def stage_human_export(cfg: Config) -> Dict[str, Any]:
    """Blinded, randomised pairwise packet: arm pairs (same context, seed 1), within-context judge-validity
    items (two SFT replies to one context from cross-eval) and attention checks.  The key is kept apart."""
    ex = Experiment(cfg, "8_human_export")
    rng = np.random.default_rng(cfg.seed + 2024)
    d = cfg.out_dir / "human_study"
    d.mkdir(parents=True, exist_ok=True)
    turn_of = {t.uid: t for t in ex.turns}
    rows = load_rows(cfg, "eval")
    sd0 = cfg.seeds[0]
    items: List[Dict[str, Any]] = []
    for comp in cfg.human_comparisons:
        a, b = comp.split(":")
        if sd0 not in rows.get(a, {}) or sd0 not in rows.get(b, {}):
            ex.logger.warning("comparison %s skipped: eval rows missing", comp)
            continue
        ia, ib = {r["uid"]: r for r in rows[a][sd0]}, {r["uid"]: r for r in rows[b][sd0]}
        by_d: Dict[str, List[str]] = {}
        for u in sorted(set(ia) & set(ib)):
            if norm_text(ia[u]["response"]) != norm_text(ib[u]["response"]):
                by_d.setdefault(ia[u]["dialogue_id"], []).append(u)
        for k in list(rng.permutation(sorted(by_d)))[:cfg.human_contexts]:
            u = by_d[k][int(rng.integers(len(by_d[k])))]
            items.append({"kind": "arm_pair", "comparison": comp, "uid": u, "dialogue_id": k,
                          "A": (a, ia[u]["response"], ia[u]["outcome"]), "B": (b, ib[u]["response"], ib[u]["outcome"])})
    cs = cfg.out_dir / "cross_eval_samples.json"
    if cs.exists():
        cand = []
        for s in load_json(cs):
            ok = [i for i, y in enumerate(s["replies"]) if hygiene_ok(y)]
            if len(ok) >= 2:
                i, j = (int(x) for x in rng.choice(ok, 2, replace=False))
                if norm_text(s["replies"][i]) != norm_text(s["replies"][j]):
                    cand.append((s, i, j, abs(s["sim"][i] - s["sim"][j])))
        cand.sort(key=lambda c: -c[3])
        top = min(len(cand) // 3, cfg.human_validity_pairs // 2)
        rest = [int(k) for k in rng.permutation(np.arange(top, len(cand)))[:max(0, cfg.human_validity_pairs - top)]]
        for k in list(range(top)) + rest:
            s, i, j, m = cand[k]
            items.append({"kind": "validity", "comparison": "judge", "uid": s["uid"], "dialogue_id": s["dialogue_id"],
                          "A": ("reply_i", s["replies"][i], s["sim"][i]), "B": ("reply_j", s["replies"][j], s["sim"][j]),
                          "margin": m, "stratum": "top_margin" if k < top else "random"})
    pool = [t for t in ex.turns if t.split == "test" and n_words(t.gold_response) >= 5]
    for k in range(max(4, round(0.05 * len(items)))):
        t = pool[int(rng.integers(len(pool)))]
        items.append({"kind": "catch", "comparison": "catch", "uid": t.uid, "dialogue_id": t.dialogue_id,
                      "A": ("gold", t.gold_response, float("nan")),
                      "B": ("catch", CATCH_RESPONSES[k % len(CATCH_RESPONSES)], float("nan"))})
    if not items:
        raise RuntimeError("no items for the human packet: run eval (and cross-eval) first")
    packet, key = [], {}
    for pos, k in enumerate(rng.permutation(len(items))):
        it = items[int(k)]
        L, R = (it["B"], it["A"]) if rng.random() < 0.5 else (it["A"], it["B"])
        iid = f"item{pos:05d}"
        packet.append({"item_id": iid, "context": context_text(turn_of[it["uid"]]), "reply_1": L[1], "reply_2": R[1],
                       "double_annotate": int(rng.random() < cfg.human_overlap or it["kind"] == "catch")})
        key[iid] = {"kind": it["kind"], "comparison": it["comparison"], "uid": it["uid"],
                    "dialogue_id": it["dialogue_id"], "left": L[0], "right": R[0], "left_score": L[2],
                    "right_score": R[2], **({"margin": it["margin"], "stratum": it["stratum"]}
                                            if it["kind"] == "validity" else {})}
    with open(d / "items.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(packet[0]))
        w.writeheader()
        w.writerows(packet)
    with open(d / "annotations_template.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["annotator_id", "item_id"] + list(HUMAN_QUESTIONS))
    dump_json(key, d / "key_DO_NOT_SHARE.json")
    (d / "instructions.md").write_text(
        "# Annotation instructions\n\nEach item shows a customer-service conversation and two agent replies in "
        "random order. For each question answer 1 (reply_1), 2 (reply_2) or 0 (no real difference).\n\n"
        + "\n".join(f"* **{k}**: {v}" for k, v in HUMAN_QUESTIONS.items())
        + "\n\nItems with double_annotate=1 must be labelled by two different annotators.\n", encoding="utf-8")
    need = binomial_n(0.60)
    per = {c: sum(1 for it in items if it["comparison"] == c) for c in cfg.human_comparisons}
    ex.logger.info("human packet | %d items -> %s | a 60%% win rate needs ~%d non-tie judgements per comparison; "
                   "planned %s", len(packet), d, need, per)
    meta = {"n_items": len(packet), "per_comparison": per, "power_n_60pct": need}
    dump_json(meta, d / "packet_meta.json")
    return meta


def stage_human_analyze(cfg: Config) -> Dict[str, Any]:
    """Win rates from filled annotations*.csv against the hidden key.  Annotators below 80% on the attention
    checks are excluded (pre-registered).  Sign-flip tests on item scores clustered by dialogue, Holm across
    comparisons x questions, Krippendorff alpha on double-annotated items."""
    ex = Experiment(cfg, "8_human_analyze")
    d = cfg.out_dir / "human_study"
    key = load_json(d / "key_DO_NOT_SHARE.json")
    ann = []
    for f in sorted(d.glob("annotations*.csv")):
        if f.name != "annotations_template.csv":
            with open(f, encoding="utf-8") as fh:
                ann.extend(csv.DictReader(fh))
    if not ann:
        raise FileNotFoundError(f"no filled annotations*.csv in {d}")
    code = {"1": 1.0, "2": -1.0, "0": 0.0}
    acc: Dict[str, List[float]] = {}
    for a in ann:
        k = key.get(a["item_id"])
        v = code.get(str(a.get("q_satisfaction", "")).strip())
        if k and k["kind"] == "catch" and v is not None:
            acc.setdefault(a["annotator_id"], []).append(float(v != 0 and (v > 0) == (k["left"] == "gold")))
    keep = {u for u, v in acc.items() if np.mean(v) >= 0.8}
    dropped = sorted({a["annotator_id"] for a in ann} - keep)
    per_item: Dict[str, Dict[str, List[float]]] = {}
    for a in ann:
        k = key.get(a["item_id"])
        if a["annotator_id"] not in keep or not k or k["kind"] == "catch":
            continue
        for q in HUMAN_QUESTIONS:
            v = code.get(str(a.get(q, "")).strip())
            if v is None:
                continue
            if k["kind"] == "arm_pair":
                v = v if k["left"] == k["comparison"].split(":")[0] else -v
            else:
                v = v if k["left_score"] >= k["right_score"] else -v       # +1: the human agrees with the judge
            per_item.setdefault(a["item_id"], {}).setdefault(q, []).append(v)
    res: Dict[str, Any] = {"excluded_annotators": dropped, "comparisons": {}, "agreement": {}}
    names, pv = [], []
    for comp in sorted({k["comparison"] for k in key.values() if k["kind"] != "catch"}):
        for q in HUMAN_QUESTIONS:
            ids = [i for i, k in key.items() if k["comparison"] == comp and q in per_item.get(i, {})]
            if len(ids) < 10:
                continue
            sc = np.asarray([np.mean(per_item[i][q]) for i in ids])
            cl = np.asarray([key[i]["dialogue_id"] for i in ids], dtype=object)
            ci = cluster_bootstrap_ci(lambda ix: float(np.mean(sc[ix] > 0) + 0.5 * np.mean(sc[ix] == 0)), cl,
                                      cfg.n_boot, cfg.seed)
            ci90 = cluster_bootstrap_ci(lambda ix: float(np.mean(sc[ix] > 0) + 0.5 * np.mean(sc[ix] == 0)), cl,
                                        cfg.n_boot, cfg.seed + 1, conf=0.90)
            sf = sign_flip_test(sc, cl, cfg.n_signflip, cfg.seed + 2)
            r = {"n_items": len(ids), "win_rate": ci[0], "ci95": list(ci[1:]), "ci90": list(ci90[1:]),
                 "p_value": sf["p"], "p_text": sf["text"], "tie_rate": float(np.mean(sc == 0))}
            if q == "q_information" and comp != "judge":
                r["noninferior"] = bool(ci90[1] > 0.5 - cfg.tacit.margin_info)
            res["comparisons"].setdefault(comp, {})[q] = r
            names.append((comp, q))
            pv.append(sf["p"])
            ex.logger.info("human | %-24s %-15s win %.3f CI95[%.3f,%.3f] ties %.2f p %s n=%d", comp, q, ci[0], ci[1],
                           ci[2], r["tie_rate"], sf["text"], len(ids))
    for (comp, q), p in zip(names, holm(pv)):
        res["comparisons"][comp][q]["p_holm"] = p
    for q in HUMAN_QUESTIONS:
        units = {i: v[q] for i, v in per_item.items() if len(v.get(q, [])) >= 2}
        res["agreement"][q] = {"krippendorff_alpha": krippendorff_alpha_interval(units), "n_units": len(units)}
    ex.logger.info("human | %d annotators excluded by the attention check: %s", len(dropped), dropped)
    dump_json(res, cfg.out_dir / "human_study_results.json")
    return res


def certification_analysis(cfg: Config) -> Dict[str, Any]:
    """Seed-level evidence, from the train_*.json files: certification rates with Clopper-Pearson intervals,
    one-sided Fisher exact tests of tacit2 against every other selecting arm, the paired power comparison of
    Monte Carlo (k replies) vs single-reply certification on the SAME safety data (exact McNemar; sign test
    on the content-bound half-width ratio), and the agreement of the probe with the selection split
    (TACIT-1 E2: on held-out RL contexts the two should agree)."""
    out: Dict[str, Any] = {"arms": {}, "fisher_vs_primary": {}, "mc_power": {}, "probe_agreement": {}}
    runs: Dict[str, List[Dict[str, Any]]] = {}
    for arm in cfg.arms:
        if arm in ARM_FLAGS and ARM_FLAGS[arm]["select"]:
            rs = [load_json(f) for f in (cfg.out_dir / f"train_{arm}_s{sd}.json" for sd in cfg.seeds) if f.exists()]
            if rs:
                runs[arm] = rs
    b = c = 0
    ratios = []
    for arm, rs in runs.items():
        n, k = len(rs), sum(not r["returned_sft"] for r in rs)
        cert1 = [r["certification"]["k1"]["accept"] for r in rs if r["certification"].get("run")]
        out["arms"][arm] = {"n_seeds": n, "certified": k, "rate": k / n, "cp95": clopper_pearson(k, n),
                            "candidates": sum(r["selected_round"] is not None for r in rs),
                            "certified_with_one_reply": int(sum(cert1)),
                            "failed_checks": [[x for x, v in r["certification"].get("checks", {}).items() if not v]
                                              for r in rs if r["certification"].get("run")]}
        for r in rs:
            ce = r["certification"]
            if ce.get("run") and r.get("mc_k", 1) > 1:
                b += int(ce["accept"] and not ce["k1"]["accept"])
                c += int(ce["k1"]["accept"] and not ce["accept"])
                hk = ce["means"]["content"] - ce["lower_bounds"]["content"]
                h1 = ce["k1"]["means"]["content"] - ce["k1"]["lower_bounds"]["content"]
                if all(math.isfinite(x) for x in (hk, h1)) and h1 > 0:
                    ratios.append(hk / h1)
        pa = [(x["drift_after"]["content"], x["selection"]["means"]["content"]) for r in rs for x in r["rounds"]
              if "drift_after" in x and "selection" in x]
        if len(pa) >= 3:
            u, v = np.asarray(pa).T
            out["probe_agreement"][arm] = {"pairs": len(pa), "probe_content_drift": float(u.mean()),
                                           "selection_content": float(v.mean()), "pearson": pearson(u, v),
                                           "sign_disagreement": float(np.mean(np.sign(u) != np.sign(v)))}
    P = PRIMARY
    if P in out["arms"]:
        a = out["arms"][P]
        for arm, s in out["arms"].items():
            if arm != P:
                out["fisher_vs_primary"][arm] = fisher_exact_greater(a["certified"], a["n_seeds"], s["certified"],
                                                                     s["n_seeds"])
    r_ = np.asarray(ratios)
    out["mc_power"] = {"runs": int(r_.size), "k_pass_one_fail": b, "one_pass_k_fail": c,
                       "mcnemar_p": mcnemar_exact(b, c),
                       "content_halfwidth_ratio_median": float(np.median(r_)) if r_.size else float("nan"),
                       "content_halfwidth_ratio_max": float(r_.max()) if r_.size else float("nan"),
                       "sign_test_p": binom_sf(int(np.sum(r_ < 1)), int(r_.size), 0.5) if r_.size else float("nan")}
    return out


def stage_certification(cfg: Config) -> Dict[str, Any]:
    ex = Experiment(cfg, "9_certification")
    res = certification_analysis(cfg)
    dump_json(res, cfg.out_dir / "certification.json")
    for arm, s in res["arms"].items():
        ex.logger.info("certification %-22s %d/%d seeds (CP95 [%.2f, %.2f]) | one reply: %d | candidates %d | "
                       "Fisher p (tacit2 higher) %s", arm, s["certified"], s["n_seeds"], *s["cp95"],
                       s["certified_with_one_reply"], s["candidates"],
                       "-" if arm == PRIMARY else f"{res['fisher_vs_primary'].get(arm, float('nan')):.3f}")
    m = res["mc_power"]
    ex.logger.info("Monte Carlo certification over %d runs | content half-width ratio k vs 1: median %.3f, max %.3f, "
                   "sign test p %.4f | discordant: k-only pass %d, one-reply-only pass %d, McNemar p %.4f",
                   m["runs"], m["content_halfwidth_ratio_median"], m["content_halfwidth_ratio_max"], m["sign_test_p"],
                   m["k_pass_one_fail"], m["one_pass_k_fail"], m["mcnemar_p"])
    for arm, a in res["probe_agreement"].items():
        ex.logger.info("probe vs selection content (%s): %+.4f vs %+.4f | r %.2f | sign disagreement %.2f over %d",
                       arm, a["probe_content_drift"], a["selection_content"], a["pearson"], a["sign_disagreement"],
                       a["pairs"])
    return res


def stage_claims(cfg: Config) -> Dict[str, Any]:
    """Turn the artefacts into verdicts with pre-registered rules; missing evidence is "untested".

    Constraint satisfaction of an arm X (vs SFT, test split): information, hygiene and monitor
    non-inferior at the registered margins and log length equivalent (TOST).  A comparison with an arm
    that buys outcome by violating a constraint is reported as Pareto-incomparable, never as a loss or a
    win.  A component CONTRIBUTES only with (i) a higher certification rate (one-sided Fisher exact, 0.05),
    (ii) an ablation candidate that violates a constraint tacit2's candidate meets, or (iii) a significantly
    better outcome with constraints met, on the returned policies, or on the candidates (per-protocol).
    The returned-policy comparison is never used when the ablation returned SFT in every seed, because it
    would then measure the fallback, not the component (TACIT-1 E4).  Non-inferiority is never evidence of a
    contribution."""
    ex = Experiment(cfg, "9_claims")

    def get(name: str) -> Dict[str, Any]:
        f = cfg.out_dir / name
        return load_json(f) if f.exists() else {}
    rep, ext, hs = get("report_eval.json"), get("report_ext.json"), get("human_study_results.json")
    jv, ce = get("judge_validity.json"), get("cross_eval.json")
    cert = certification_analysis(cfg)
    dump_json(cert, cfg.out_dir / "certification.json")
    C, P = rep.get("contrasts", {}), PRIMARY
    eq = rep.get("equiv_margin_outcome", cfg.equiv_margin_outcome)
    claims: List[Dict[str, Any]] = []

    def add(cid: str, text: str, verdict: str, evidence: Any = None):
        claims.append({"id": cid, "claim": text, "verdict": verdict, "evidence": evidence})

    def sig_pos(c):
        return None if not c else bool(c["ci95"][0] > 0 and c.get("p_holm", 1.0) < 0.05)

    def verdict(v):
        return "untested" if v is None else ("supported" if v else "not supported")

    def constraints(arm: str) -> Tuple[Optional[bool], List[str]]:
        if arm == "sft":
            return True, []
        need = {"content": "noninferior", "hygiene": "noninferior", "monitor": "noninferior",
                "log_words": "equivalent"}
        bad = []
        for m, key in need.items():
            c = C.get(f"{arm}_vs_sft:{m}")
            if c is None or key not in c:
                return None, []
            if not c[key]:
                bad.append(m)
        return not bad, bad

    def all_sft(arm: str) -> Optional[bool]:
        rs = [r for r in (get(f"train_{arm}_s{sd}.json") for sd in cfg.seeds) if r]
        return all(r["returned_sft"] for r in rs) if rs else None

    def compare(a: str, b: str, mode: str) -> Tuple[str, Any]:
        """mode "pareto": a vs b as alternatives; "noninferior": a must be non-inferior to b on outcome."""
        c = C.get(f"{a}_vs_{b}:outcome")
        ok_a, bad_a = constraints(a)
        ok_b, bad_b = constraints(b)
        if c is None or ok_a is None or ok_b is None:
            return "untested", None
        ev = {"delta": c["delta"], "ci90": c["ci90"], "ci95": c["ci95"], f"{a}_violates": bad_a,
              f"{b}_violates": bad_b}
        if not ok_a:
            return f"not supported: {a} violates {bad_a}", ev
        if sig_pos(c):
            return "supported (higher outcome, constraints met)", ev
        if c["ci90"][0] > -eq:
            return ("supported (non-inferior outcome, constraints met)" if mode == "noninferior"
                    else "not supported: only non-inferior"), ev
        if not ok_b:
            return f"Pareto-incomparable: {b} has a higher outcome but violates {bad_b}", ev
        return "not supported", ev

    def contribution(abl: str) -> Tuple[str, Any]:
        ca, cb = cert["arms"].get(P), cert["arms"].get(abl)
        pc_, bc_ = f"{P}_cand", f"{abl}_cand"
        ev: Dict[str, Any] = {}
        if ca and cb:
            ev.update({"certified": f"{ca['certified']}/{ca['n_seeds']} vs {cb['certified']}/{cb['n_seeds']}",
                       "fisher_p": cert["fisher_vs_primary"].get(abl)})
            if ev["fisher_p"] is not None and ev["fisher_p"] < 0.05:
                return f"supported: certification rate {ev['certified']} (Fisher p {ev['fisher_p']:.3f})", ev
        ok_pc, _ = constraints(pc_)
        ok_bc, bad_bc = constraints(bc_)
        if ok_pc and ok_bc is False:
            ev["ablation_candidate_violates"] = bad_bc
            return f"supported: the ablation's candidate violates {bad_bc}", ev
        c_cand = C.get(f"{pc_}_vs_{bc_}:outcome")
        c_ret = C.get(f"{P}_vs_{abl}:outcome")
        ok_p, _ = constraints(P)
        if c_cand:
            ev["candidate_outcome"] = {k: c_cand.get(k) for k in ("delta", "ci95", "p_holm")}
        if c_ret:
            ev["returned_outcome"] = {k: c_ret.get(k) for k in ("delta", "ci95", "p_holm")}
        if all_sft(abl) is False and ok_p and sig_pos(c_ret):
            return "supported (higher outcome of the returned policy, constraints met)", ev
        if ok_pc and ok_bc and sig_pos(c_cand):
            return "supported (higher outcome of the candidate, constraints met; per-protocol)", ev
        if not ev:
            return "untested", None
        if all_sft(abl):
            ev["note"] = "the ablation returned SFT in every seed: its returned policy was not compared"
        few = len(cfg.seeds) < 5
        return ("not supported: no detectable contribution" + (f" (underpowered: {len(cfg.seeds)} seeds)"
                                                               if few else "")), ev

    c = C.get(f"{P}_vs_sft:outcome")
    add("H1", "TACIT-2 raises simulated customer affect over SFT", verdict(sig_pos(c)),
        c and {k: c.get(k) for k in ("delta", "ci95", "p_holm")})
    ok, bad = constraints(P)
    add("H2", "TACIT-2 keeps information, hygiene and monitor (non-inferior) and length (equivalent) vs SFT",
        "untested" if ok is None else ("supported" if ok else f"not supported: {bad}"),
        {m: {k: C.get(f"{P}_vs_sft:{m}", {}).get(k) for k in ("delta", "ci90")}
         for m in ("content", "hygiene", "monitor", "log_words")})
    for abl, cid, text in (("tacit2_no_linesearch", "H3a", "the certified line search contributes"),
                           ("tacit2_fixed_small", "H3b_small", "the adaptive radius beats TACIT-1's best fixed radius"),
                           ("tacit2_fixed_large", "H3b_large", "the adaptive radius beats the fixed best-of-n radius"),
                           ("tacit2_open_loop", "H3d", "the closed-loop moment targets contribute")):
        v, e = contribution(abl)
        add(cid, text, v, e)
    m = cert["mc_power"]
    add("H3c", "Monte Carlo certification narrows the content bound (half-width ratio k vs 1 < 1)",
        "untested" if not m["runs"] else verdict(m["content_halfwidth_ratio_median"] < 1 and m["sign_test_p"] < 0.05),
        m)
    add("H3c_power", "Monte Carlo certification certifies more often than one reply on the same data (McNemar)",
        "untested" if not m["runs"] else verdict(m["mcnemar_p"] < 0.05), m)
    mem = jv.get("memorisation")
    add("H3e", "SFT carries more task information on training dialogues it was fitted on (memorisation gap)",
        "untested" if not mem else verdict(mem["content_gap"][1] > 0),
        mem and {"content_gap": mem["content_gap"], "gold_copy_gap": mem["gold_copy_gap"],
                 "probe_agreement_on_heldout_contexts": cert["probe_agreement"].get(P)})
    v, e = compare(P, "sft_bon", "noninferior")
    add("H4a", f"amortisation: TACIT-2 at N = 1 (no judge at inference) is non-inferior to sft_bon at N = {cfg.bon_n}",
        v, e)
    v, e = compare(f"{P}_bon", "sft_bon", "pareto")
    add("H4b", f"compute-matched dominance: best-of-{cfg.bon_n} on TACIT-2 beats best-of-{cfg.bon_n} on SFT", v, e)
    for base, cid in (("tacit1", "H4c_tacit1"), ("campo", "H4c_campo")):
        v, e = compare(P, base, "pareto")
        add(cid, f"TACIT-2 beats {base} under the same budget, constraints met", v, e)
    for base in ("online_dpo", "offline_dpo"):
        v, e = compare(P, base, "pareto")
        add(f"H4d_{base}", f"Pareto: TACIT-2 vs {base}", v, e)
    e = ext.get("contrasts", {}).get(f"{P}_vs_sft:outcome")
    add("H5a", "the gain replicates with an out-of-family customer and an unseen emotion classifier",
        verdict(sig_pos(e)), e and {k: e.get(k) for k in ("delta", "ci95", "p_holm")})
    hc = hs.get("comparisons", {}).get(f"{P}:sft", {}).get("q_satisfaction")
    add("H5b", "humans prefer TACIT-2 over SFT on satisfaction",
        "untested" if not hc else verdict(hc["ci95"][0] > 0.5 and hc.get("p_holm", 1) < 0.05), hc)
    hv = hs.get("comparisons", {}).get("judge", {}).get("q_satisfaction")
    add("H5c", "the judge ranks alternative replies to the SAME context as humans do",
        "untested (needs the human study)" if not hv else verdict(hv["ci95"][0] > 0.5), hv)
    add("H5c_automatic", "two independent automatic evaluators agree on within-context rankings above chance",
        "untested" if not ce else verdict(ce["agreement"][1] > 0.5), ce and {"agreement": ce["agreement"]})
    add("judge_partial_anchor", "the judge tracks the next-turn human emotion beyond the current emotion",
        "untested" if not jv else verdict(jv["partial_anchor"][1] > 0), jv and jv.get("partial_anchor"))
    rows = load_rows(cfg, "eval") if (cfg.out_dir / f"eval_sft_{cfg.seeds[0]}.json").exists() else {}
    if P in rows and "sft" in rows:
        worse = []
        for sd in cfg.seeds:
            for mt, mg in (("outcome", 0.0), ("monitor", cfg.tacit.margin_monitor), ("content", cfg.tacit.margin_info),
                           ("hygiene", cfg.tacit.margin_hygiene)):
                cc = paired_contrast(rows[P], rows["sft"], mt, cfg.seed, cfg.n_boot, cfg.n_signflip, seeds=[sd])
                if cc.get("n", 0) >= 20 and cc["ci95"][1] < -mg:
                    worse.append({"seed": sd, "metric": mt, "delta": cc["delta"], "ci95": cc["ci95"]})
        add("H6", "no returned TACIT-2 policy is significantly worse than SFT on any objective (test split)",
            "supported" if not worse else "not supported", worse or None)
    s = cert["arms"].get(P)
    if s:
        add("H7", "TACIT-2 certification rate over seeds (Clopper-Pearson 95%)",
            f"{s['certified']}/{s['n_seeds']} certified, CP95 [{s['cp95'][0]:.2f}, {s['cp95'][1]:.2f}]"
            + ("" if s["certified"] < s["n_seeds"] else " (every seed)"), s)
    dump_json(claims, cfg.out_dir / "claims.json")
    lines = ["# Claims audit (generated; edit the rules, never the verdicts)", "", "| id | claim | verdict |", "|---|---|---|"]
    lines += [f"| {c['id']} | {c['claim']} | {c['verdict']} |" for c in claims]
    (cfg.out_dir / "claims.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for c in claims:
        ex.logger.info("CLAIM %-24s %s", c["id"], c["verdict"])
    return {"claims": claims}


def stage_train(cfg: Config) -> None:
    for arm in cfg.arms:
        if arm in ARM_FLAGS:
            for sd in cfg.seeds:
                train_arm(cfg, arm, sd)


def stage_all(cfg: Config) -> None:
    stage_data(cfg)
    stage_sft(cfg)
    stage_simulator(cfg)
    stage_validate(cfg)
    stage_preregister(cfg)
    stage_train(cfg)
    stage_eval(cfg)
    stage_eval_external(cfg)
    stage_cross_eval(cfg)
    stage_report(cfg, "eval")
    stage_report(cfg, "ext")
    stage_length(cfg)
    stage_human_export(cfg)
    stage_certification(cfg)
    stage_claims(cfg)


# ==========================================================================================================
# diagnose-v1: evidence E1-E5 from TACIT-1's publication tables
# ==========================================================================================================

def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def v1_diagnosis(d: Path, pc: TACITConfig) -> List[Dict[str, Any]]:
    """E1-E5 of the module docstring, recomputed from the CSVs that TACIT-1's publication export wrote
    (06_per_seed_certification, 08_round_trajectories, 04_contrasts, 03_arm_means)."""
    rows: List[Dict[str, Any]] = []

    def add(eid, question, verdict, evidence):
        rows.append({"id": eid, "question": question, "verdict": verdict, "evidence": evidence})
    cert = _read_csv(d / "06_per_seed_certification.csv")
    traj = _read_csv(d / "08_round_trajectories.csv")
    con = {(r["evaluator"], r["comparison"]): r for r in _read_csv(d / "04_contrasts.csv")}
    means = {(r["evaluator"], r["arm"]): r for r in _read_csv(d / "03_arm_means.csv")}
    # E1: failed checks and their slack
    fails = [r for r in cert if r.get("certification_run") == "True" and r.get("certified") == "False"]
    slack = {f"{r['arm']}_s{r['seed']}": round(_f(r["cert_content_lb"]) + pc.margin_info, 4) for r in fails
             if "content" in r.get("failed_checks", "")}
    only_content = bool(fails) and all(r["failed_checks"] == "content_noninferior" for r in fails
                                       if r["arm"].startswith("tacit"))
    add("E1", "Were TACIT-1's certification failures narrow content misses?",
        ("YES: every TACIT failure was content, max shortfall %.4f" % -min(slack.values())) if only_content and slack
        else "NO / missing", {"content_slack_of_failures": slack,
                              "failed_checks": {f"{r['arm']}_s{r['seed']}": r["failed_checks"] for r in fails}})
    # E2: probe (training contexts) vs selection (held-out) content drift of the same policy
    by = {(r["arm"], r["seed"], int(r["round"])): r for r in traj if r.get("drift_content", "") != ""}
    pc_, ps_, pl_, psl_ = [], [], [], []
    for (a, s, rd), r in by.items():
        nx = by.get((a, s, rd + 1))
        if nx and r.get("sel_content_mean", "") != "":
            pc_.append(_f(nx["drift_content"]))
            ps_.append(_f(r["sel_content_mean"]))
            pl_.append(_f(nx["drift_log_words"]))
            psl_.append(_f(r["sel_length_mean"]))
    if pc_:
        u, v, ul, vl = map(np.asarray, (pc_, ps_, pl_, psl_))
        ev = {"pairs": len(u), "probe_content": float(u.mean()), "selection_content": float(v.mean()),
              "sign_disagreement": float(np.mean(np.sign(u) != np.sign(v))), "pearson_content": pearson(u, v),
              "probe_length": float(ul.mean()), "selection_length": float(vl.mean()), "pearson_length": pearson(ul, vl)}
        blind = ev["probe_content"] > ev["selection_content"] + 0.01 and ev["sign_disagreement"] > 0.5
        add("E2", "Did the training-context probe miss the held-out content loss?",
            "YES: probe content %+.4f vs held-out %+.4f, signs disagree %.0f%%" % (
                ev["probe_content"], ev["selection_content"], 100 * ev["sign_disagreement"]) if blind else "NO", ev)
    else:
        add("E2", "Did the training-context probe miss the held-out content loss?", "missing", None)
    # E3: step size
    c = con.get(("training_judge", "tacit_vs_tacit_small_step:outcome"))
    stop = [r.get("mstep_kl_stop") == "True" for r in traj if r.get("mstep_kl_stop", "") != ""]
    sel_c = {}
    for arm in ("tacit", "tacit_small_step", "tacit_open_loop", "campo"):
        v = [_f(r["sel_content_mean"]) for r in traj if r["arm"] == arm and r.get("sel_content_mean", "") != ""]
        kb = [_f(r["kl_budget_median"]) for r in traj if r["arm"] == arm and r.get("kl_budget_median", "") != ""]
        if v:
            sel_c[arm] = {"selection_content": round(float(np.mean(v)), 4), "radius": round(float(np.mean(kb)), 3)}
    rate = {a: f"{sum(r['certified'] == 'True' for r in cert if r['arm'] == a)}/"
               f"{sum(r['arm'] == a for r in cert)}" for a in ("tacit", "tacit_small_step")}
    add("E3", "Did best-of-n-sized steps lose to small steps?",
        ("YES: tacit - tacit_small_step outcome %+.4f CI95 [%s, %s]" % (_f(c["delta"]), c["ci95_low"], c["ci95_high"]))
        if c and _f(c["ci95_high"]) < 0 else "NO / missing",
        {"selection_content_and_radius": sel_c, "mstep_stopped_at_kl_cap": float(np.mean(stop)) if stop else None,
         "certified": rate})
    # E4: the H3c artefact and the power of 3 seeds
    nop = [r for r in cert if r["arm"] == "tacit_no_predict"]
    fisher = None
    if rate.get("tacit_small_step") and rate.get("tacit"):
        k1, n1 = (int(x) for x in rate["tacit_small_step"].split("/"))
        k2, n2 = (int(x) for x in rate["tacit"].split("/"))
        if n1 and n2:
            fisher = fisher_exact_greater(k1, n1, k2, n2)
    add("E4", "Was TACIT-1's H3c verdict a fallback artefact, and could 3 seeds separate certification rates?",
        "YES: tacit_no_predict returned SFT in every seed" if nop and all(r["returned_sft"] == "True" for r in nop)
        else "NO / missing", {"tacit_no_predict_returned_sft": [r["returned_sft"] for r in nop],
                              "fisher_small_step_vs_tacit": fisher})
    # E5: amortisation by the small-step policy
    ev5 = {}
    for e in ("training_judge", "external_evaluator"):
        a, b = means.get((e, "tacit_small_step")), means.get((e, "sft_bon"))
        if a and b:
            ev5[e] = {"tacit_small_step": _f(a["outcome"]), "sft_bon": _f(b["outcome"])}
    add("E5", "Did a single-sample policy reach best-of-4 on SFT?",
        "YES (means)" if ev5 and all(v["tacit_small_step"] >= v["sft_bon"] for v in ev5.values()) else "NO / missing",
        ev5)
    return rows


def stage_diagnose_v1(cfg: Config) -> Dict[str, Any]:
    d = Path(cfg.v1_dir)
    if not d.exists():
        raise FileNotFoundError(f"--v1-dir {d} does not exist (unzip publication_tables.zip)")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    lg = make_logger(cfg.out_dir, "0_diagnose_v1")
    rows = v1_diagnosis(d, cfg.tacit)
    dump_json(rows, cfg.out_dir / "v1_diagnosis.json")
    md = ["# Diagnosis of TACIT-1 (generated from its publication tables)", "", "| id | question | verdict |",
          "|---|---|---|"] + [f"| {r['id']} | {r['question']} | {r['verdict']} |" for r in rows]
    (cfg.out_dir / "v1_diagnosis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    for r in rows:
        lg.info("%s | %s | %s | %s", r["id"], r["question"], r["verdict"], json.dumps(r["evidence"], default=_json_default))
    return {"diagnosis": rows}


# ==========================================================================================================
# tests
# ==========================================================================================================

def run_unit_tests() -> None:
    """Numerical checks of the statistics, the estimator, the E-step and the guarantee (numpy only)."""
    rng = np.random.default_rng(0)
    # quantiles, sign-flip exactness, rank helpers
    assert abs(normal_ppf(0.975) - 1.959963985) < 1e-6 and abs(normal_ppf(0.01) + 2.326347874) < 1e-6
    for p, df, ref in ((0.95, 10, 1.812461), (0.95, 30, 1.697261), (0.975, 20, 2.085963), (0.99, 50, 2.403272)):
        assert abs(t_ppf(p, df) - ref) < 2e-3, (p, df, t_ppf(p, df), ref)
    assert abs(sign_flip_test(np.ones(8), np.arange(8), 2000, 0)["p"] - 2 / 256) < 1e-12
    assert np.allclose(rankdata([3, 1, 2, 2]), [4, 1, 2.5, 2.5])
    z = rng.normal(size=3000)
    assert abs(partial_spearman(z + rng.normal(size=3000), z + rng.normal(size=3000), z)) < 0.06
    # SNIS: unbiased up to O(1/R) on a non-proposal target, and the delta-method standard error matches the
    # Monte Carlo spread of the estimate (the pessimism term of the reward)
    V = 30
    s = rng.normal(size=V)
    lg = rng.normal(size=(1, V)) * 1.5 + rng.normal(size=(4, V)) * 0.5
    Pm = np.exp(lg) / np.exp(lg).sum(1, keepdims=True)
    est, ses = [], []
    for _ in range(400):
        dr = np.concatenate([rng.choice(V, 20, p=Pm[m]) for m in range(3)])
        w, _ = snis_weights(np.log(Pm[3, dr]), balance_log_q(np.log(Pm[:3, dr])))
        est.append(float(w @ s[dr]))
        ses.append(snis_se(w, s[dr]))
    assert abs(np.mean(est) - Pm[3] @ s) < 0.05
    ratio = float(np.mean(ses) / np.std(est))
    assert 0.6 < ratio < 1.5, f"SNIS standard error off by a factor {ratio:.2f}"
    # E-step: the KL radius binds exactly, infeasible samples get no mass, degenerate rows are handled
    S = rng.normal(size=(200, 6))
    F = rng.random((200, 6)) < 0.8
    F[0] = False
    F[1] = [True, False, False, False, False, False]
    Q, eta = solve_temperatures(S, F, 0.5)
    assert np.all(Q[~F] == 0) and np.allclose(Q[F.any(1)].sum(1), 1.0) and Q[0].sum() == 0 and Q[1, 0] == 1.0
    soft = np.isfinite(eta) & (eta > 0)
    kls = np.array([kl_to_uniform(Q[c][F[c]]) for c in np.flatnonzero(soft)])
    assert np.allclose(kls, 0.5, atol=1e-3), kls[:5]
    hard = np.flatnonzero((F.sum(1) == 1) | (eta == 0))
    assert all(np.isclose(Q[c].max(), 1.0) or eta[c] == 0 for c in hard)
    # E-step moments: a reward that prefers SHORT, LESS informative replies (CASPI's failure, E4).  Unconstrained
    # tilting shortens and loses information; the constrained I-projection holds both moments and still gains.
    C, N = 400, 6
    L = rng.normal(2.6, 0.35, size=(C, N))
    I = np.clip(0.3 + 0.4 * (L - 2.6) + rng.normal(0, 0.1, size=(C, N)), 0, 1)
    R = -0.15 * (L - 2.6) + rng.normal(0, 0.1, size=(C, N))
    F = rng.random((C, N)) < 0.9
    free = estep(R, I, L, F, 0.5, False, 10.0, 5.0)
    cons = estep(R, I, L, F, 0.5, True, 10.0, 5.0)
    assert free["g_len"] < -0.05 and free["g_info"] < -0.02, (free["g_len"], free["g_info"])
    assert abs(cons["g_len"]) < 2e-3 and cons["g_info"] >= -1e-6, (cons["g_len"], cons["g_info"], cons["status"])
    gain = np.mean([(cons["weights"][c] * R[c]).sum() - R[c][F[c]].mean() for c in range(C) if F[c].any()])
    assert gain > 0, "the constrained target must still improve the reward"
    # mirror descent in the population: iterating the KL-constrained tilt raises E[r] monotonically, moves exactly
    # epsilon per step, and equals the single tilt pi_0 exp(r sum_t 1/eta_t) (policy mirror descent)
    p0 = rng.dirichlet(np.ones(50))
    r = rng.normal(size=50)
    p, inv, vals = p0.copy(), 0.0, [float(p0 @ r)]
    for _ in range(5):
        lo, hi = -12.0, 12.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            q = p * np.exp((r - r.max()) / math.exp(mid))
            q /= q.sum()
            if float(np.sum(q * np.log(q / p))) > 0.05:
                lo = mid
            else:
                hi = mid
        e_ = math.exp(hi)
        q = p * np.exp((r - r.max()) / e_)
        q /= q.sum()
        assert abs(float(np.sum(q * np.log(q / p))) - 0.05) < 1e-3
        p, inv = q, inv + 1.0 / e_
        vals.append(float(p @ r))
    assert all(b > a for a, b in zip(vals, vals[1:])), vals
    closed = p0 * np.exp((r - r.max()) * inv)
    assert np.allclose(p, closed / closed.sum(), atol=1e-8)
    # k3 KL estimator of the M-step stop: unbiased for KL(p || q) under samples of p, every term >= 0
    p_, q_ = rng.dirichlet(np.ones(40)), rng.dirichlet(np.ones(40))
    x_ = rng.choice(40, 200000, p=p_)
    d_ = np.log(q_[x_]) - np.log(p_[x_])
    k3 = np.expm1(d_) - d_
    assert k3.min() >= 0 and abs(k3.mean() - float(np.sum(p_ * np.log(p_ / q_)))) < 0.03 * float(np.sum(p_ * np.log(p_ / q_))) + 0.01
    # the certification guarantee: under every null the false-acceptance rate stays <= alpha; a real
    # improvement is accepted most of the time.  Units are dialogues (one context each).
    pc = TACITConfig()
    G, trials, alpha = 300, 600, 0.05
    cl = np.arange(G)

    def draw(shift):
        out = {k: rng.normal(shift.get(k, 0.0), 0.06 if k == "monitor" else 0.1, G)
               for k in ("monitor", "content", "hygiene", "length")}
        out["affect"] = rng.normal(shift.get("affect", 0.0), 0.06, G)
        return out
    for bound in ("t", "bernstein"):
        pcb = dataclasses.replace(pc, bound=bound)
        nulls = {"affect_zero": {"affect": 0.0}, "monitor_at_margin": {"affect": 0.02, "monitor": -pc.margin_monitor},
                 "content_at_margin": {"affect": 0.02, "content": -pc.margin_info},
                 "hygiene_at_margin": {"affect": 0.02, "hygiene": -pc.margin_hygiene},
                 "length_at_margin": {"affect": 0.02, "length": -pc.length_equiv}}
        for name, sh in nulls.items():
            rate = np.mean([vector_test(draw(sh), cl, alpha, pcb, 1e9, 1)["accept"] for _ in range(trials)])
            assert rate <= alpha + 3 * math.sqrt(alpha * (1 - alpha) / trials), (bound, name, rate)
    good = np.mean([vector_test(draw({"affect": 0.03}), cl, alpha, pc, 1e9, 1)["accept"] for _ in range(200)])
    assert good > 0.5, f"a real improvement is rarely certified (rate {good:.2f})"
    # information helpers
    assert info_recall("Booked at the Gonville, ref XYZ123.", "Your Gonville booking is done, reference XYZ123.") > 0.3
    assert info_recall("So sorry, glad to help!", "Your Gonville booking is done, reference XYZ123.") == 0.0
    assert abs(krippendorff_alpha_interval({"a": [1, 1], "b": [-1, -1], "c": [0, 0]}) - 1.0) < 1e-12
    # ---- TACIT-specific checks --------------------------------------------------------------------------
    # (a) the best-of-n KL bound: exact best-of-n of a discrete base never exceeds log n - (n-1)/n
    for _ in range(50):
        V_, n_ = int(rng.integers(5, 60)), int(rng.integers(2, 16))
        p_ = rng.dirichlet(np.ones(V_) * rng.uniform(0.2, 3.0))
        order = np.argsort(rng.normal(size=V_))          # distinct rewards: rank order
        cdf = np.cumsum(p_[order])
        pb = np.empty(V_)
        pb[order] = cdf ** n_ - np.r_[0.0, cdf[:-1]] ** n_
        kl_b = float(np.sum(np.where(pb > 0, pb * np.log(np.maximum(pb, 1e-300) / p_), 0.0)))
        assert kl_b <= bon_kl_budget(n_) + 1e-9, (kl_b, bon_kl_budget(n_), n_)
    # (b) per-row KL radii bind exactly (TACIT's best-of-n-sized steps)
    S2 = rng.normal(size=(100, 8))
    F2 = rng.random((100, 8)) < 0.9
    eps2 = np.minimum(bon_kl_budget(F2.sum(1)), 1.5)
    Q2, eta2 = solve_temperatures(S2, F2, eps2)
    soft2 = np.isfinite(eta2) & (eta2 > 0)
    assert np.allclose([kl_to_uniform(Q2[c][F2[c]]) for c in np.flatnonzero(soft2)], eps2[soft2], atol=1e-3)
    # (c) closed-loop targets are met exactly by the E-step
    tgt = estep(R, I, L, F, 0.8, True, 10.0, 5.0, len_target=0.04, info_target=0.01)
    assert abs(tgt["g_len"] - 0.04) < 2e-3 and tgt["g_info"] >= 0.01 - 1e-6, (tgt["g_len"], tgt["g_info"], tgt["status"])
    # (d) G3 as an identity: a per-context constant (pessimism with homogeneous se) leaves the weights unchanged
    Qa, _ = solve_temperatures(S2, F2, eps2)
    Qb, _ = solve_temperatures(S2 - 0.09, F2, eps2)
    assert np.allclose(Qa, Qb, atol=1e-9)
    # (e) the controller: a systematic per-round projection bias (-0.04 log length, the size seen in CAMPO seed
    # 43) plus probe noise accumulates without feedback and stays inside the margin with it
    open_d, closed_d = [], []
    for _ in range(400):
        do = dc = 0.0
        for _ in range(4):
            do += -0.04 + rng.normal(0, 0.01)
            meas = dc + rng.normal(0, 0.027)               # probe noise: 128 paired contexts, sd ~0.3
            dc += float(np.clip(-meas, -0.1, 0.1)) - 0.04 + rng.normal(0, 0.01)
        open_d.append(do)
        closed_d.append(dc)
    open_d, closed_d = np.abs(open_d), np.abs(closed_d)
    assert np.mean(open_d > 0.12) > 0.5 and np.mean(closed_d > 0.12) < 0.03, (np.mean(open_d > 0.12),
                                                                           np.mean(closed_d > 0.12))
    # (f) predicted-pass selection: among rounds that pass the SELECTION test, the share that then FAILS the
    # safety test is lower with the inflated prediction (Thomas et al., 2019)
    fail_plain, fail_pred, n_plain, n_pred = 0, 0, 0, 0
    pcs = TACITConfig()
    cl_sel, cl_safe = np.arange(400), np.arange(600)
    for _ in range(300):
        true_len = rng.uniform(-0.14, 0.0)

        def dd(G, mu_len):
            return {"affect": rng.normal(0.05, 0.12, G), "monitor": rng.normal(0.03, 0.12, G),
                    "content": rng.normal(0.0, 0.2, G), "hygiene": rng.normal(0.0, 0.05, G),
                    "length": rng.normal(mu_len, 0.3, G)}
        d_sel, d_safe = dd(400, true_len), dd(600, true_len)
        sel_ok = vector_test(d_sel, cl_sel, pcs.select_alpha, pcs, 1e9, 1)["accept"]
        safe_ok = vector_test(d_safe, cl_safe, pcs.alpha, pcs, 1e9, 1)["accept"]
        if sel_ok:
            n_plain += 1
            fail_plain += int(not safe_ok)
            if predict_pass(d_sel, cl_sel, 600, pcs.alpha, pcs, 2.0)["pass"]:
                n_pred += 1
                fail_pred += int(not safe_ok)
    r_plain, r_pred = fail_plain / max(n_plain, 1), fail_pred / max(n_pred, 1)
    assert n_pred > 0 and r_pred < r_plain, (r_plain, r_pred, n_plain, n_pred)
    # ---- TACIT-2 checks ---------------------------------------------------------------------------------
    # (g) Monte Carlo certification (step 6).  Per-dialogue content differences = between-dialogue effect +
    # decoder noise (sd 0.12 and 0.35: TACIT-1's single-reply sd was ~0.37).  Averaging k = 8 replies cuts the
    # sd to sqrt(s_b^2 + s_w^2 / k); the test keeps its level at the margin and gains power at a small loss.
    rng_mc = np.random.default_rng(2026)          # own stream: the check does not depend on test order

    def content_draw(mu, k, G=600):
        return mu + rng_mc.normal(0, 0.12, G) + rng_mc.normal(0, 0.35, (k, G)).mean(0)

    def cert_rate(mu, k, trials):
        return float(np.mean([confidence_bound(content_draw(mu, k), np.arange(600), 0.05, "lower", "t", 2.0)
                              > -pc.margin_info for _ in range(trials)]))
    sd1, sd8 = float(np.std(content_draw(0.0, 1, 20000))), float(np.std(content_draw(0.0, 8, 20000)))
    sd_theory = math.sqrt((0.12 ** 2 + 0.35 ** 2 / 8) / (0.12 ** 2 + 0.35 ** 2))
    assert abs(sd8 / sd1 - sd_theory) < 0.03, (sd8 / sd1, sd_theory)
    null8 = cert_rate(-pc.margin_info, 8, 3000)
    assert null8 <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / 3000), f"Monte Carlo test exceeds its level ({null8:.3f})"
    pow1, pow8 = cert_rate(-0.02, 1, 600), cert_rate(-0.02, 8, 600)
    assert pow8 > pow1 + 0.2, (pow1, pow8)
    # (h) the line-search tube (step 4): inside passes; each violated component fails, with negative slack
    pc2 = TACITConfig()
    inside = {"content": (-0.005, 0.005), "log_words": (0.01, 0.01), "hygiene": (0.0, 0.002)}
    ok_in, sl = constraint_tube(inside, pc2)
    assert ok_in and all(v > 0 for v in sl.values()), sl
    for k, bad in (("content", (-0.03, 0.005)), ("log_words", (-0.07, 0.01)), ("hygiene", (-0.02, 0.002))):
        ok_b, sl_b = constraint_tube({**inside, k: bad}, pc2)
        assert not ok_b and sl_b["length" if k == "log_words" else k] < 0, (k, sl_b)
    assert not constraint_tube({**inside, "content": (-0.005, 0.05)}, pc2)[0], "a noisy probe must not pass"
    # (i) trust-region radius: grows to the cap after full steps, shrinks to the floor after rejections
    e = pc2.eps_init
    for _ in range(10):
        e = next_radius(e, 1.0, pc2, 1.2)
    assert abs(e - 1.2) < 1e-12
    for _ in range(10):
        e = next_radius(e, 0.0, pc2, 1.2)
    assert abs(e - pc2.eps_min) < 1e-12 and next_radius(0.4, 0.5, pc2, 1.2) == 0.2
    Fk = np.ones((5, 4), bool)
    Fk[0, 1:] = False
    radii, kmax = step_radii(_MPO, Fk, 0.5, pc2)
    assert np.all(radii <= 0.5 + 1e-12) and radii[0] == 0.0 and kmax == 0.5
    Fk8 = np.ones((3, 8), bool)
    assert abs(step_radii(_MPO, Fk8, 1.5, pc2)[1] - bon_kl_budget(8)) < 1e-12, "M-step stop above the E-step radius"
    assert np.allclose(step_radii({**_MPO, "step": "bon"}, Fk, 0.5, pc2)[0][1:], bon_kl_budget(4))
    # (j) exact tests of the certification-rate claims
    assert abs(fisher_exact_greater(3, 3, 0, 3) - 0.05) < 1e-12 and abs(fisher_exact_greater(3, 3, 1, 3) - 0.2) < 1e-12
    assert abs(fisher_exact_greater(5, 5, 1, 5) - 6 / 252) < 1e-12
    assert abs(mcnemar_exact(5, 0) - 1 / 32) < 1e-12 and mcnemar_exact(0, 0) == 1.0
    assert abs(clopper_pearson(0, 5)[1] - (1 - 0.025 ** 0.2)) < 1e-6
    assert abs(clopper_pearson(5, 5)[0] - 0.025 ** 0.2) < 1e-6
    for p_true, n_cp in ((0.3, 20), (0.8, 5), (0.5, 12)):
        cov = sum(math.comb(n_cp, k) * p_true ** k * (1 - p_true) ** (n_cp - k) for k in range(n_cp + 1)
                  if clopper_pearson(k, n_cp)[0] <= p_true <= clopper_pearson(k, n_cp)[1])
        assert cov >= 0.95 - 1e-9, (p_true, n_cp, cov)
    # (k) the RL split: deterministic, whole dialogues, about the requested share, other splits untouched
    toy = [Turn(f"d{i // 3}", f"d{i // 3}#{i % 3}", "train" if i % 30 < 24 else "test", "multiwoz", i % 3, "", "u",
                "g", None, 0, None) for i in range(6000)]
    s1, s2 = assign_rl_split(toy, 0.15), assign_rl_split(toy, 0.15)
    assert [t.split for t in s1] == [t.split for t in s2]
    rl_d = {t.dialogue_id for t in s1 if t.split == "rl"}
    assert all(t.split == "rl" for t in s1 if t.dialogue_id in rl_d)
    assert all(t.split == "test" for t, o in zip(s1, toy) if o.split == "test")
    share = len(rl_d) / len({t.dialogue_id for t in toy if t.split == "train"})
    assert 0.11 < share < 0.19, share
    # (l) paired drift statistics
    a_ = {"content": np.array([0.2, np.nan, 0.4, 0.5]), "log_words": np.array([2.0, 2.1, 2.2, 2.3]),
          "hygiene": np.ones(4)}
    b_ = {"content": np.array([0.1, 0.3, 0.3, 0.5]), "log_words": np.array([2.0, 2.0, 2.0, 2.0]),
          "hygiene": np.ones(4)}
    ds = drift_stats(a_, b_)
    assert abs(ds["content"][0] - 0.2 / 3) < 1e-12 and abs(ds["log_words"][0] - 0.15) < 1e-12 and ds["hygiene"] == (0.0, 0.0)
    print(f"unit tests OK | quantiles, sign-flip, SNIS (mean {np.mean(est):+.3f} vs truth {Pm[3] @ s:+.3f}; se ratio "
          f"{ratio:.2f}), E-step KL radius, moment constraints (free: d len {free['g_len']:+.3f}, d info "
          f"{free['g_info']:+.3f}; constrained: {cons['g_len']:+.4f}, {cons['g_info']:+.4f}, reward gain {gain:+.3f}), "
          f"mirror descent, certification level and power ({good:.2f}) | TACIT: best-of-n KL bound, per-row radii, "
          f"closed-loop targets, shift invariance (G3), controller |drift|>0.12: open {np.mean(open_d > 0.12):.2f} vs "
          f"closed {np.mean(closed_d > 0.12):.2f}, certification failures after selection: plain {r_plain:.2f} vs "
          f"predicted-pass {r_pred:.2f} | TACIT-2: Monte Carlo sd ratio {sd8 / sd1:.3f} (theory {sd_theory:.3f}), "
          f"level at the margin {null8:.3f}, power at -0.02 content: one reply {pow1:.2f} vs 8 replies {pow8:.2f}, "
          f"line-search tube, radius update, Fisher/McNemar/Clopper-Pearson, RL split share {share:.3f}, drift stats")


def make_synthetic_emowoz(data_dir: Path, n_dialogues: int, seed: int) -> List[str]:
    """Small EmoWOZ-format corpus for the self-test; returns every text for the tiny tokenizer."""
    rng = np.random.default_rng(seed)
    split = {"train": {"multiwoz": [], "dialmage": []}, "dev": {"multiwoz": [], "dialmage": []},
             "test": {"multiwoz": [], "dialmage": []}}
    data: Dict[str, Dict[str, Any]] = {"multiwoz": {}, "dialmage": {}}
    texts = []
    for i in range(n_dialogues):
        src = "multiwoz" if i % 4 else "dialmage"
        log, mood = [], 0.0
        for _ in range(int(rng.integers(3, 6))):
            emo = 6 if mood > 0.5 else 2 if mood < -0.5 else 0
            lead = {6: "Great, thanks!", 2: "That is really annoying.", 0: ""}[emo]
            log.append({"text": norm_text(f"{lead} I need a hotel for {int(rng.integers(2, 6))} people on monday."),
                        "emotion": [{"emotion": emo}] * 4})
            bad = rng.random() < 0.4
            emp = rng.random() < 0.5
            body = ("There is no hotel available at 10:00." if bad else
                    f"Your hotel is booked, the reference number is AB{int(rng.integers(100, 999))}.")
            log.append({"text": ("I am sorry about that. " if emp else "") + body, "emotion": []})
            mood = 0.5 * mood + (-0.9 if bad else 0.6) + (0.5 if emp else 0.0)
        log.append({"text": "Thanks, that is all." if mood > 0 else "This is not good enough.",
                    "emotion": [{"emotion": 6 if mood > 0 else 2}] * 4})
        did = f"SYN{i:04d}.json"
        data[src][did] = {"log": log}
        split["train" if i % 10 < 6 else "dev" if i % 10 < 8 else "test"][src].append(did)
        texts += [u["text"] for u in log]
    data_dir.mkdir(parents=True, exist_ok=True)
    dump_json(data["multiwoz"], data_dir / "emowoz-multiwoz.json")
    dump_json(data["dialmage"], data_dir / "emowoz-dialmage.json")
    dump_json(split, data_dir / "data-split.json")
    return texts


def build_tiny_models(root: Path, texts: Sequence[str]) -> Dict[str, str]:
    """Tiny randomly initialised models and a word-level tokenizer, saved locally (no downloads)."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import (LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, RobertaConfig,
                              RobertaForSequenceClassification)
    torch.manual_seed(0)
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
    for t in list(texts) + [AGENT_SYSTEM, CUSTOMER_SYSTEM,
                            "Conversation so far: Customer: Agent: Write only one two"]:
        for w in re.findall(r"\w+|[^\w\s]", t):
            vocab.setdefault(w, len(vocab))

    def tokenizer(pair_template: bool):
        core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        core.pre_tokenizer = pre_tokenizers.Whitespace()
        core.decoder = decoders.WordPiece(prefix="##")
        if pair_template:
            core.post_processor = processors.TemplateProcessing(
                single="<s> $A </s>", pair="<s> $A </s> </s> $B </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
        return PreTrainedTokenizerFast(tokenizer_object=core, bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                                       unk_token="<unk>", cls_token="<s>", sep_token="</s>")
    enc = dict(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
               max_position_embeddings=520, pad_token_id=1, bos_token_id=0, eos_token_id=2, type_vocab_size=1)
    labels = {"sentiment": ["negative", "neutral", "positive"], "monitor": ["anger", "neutral", "joy"],
              "external_emotion": ["sadness", "neutral", "joy"]}
    paths = {}
    for name, labs in labels.items():
        p = root / name
        RobertaForSequenceClassification(RobertaConfig(**enc, num_labels=3, id2label=dict(enumerate(labs)),
                                                       label2id={v: k for k, v in enumerate(labs)})).save_pretrained(p)
        tokenizer(True).save_pretrained(p)
        paths[name] = str(p)
    for name in ("lm", "external_lm"):
        p = root / name
        LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=48, intermediate_size=96, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                                     tie_word_embeddings=True, bos_token_id=0, eos_token_id=2,
                                     pad_token_id=1)).save_pretrained(p)
        tokenizer(False).save_pretrained(p)
        paths[name] = str(p)
    return paths


def _fake_annotations(cfg: Config) -> None:
    """Self-test only: two annotators who follow the judge with noise and one who clicks at random."""
    key = load_json(cfg.out_dir / "human_study" / "key_DO_NOT_SHARE.json")
    rng = np.random.default_rng(5)
    rows = []
    for ann in ("a1", "a2", "random_clicker"):
        for iid, k in key.items():
            ans = {}
            for q in HUMAN_QUESTIONS:
                if ann == "random_clicker":
                    ans[q] = str(int(rng.integers(0, 3)))
                elif k["kind"] == "catch":
                    ans[q] = "1" if k["left"] == "gold" else "2"
                else:
                    ans[q] = ("1" if k["left_score"] > k["right_score"] else "2") if rng.random() < 0.7 \
                        else str(int(rng.integers(0, 3)))
            rows.append({"annotator_id": ann, "item_id": iid, **ans})
    with open(cfg.out_dir / "human_study" / "annotations_selftest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["annotator_id", "item_id"] + list(HUMAN_QUESTIONS))
        w.writeheader()
        w.writerows(rows)


def _check_scoring_and_updates(cfg: Config) -> None:
    """Real-code-path checks on the tiny model: (i) tail-logit scoring equals a plain full-logit computation;
    (ii) the base model (adapter disabled) equals the reference policy of a fresh adapter; (iii) a DPO step
    raises the chosen-vs-rejected margin; (iv) adapters survive a save/load round trip."""
    import torch
    lg = logging.getLogger("tacit.check")
    pol = Policy(cfg.base_model, cfg, lg)
    prefixes = ["Customer : I need a hotel .", "Agent : Your hotel is booked ."]
    conts = [pol.text_ids("Thanks , that is all ."), pol.text_ids("This is not good enough .")]
    fast = pol.score(prefixes, conts)
    with torch.no_grad():
        for k, (p, c) in enumerate(zip(prefixes, conts)):
            ids = torch.tensor([pol.prefix_ids(p) + c])
            logp = torch.log_softmax(pol.model(input_ids=ids).logits[0, :-1].float(), -1)
            ref = float(logp[torch.arange(len(pol.prefix_ids(p)) - 1, ids.shape[1] - 1), ids[0, len(pol.prefix_ids(p)):]].sum())
            assert abs(fast[k] - ref) < 1e-3, (fast[k], ref)
    assert np.allclose(pol.score(prefixes, conts, adapter=False), fast, atol=1e-5), "fresh adapter != base model"
    ref_snap = pol.snapshot()
    pairs = [(prefixes[0], "Thanks , that is all .", "This is not good enough .")] * 4
    before = pol.score([prefixes[0]] * 2, [pol.text_ids(pairs[0][1]), pol.text_ids(pairs[0][2])])
    # 1e-2: at 5e-2 Adam overshoots on this 2-layer random model (margin 0.34 -> 0.03); at 1e-2 and 3e-3 it rises
    upd = pol.dpo_update(pairs, [1.0] * 4, ref_snap, 0.5, 1e-2, 2, 2, 0.0, 0)
    after = pol.score([prefixes[0]] * 2, [pol.text_ids(pairs[0][1]), pol.text_ids(pairs[0][2])])
    assert (after[0] - after[1]) > (before[0] - before[1]), (before, after)
    assert upd["d_chosen_logp"] - upd["d_rejected_logp"] > 0
    # the weighted M-step raises the weighted target log-likelihood, and the KL stop engages when kl_max is tiny
    pol.restore(ref_snap)
    tgt = [pol.text_ids("Thanks , that is all ."), pol.text_ids("This is not good enough .")]
    probe_ids = pol.generate_ids([prefixes[1]] * 8, EXACT_GEN, 0)     # exact pi_t samples, as in training
    m1 = pol.fit_weighted([prefixes[0]] * 2, tgt, [0.9, 0.1], [prefixes[1]] * 8, probe_ids, 1e-2, 3, 2, None, 1, 0)
    assert m1["d_target_logp"] > 0, m1
    pol.restore(ref_snap)
    m2 = pol.fit_weighted([prefixes[0]] * 2, tgt, [0.9, 0.1], [prefixes[1]] * 8, probe_ids, 1e-2, 3, 2, 1e-6, 1, 0)
    assert m1["kl_estimate"] >= 0 and m2["kl_stop"] and m2["kl_estimate"] <= 1e-6 + 1e-9, m2
    # the line-search path: alpha = 0 is theta_t (here the fresh adapter), alpha = 1 is the M-step result
    end = pol.snapshot()
    s_end = pol.score(prefixes, conts)
    pol.interpolate(ref_snap, end, 0.0)
    assert np.allclose(pol.score(prefixes, conts), fast, atol=1e-5), "interpolate(0) != start"
    pol.interpolate(ref_snap, end, 1.0)
    assert np.allclose(pol.score(prefixes, conts), s_end, atol=1e-5), "interpolate(1) != end"
    pol.save_adapter(cfg.out_dir / "check_adapter")
    pol2 = Policy(cfg.base_model, cfg, lg)
    pol2.load_adapter(cfg.out_dir / "check_adapter")
    assert np.allclose(pol2.score(prefixes, conts), pol.score(prefixes, conts), atol=1e-5), "adapter round trip"
    print("scoring/update checks OK | tail-logit scoring exact | DPO margin %+.3f -> %+.3f | weighted M-step d log p "
          "%+.3f, KL stop engaged at step %d | line-search interpolation exact at both ends" % (
              before[0] - before[1], after[0] - after[1], m1["d_target_logp"], m2["steps"]))



def _write_v1_tables(d: Path) -> None:
    """Self-test only: minimal TACIT-1 publication tables with the published schema and E1-E5's pattern."""
    d.mkdir(parents=True, exist_ok=True)

    def w(name, header, rows):
        with open(d / name, "w", newline="", encoding="utf-8") as fh:
            cw = csv.writer(fh)
            cw.writerow(header)
            cw.writerows(rows)
    w("06_per_seed_certification.csv", ["arm", "seed", "returned_sft", "certification_run", "certified",
                                        "failed_checks", "cert_content_lb"],
      [["tacit", 42, True, True, False, "content_noninferior", -0.0519],
       ["tacit", 43, False, True, True, "", -0.0126],
       ["tacit", 44, True, True, False, "content_noninferior", -0.0508],
       ["tacit_no_predict", 42, True, True, False, "content_noninferior", -0.0519],
       ["tacit_small_step", 42, False, True, True, "", -0.0414],
       ["tacit_small_step", 43, False, True, True, "", -0.0236],
       ["tacit_small_step", 44, False, True, True, "", -0.0387]])
    hdr = ["arm", "seed", "round", "drift_content", "drift_log_words", "sel_content_mean", "sel_length_mean",
           "mstep_kl_stop", "kl_budget_median"]
    w("08_round_trajectories.csv", hdr,
      [["tacit", 42, r, 0.01 if r > 1 else 0.0, -0.05, -0.03, -0.05, True, 1.2] for r in (1, 2, 3, 4)]
      + [["tacit_small_step", 42, r, 0.0, -0.04, -0.01, -0.04, True, 0.5] for r in (1, 2, 3, 4)])
    w("04_contrasts.csv", ["evaluator", "comparison", "delta", "ci95_low", "ci95_high"],
      [["training_judge", "tacit_vs_tacit_small_step:outcome", -0.0731, -0.0845, -0.0625]])
    w("03_arm_means.csv", ["evaluator", "arm", "outcome"],
      [["training_judge", "tacit_small_step", 0.3772], ["training_judge", "sft_bon", 0.3655],
       ["external_evaluator", "tacit_small_step", 0.2748], ["external_evaluator", "sft_bon", 0.2445]])


def run_selftest(root: Path = Path("tacit2_selftest")) -> None:
    """The whole pipeline on tiny random local models (CPU, minutes).  Checks plumbing and invariants; the
    numbers it produces mean nothing."""
    run_unit_tests()
    if root.exists():
        shutil.rmtree(root)
    texts = make_synthetic_emowoz(root / "data", 220, 0)
    paths = build_tiny_models(root / "tiny", texts)
    cfg = build_config(parse_args([
        "all", "--device", "cpu", "--data-dir", str(root / "data"), "--out", str(root / "run"),
        "--base-model", paths["lm"], "--external-model", paths["external_lm"], "--sentiment-model", paths["sentiment"],
        "--monitor-model", paths["monitor"], "--external-emotion-model", paths["external_emotion"],
        "--seeds", "42", "43", "--sft-epochs", "1", "--sim-sft-epochs", "1", "--rl-holdout-frac", "0.3",
        "--sft-max-train", "200", "--sim-turns", "200", "--sft-dev-examples", "40", "--sim-sft-dev-examples", "40",
        "--tacit-rounds", "2", "--tacit-contexts-per-round", "10", "--tacit-kl-holdout", "4", "--tacit-n-samples", "4",
        "--tacit-probe-contexts", "8", "--tacit-probe-k", "2", "--tacit-mc-k", "2", "--tacit-pool-size", "6",
        "--tacit-guard-pool-size", "6", "--tacit-select-contexts", "30", "--tacit-safety-contexts", "30",
        "--tacit-epochs", "1", "--tacit-kl-check-every", "2", "--memorisation-contexts", "20",
        "--validate-contexts", "40", "--eval-turns", "30",
        "--eval-pool-size", "8", "--external-pool-size", "6", "--cross-eval-contexts", "30", "--cross-eval-k", "3",
        "--human-contexts", "20", "--human-validity-pairs", "20", "--n-boot", "200", "--n-signflip", "2000",
        "--sim-max-new-tokens", "8", "--no-load-4bit"]))
    _check_scoring_and_updates(cfg)
    ex = Experiment(cfg, "selftest_split")
    rl_d = {t.dialogue_id for t in ex.turns if t.split == "rl"}
    assert rl_d and not rl_d & {t.dialogue_id for t in ex.turns if t.split == "train"}, "RL dialogues leak into SFT"
    stage_all(cfg)
    # A random tiny model almost never emits EOS after a terminated sentence, so no sample is feasible and the
    # pipeline above (correctly) made no update.  To exercise the E-step, M-step, line search and select ->
    # certify path, these runs open the feasibility filter (random 70% of samples) and force the selection
    # decision (both the selection-split test and the predicted-pass rule) to "accept"; the probe, the closed-
    # loop E-step, the M-step, the line search, the radius update and the certification test are the real ones.
    real_test, real_feat, real_pred = globals()["vector_test"], globals()["sample_features"], globals()["predict_pass"]
    frng = np.random.default_rng(7)

    def forced(diffs, cluster, alpha, pc, ess, pool):
        out = real_test(diffs, cluster, alpha, pc, ess, pool)
        return out if alpha == pc.alpha else {**out, "accept": True}

    def open_features(pol, ctx, ids, texts):
        I, L, F = real_feat(pol, ctx, ids, texts)
        return I, L, F | (frng.random(F.shape) < 0.7)

    def forced_pred(*a, **k):
        return {**real_pred(*a, **k), "pass": True, "predicted_affect_lb": 0.0}
    globals()["vector_test"], globals()["sample_features"], globals()["predict_pass"] = forced, open_features, forced_pred
    try:
        # a wide tube: the line search accepts the full step, the radius grows
        wide = dataclasses.replace(cfg, tacit=dataclasses.replace(cfg.tacit, lr=5e-2, line_tol=1e6))
        res = train_arm(wide, PRIMARY, 99)
        # a closed tube: every step is rejected, the policy stays SFT, the radius shrinks, nothing is selected
        shut = dataclasses.replace(cfg, tacit=dataclasses.replace(cfg.tacit, lr=5e-2, line_tol=0.0))
        res0 = train_arm(shut, PRIMARY, 98)
    finally:
        globals()["vector_test"], globals()["sample_features"], globals()["predict_pass"] = real_test, real_feat, real_pred
    assert res["selected_round"] is not None and res["certification"]["run"], res["certification"]
    assert res["returned_sft"] == (not res["certification"]["accept"])
    assert "k1" in res["certification"] and res["certification"]["k"] == cfg.tacit.mc_k
    radius = cfg.tacit.eps_init
    for r in res["rounds"]:
        es = r["estep"]
        assert es["m_step_targets"] > 0 and r["update"]["steps"] > 0 and "drift" in r and "probe" in r, r
        assert r["line_search"]["alpha"] == 1.0 and r["radius"] == radius, r["line_search"]
        assert 0.0 < es["kl_budget_median"] <= min(radius, bon_kl_budget(4)) + 1e-9, es
        radius = min(radius * cfg.tacit.eps_up, cfg.tacit.epsilon_cap)
        assert abs(r["radius_next"] - radius) < 1e-12
        if "length ok" in es["status"]:
            assert abs(es["g_len"] - es["len_target"]) < 1e-2, es
        if es["status"].startswith("info"):
            assert es["g_info"] >= es["info_target"] - 1e-6 or "infeasible" in es["status"], es
    radius = cfg.tacit.eps_init
    for r in res0["rounds"]:
        assert r["line_search"]["alpha"] == 0.0 and len(r["line_search"]["tried"]) == len(cfg.tacit.line_alphas), r
        assert "selection_skipped" in r and r["update"]["steps"] > 0, r
        radius = max(cfg.tacit.eps_min, radius * cfg.tacit.eps_down)
        assert abs(r["radius_next"] - radius) < 1e-12
    assert res0["returned_sft"] and res0["selected_round"] is None and (cfg.out_dir / "cand_tacit2_s98").exists()
    _fake_annotations(cfg)
    hs = stage_human_analyze(cfg)
    claims = {c["id"]: c for c in stage_claims(cfg)["claims"]}
    rep = load_json(cfg.out_dir / "report_eval.json")
    assert set(rep["arms"]) == set(KNOWN_ARMS), set(KNOWN_ARMS) ^ set(rep["arms"])
    assert f"{PRIMARY}_cand_vs_tacit1_cand:outcome" in rep["contrasts"]
    for arm in ARM_FLAGS:
        run = load_json(cfg.out_dir / f"train_{arm}_s42.json")
        assert run["rounds"] and run["flags"] == ARM_FLAGS[arm]
        if ARM_FLAGS[arm]["select"] and not ARM_FLAGS[arm]["offline"]:
            assert all("selection" in r or "selection_skipped" in r for r in run["rounds"]), arm
            assert (cfg.out_dir / f"cand_{arm}_s42").exists(), arm
    rows = load_json(cfg.out_dir / "eval_sft_42.json")
    assert all(math.isfinite(r["outcome"]) and math.isfinite(r["monitor"]) for r in rows)
    assert "random_clicker" in hs["excluded_annotators"]
    assert {"H1", "H2", "H3a", "H3b_small", "H3b_large", "H3c", "H3c_power", "H3d", "H3e", "H4a", "H4b", "H4c_tacit1",
            "H4c_campo", "H4d_online_dpo", "H6", "H7"} <= set(claims), set(claims)
    assert any(r["arm"] == f"{PRIMARY}_bon" and r["test_time_samples"] == cfg.bon_n
               for r in load_json(cfg.out_dir / f"eval_{PRIMARY}_bon_42.json"))
    jv = load_json(cfg.out_dir / "judge_validity.json")
    assert "memorisation" in jv and len(jv["memorisation"]["content_gap"]) == 3
    for name in ("preregistration.json", "judge_validity.json", "cross_eval.json", "report_ext.json",
                 "length_analysis.json", "claims.md", "certification.json", "human_study/items.csv"):
        assert (cfg.out_dir / name).exists(), name
    _write_v1_tables(root / "v1_tables")
    dg = {r["id"]: r for r in stage_diagnose_v1(dataclasses.replace(cfg, v1_dir=str(root / "v1_tables")))["diagnosis"]}
    assert set(dg) == {"E1", "E2", "E3", "E4", "E5"} and all(v["verdict"].startswith("YES") for v in dg.values()), dg
    print(f"\nSELFTEST OK | {len(KNOWN_ARMS)} arms trained/evaluated on tiny models | claims: "
          f"{ {k: v['verdict'] for k, v in claims.items()} }")
# ==========================================================================================================
# command line
# ==========================================================================================================

STAGES = {"all": stage_all, "data": stage_data, "sft": stage_sft, "simulator": stage_simulator,
          "validate": stage_validate, "preregister": stage_preregister, "train": stage_train, "eval": stage_eval,
          "eval-external": stage_eval_external, "cross-eval": stage_cross_eval,
          "report": lambda c: (stage_report(c, "eval"), stage_report(c, "ext") if
                               (c.out_dir / f"ext_sft_{c.seeds[0]}.json").exists() else None),
          "length": stage_length, "human-export": stage_human_export, "human-analyze": stage_human_analyze,
          "certification": stage_certification, "claims": stage_claims, "diagnose-v1": stage_diagnose_v1}


def _flags(obj: Any, prefix: str = "") -> List[Tuple[str, str, Any]]:
    """(cli flag, dotted path, default) for every scalar/tuple field of the configuration, recursively, so the
    command line can never drift from the dataclass defaults."""
    out = []
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        path = f"{prefix}{f.name}"
        if dataclasses.is_dataclass(v):
            out += _flags(v, path + ".")
        else:
            out.append(("--" + path.replace(".", "-").replace("_", "-"), path, v))
    return out


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"TACIT-2 ({VERSION})")
    p.add_argument("stage", choices=list(STAGES) + ["unittest", "selftest"])
    p.add_argument("--arm", default=None, help="restrict train to one arm")
    for flag, path, v in _flags(Config()):
        dest = path.replace(".", "__")
        if isinstance(v, bool):
            p.add_argument(flag, dest=dest, action="store_true", default=v)
            p.add_argument("--no-" + flag[2:], dest=dest, action="store_false")
        elif isinstance(v, tuple):
            p.add_argument(flag, dest=dest, nargs="+", type=type(v[0]) if v else str, default=list(v))
        else:
            p.add_argument(flag, dest=dest, type=type(v), default=v)
    return p.parse_args(list(argv))


def build_config(a: argparse.Namespace) -> Config:
    cfg = Config()
    for _, path, v in _flags(cfg):
        val = getattr(a, path.replace(".", "__"))
        obj = cfg
        parts = path.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], tuple(val) if isinstance(v, tuple) else val)
    if not cfg.device:
        try:
            import torch
            cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            cfg.device = "cpu"
    if a.arm:
        cfg.arms = (a.arm,)
    unknown = [x for x in cfg.arms if x not in KNOWN_ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; choose from {KNOWN_ARMS}")
    if cfg.tacit.bound not in ("t", "bernstein"):
        raise ValueError("--tacit-bound must be t or bernstein")
    al = list(cfg.tacit.line_alphas)
    if not al or al != sorted(al, reverse=True) or not all(0.0 < a <= 1.0 for a in al):
        raise ValueError("--tacit-line-alphas must be decreasing values in (0, 1]")
    if cfg.tacit.mc_k < 1 or cfg.tacit.mc_k_judge < 1 or cfg.tacit.probe_k < 1:
        raise ValueError("--tacit-mc-k, --tacit-mc-k-judge and --tacit-probe-k must be >= 1")
    if not 0.0 <= cfg.rl_holdout_frac < 0.9:
        raise ValueError("--rl-holdout-frac must be in [0, 0.9)")
    if not 0.0 < cfg.tacit.eps_min <= cfg.tacit.eps_init <= cfg.tacit.epsilon_cap:
        raise ValueError("need 0 < eps_min <= eps_init <= epsilon_cap")
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if a.stage == "unittest":
        run_unit_tests()
    elif a.stage == "selftest":
        run_selftest()
    else:
        STAGES[a.stage](build_config(a))


if __name__ == "__main__":
    main()
