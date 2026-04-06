"""
scripts/run_envA_v2_iql_sanity.py
Shared IQL training infrastructure for EnvA_v2 discrete experiments.

Defines the frozen IQL configuration and training framework reused by
all subsequent IQL experiment scripts (R2 main, R4 quality sweep, R3
EnvB/C validation).  All downstream scripts import directly from this
module and must not modify these frozen definitions.

Exports
-------
IQL_CFG                    -- frozen IQL hyperparameters
train_iql(...)             -- IQL training loop for OBS_DIM=900
save_iql_checkpoint(...)   -- save all four IQL network states
load_iql_checkpoint(...)   -- load IQL networks from checkpoint
resolve_iql_path(rel)      -- resolve relative checkpoint path to absolute
PROJECT_ROOT               -- repository root path
AUDIT_PATH, MANIFEST_PATH  -- frozen dataset file paths
SANITY_PATH                -- BC/CQL sanity summary path (internal reference)
SUMMARY_PATH               -- IQL sanity summary path (internal reference)
"""

import sys, os, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_envA_v2_sanity import (
    MLP, OBS_DIM, N_ACTIONS, AUDIT_PATH, MANIFEST_PATH,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

# Internal reference paths (used for pre-flight checks in downstream scripts)
SANITY_PATH  = os.path.join(PROJECT_ROOT, "artifacts", "training_sanity",
                            "envA_v2_sanity_summary.csv")
SUMMARY_PATH = os.path.join(PROJECT_ROOT, "artifacts", "training_iql",
                            "envA_v2_iql_sanity_summary.csv")

# ── Frozen IQL configuration ──────────────────────────────────────────────────

IQL_CFG = {
    "hidden_dims":  [256, 256],
    "batch_size":   256,
    "num_updates":  5000,
    "gamma":        0.99,
    "expectile":    0.7,
    "temperature":  3.0,
    "actor_lr":     3e-4,
    "critic_lr":    3e-4,
    "value_lr":     3e-4,
    "weight_decay": 1e-4,
    "target_tau":   0.005,
    "adv_clip":     100.0,
}

# ── Path helpers ──────────────────────────────────────────────────────────────

def resolve_iql_path(rel_path):
    """Normalize a relative checkpoint path to an absolute path."""
    normalized = rel_path.replace("\\", "/")
    return os.path.normpath(os.path.join(PROJECT_ROOT, normalized))

# ── IQL training loop ─────────────────────────────────────────────────────────

def train_iql(obs, acts, rews, nobs, terms, cfg, seed):
    """Discrete IQL training on EnvA_v2 (OBS_DIM=900, N_ACTIONS=4).

    Algorithm
    ---------
    1. Value network (V) trained via expectile regression on Q_min(s,a).
    2. Twin Q networks (Q1, Q2) trained via Bellman backup through V.
    3. Actor trained via advantage-weighted behavioral cloning.

    Returns
    -------
    actor, q1, q2, value_net  : trained networks
    final_actor_loss          : last actor update loss
    final_q_loss              : last Q update loss (Q1 + Q2 combined)
    final_value_loss          : last value update loss
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(obs)

    actor     = MLP(OBS_DIM, N_ACTIONS, cfg["hidden_dims"])
    q1        = MLP(OBS_DIM, N_ACTIONS, cfg["hidden_dims"])
    q2        = MLP(OBS_DIM, N_ACTIONS, cfg["hidden_dims"])
    q1_tgt    = MLP(OBS_DIM, N_ACTIONS, cfg["hidden_dims"])
    q2_tgt    = MLP(OBS_DIM, N_ACTIONS, cfg["hidden_dims"])
    value_net = MLP(OBS_DIM, 1,         cfg["hidden_dims"])

    q1_tgt.load_state_dict(q1.state_dict())
    q2_tgt.load_state_dict(q2.state_dict())

    actor_opt = optim.Adam(actor.parameters(),
                           lr=cfg["actor_lr"],  weight_decay=cfg["weight_decay"])
    q_opt     = optim.Adam(list(q1.parameters()) + list(q2.parameters()),
                           lr=cfg["critic_lr"], weight_decay=cfg["weight_decay"])
    v_opt     = optim.Adam(value_net.parameters(),
                           lr=cfg["value_lr"],  weight_decay=cfg["weight_decay"])

    obs_t   = torch.from_numpy(obs)
    acts_t  = torch.from_numpy(acts)
    rews_t  = torch.from_numpy(rews)
    nobs_t  = torch.from_numpy(nobs)
    terms_t = torch.from_numpy(terms)

    gamma       = cfg["gamma"]
    tau         = cfg["expectile"]
    temperature = cfg["temperature"]
    adv_clip    = cfg["adv_clip"]
    target_tau  = cfg["target_tau"]

    final_actor_loss = float("nan")
    final_q_loss     = float("nan")
    final_value_loss = float("nan")

    for _ in range(cfg["num_updates"]):
        idx  = np.random.randint(0, n, size=cfg["batch_size"])
        s, a, r, ns, done = (obs_t[idx], acts_t[idx], rews_t[idx],
                              nobs_t[idx], terms_t[idx])

        # 1. Value network — expectile regression
        with torch.no_grad():
            q_min_sa = torch.min(
                q1_tgt(s).gather(1, a.unsqueeze(1)).squeeze(1),
                q2_tgt(s).gather(1, a.unsqueeze(1)).squeeze(1))
        v    = value_net(s).squeeze(1)
        diff = q_min_sa - v
        wt   = torch.where(diff >= 0,
                           torch.full_like(diff, tau),
                           torch.full_like(diff, 1.0 - tau))
        value_loss = (wt * diff.pow(2)).mean()
        v_opt.zero_grad()
        value_loss.backward()
        v_opt.step()
        final_value_loss = value_loss.item()

        # 2. Twin Q networks — Bellman backup through V(s')
        with torch.no_grad():
            td_target = r + gamma * (1.0 - done) * value_net(ns).squeeze(1)
        q1_a = q1(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q2_a = q2(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q_loss = (nn.functional.mse_loss(q1_a, td_target) +
                  nn.functional.mse_loss(q2_a, td_target))
        q_opt.zero_grad()
        q_loss.backward()
        q_opt.step()
        final_q_loss = q_loss.item()

        # 3. Actor — advantage-weighted behavioral cloning
        with torch.no_grad():
            adv = (torch.min(q1_tgt(s).gather(1, a.unsqueeze(1)).squeeze(1),
                             q2_tgt(s).gather(1, a.unsqueeze(1)).squeeze(1))
                   - value_net(s).squeeze(1))
            exp_adv = torch.exp(adv / temperature).clamp(max=adv_clip)
        log_probs  = nn.functional.log_softmax(actor(s), dim=1)
        log_pi_a   = log_probs.gather(1, a.unsqueeze(1)).squeeze(1)
        actor_loss = -(exp_adv * log_pi_a).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()
        final_actor_loss = actor_loss.item()

        # Soft target update for Q networks
        for p, tp in zip(q1.parameters(), q1_tgt.parameters()):
            tp.data.copy_(target_tau * p.data + (1.0 - target_tau) * tp.data)
        for p, tp in zip(q2.parameters(), q2_tgt.parameters()):
            tp.data.copy_(target_tau * p.data + (1.0 - target_tau) * tp.data)

    return actor, q1, q2, value_net, final_actor_loss, final_q_loss, final_value_loss

# ── Checkpoint I/O ────────────────────────────────────────────────────────────

def save_iql_checkpoint(path, actor, q1, q2, value_net,
                        dataset_name, seed, cfg,
                        actor_loss, q_loss, value_loss):
    """Save all four IQL network state dicts plus metadata."""
    torch.save({
        "actor_state_dict":  actor.state_dict(),
        "q1_state_dict":     q1.state_dict(),
        "q2_state_dict":     q2.state_dict(),
        "value_state_dict":  value_net.state_dict(),
        "dataset_name":      dataset_name,
        "train_seed":        seed,
        "num_updates":       cfg["num_updates"],
        "final_actor_loss":  actor_loss,
        "final_q_loss":      q_loss,
        "final_value_loss":  value_loss,
        "obs_dim":           OBS_DIM,
        "n_actions":         N_ACTIONS,
        "config":            cfg,
    }, path)


def load_iql_checkpoint(path, obs_dim=None, cfg=None):
    """Load IQL checkpoint and reconstruct all four networks.

    If obs_dim/cfg are None, values are read from the checkpoint metadata.
    Returns (actor, q1, q2, value_net).
    """
    ckpt = torch.load(path, weights_only=False)
    _obs_dim = obs_dim if obs_dim is not None else ckpt.get("obs_dim", OBS_DIM)
    _cfg     = cfg     if cfg     is not None else ckpt["config"]
    hidden   = _cfg["hidden_dims"]

    actor     = MLP(_obs_dim, N_ACTIONS, hidden)
    q1        = MLP(_obs_dim, N_ACTIONS, hidden)
    q2        = MLP(_obs_dim, N_ACTIONS, hidden)
    value_net = MLP(_obs_dim, 1,         hidden)

    actor.load_state_dict(ckpt["actor_state_dict"])
    q1.load_state_dict(ckpt["q1_state_dict"])
    q2.load_state_dict(ckpt["q2_state_dict"])
    value_net.load_state_dict(ckpt["value_state_dict"])

    return actor, q1, q2, value_net
