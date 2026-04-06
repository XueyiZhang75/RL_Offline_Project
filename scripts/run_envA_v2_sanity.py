"""
scripts/run_envA_v2_sanity.py
Shared training infrastructure for EnvA_v2 discrete experiments.

Defines the frozen BC/CQL training framework (configs, MLP, dataset loader,
training functions, evaluator) reused by all subsequent EnvA_v2 experiment
scripts.  All downstream scripts import directly from this module and must
not modify these frozen definitions.

Exports
-------
BC_CFG, CQL_CFG           -- frozen algorithm hyperparameters
OBS_DIM, N_ACTIONS         -- environment constants (900-d one-hot, 4 actions)
EVAL_EPISODES              -- standard evaluation episode count
MLP                        -- shared neural network architecture
encode_obs(raw_obs)        -- encode (N,2) int32 pos array → float32 (N,900)
encode_single(r, c)        -- encode one grid position → FloatTensor (900,)
load_dataset(name)         -- load frozen .npz and return encoded tensors
train_bc(obs, acts, cfg, seed)                    -- BC training loop
train_cql(obs, acts, rews, nobs, terms, cfg, seed) -- CQL training loop
evaluate(model, algo)      -- online greedy evaluation on EnvA_v2
SUMMARY_COLUMNS            -- CSV column order for all experiment summaries
AUDIT_PATH, MANIFEST_PATH  -- frozen dataset audit / manifest file paths
"""

import sys, os, csv, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from envs.gridworld_envs import EnvA_v2, HORIZON, N_ACTIONS

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR   = os.path.join(PROJECT_ROOT, "artifacts", "final_datasets")
AUDIT_PATH    = os.path.join(DATASET_DIR, "final_dataset_audit.csv")
MANIFEST_PATH = os.path.join(DATASET_DIR, "final_dataset_manifest.csv")

# ── Frozen environment constants ──────────────────────────────────────────────

_GRID_SIZE   = EnvA_v2.grid_size          # 30
OBS_DIM      = _GRID_SIZE * _GRID_SIZE    # 900
EVAL_EPISODES = 100

# ── Frozen algorithm configurations ──────────────────────────────────────────

BC_CFG = {
    "hidden_dims":  [256, 256],
    "num_updates":  5000,
    "batch_size":   256,
    "lr":           3e-4,
    "weight_decay": 1e-4,
}

CQL_CFG = {
    "hidden_dims":  [256, 256],
    "num_updates":  5000,
    "batch_size":   256,
    "lr":           3e-4,
    "weight_decay": 1e-4,
    "gamma":        0.99,
    "cql_alpha":    1.0,
}

# ── Summary CSV column order (shared by main, quality, validation) ────────────

SUMMARY_COLUMNS = [
    "dataset_name", "algorithm", "train_seed",
    "num_updates", "final_train_loss",
    "eval_episodes", "avg_return", "success_rate", "avg_episode_length",
    "checkpoint_path", "status", "notes",
]

# ── Neural network ────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Feedforward network shared by BC, CQL, and IQL (actor/critic/value)."""

    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ── Observation encoding ──────────────────────────────────────────────────────

def encode_obs(raw_obs):
    """Convert (N, 2) int32 (row, col) array to (N, OBS_DIM) float32 one-hot."""
    n = len(raw_obs)
    out = np.zeros((n, OBS_DIM), dtype=np.float32)
    rows = raw_obs[:, 0].astype(np.int32)
    cols = raw_obs[:, 1].astype(np.int32)
    idx  = rows * _GRID_SIZE + cols
    out[np.arange(n), idx] = 1.0
    return out


def encode_single(r, c):
    """Return a (OBS_DIM,) FloatTensor one-hot for grid position (r, c)."""
    v = torch.zeros(OBS_DIM, dtype=torch.float32)
    v[int(r) * _GRID_SIZE + int(c)] = 1.0
    return v

# ── Dataset loader ────────────────────────────────────────────────────────────

def load_dataset(name):
    """Load frozen .npz dataset and return encoded float32 arrays.

    Returns
    -------
    obs   : float32 (N, OBS_DIM) one-hot observations
    acts  : int64   (N,)         actions
    rews  : float32 (N,)         rewards
    nobs  : float32 (N, OBS_DIM) one-hot next-observations
    terms : float32 (N,)         terminal flags (0 or 1)
    """
    path = os.path.join(DATASET_DIR, f"{name}.npz")
    d    = np.load(path)
    obs  = encode_obs(d["observations"])
    nobs = encode_obs(d["next_observations"])
    acts = d["actions"].astype(np.int64)
    rews = d["rewards"].astype(np.float32)
    terms = d["terminals"].astype(np.float32)
    return obs, acts, rews, nobs, terms

# ── Training loops ────────────────────────────────────────────────────────────

def train_bc(obs, acts, cfg, seed):
    """Behavior Cloning: supervised cross-entropy on (obs, acts) pairs.

    Returns (model, final_train_loss).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    n     = len(obs)
    model = MLP(OBS_DIM, N_ACTIONS, cfg["hidden_dims"])
    opt   = optim.Adam(model.parameters(),
                       lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    obs_t  = torch.from_numpy(obs)
    acts_t = torch.from_numpy(acts)
    final_loss = float("nan")
    for _ in range(cfg["num_updates"]):
        idx    = np.random.randint(0, n, size=cfg["batch_size"])
        logits = model(obs_t[idx])
        loss   = loss_fn(logits, acts_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = loss.item()
    return model, final_loss


def train_cql(obs, acts, rews, nobs, terms, cfg, seed):
    """Conservative Q-Learning for discrete actions.

    Returns (q_network, final_train_loss).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    n         = len(obs)
    q_net     = MLP(OBS_DIM, N_ACTIONS, cfg["hidden_dims"])
    target_net = MLP(OBS_DIM, N_ACTIONS, cfg["hidden_dims"])
    target_net.load_state_dict(q_net.state_dict())
    opt = optim.Adam(q_net.parameters(),
                     lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    obs_t  = torch.from_numpy(obs)
    acts_t = torch.from_numpy(acts)
    rews_t = torch.from_numpy(rews)
    nobs_t = torch.from_numpy(nobs)
    terms_t = torch.from_numpy(terms)
    gamma = cfg["gamma"]
    alpha = cfg["cql_alpha"]
    final_loss = float("nan")
    for step in range(cfg["num_updates"]):
        idx  = np.random.randint(0, n, size=cfg["batch_size"])
        s, a, r, ns, done = (obs_t[idx], acts_t[idx], rews_t[idx],
                              nobs_t[idx], terms_t[idx])
        with torch.no_grad():
            max_q_next = target_net(ns).max(dim=1).values
            td_target  = r + gamma * (1.0 - done) * max_q_next
        q_vals = q_net(s)
        q_a    = q_vals.gather(1, a.unsqueeze(1)).squeeze(1)
        td_loss     = nn.functional.mse_loss(q_a, td_target)
        cql_penalty = (torch.logsumexp(q_vals, dim=1) - q_a).mean()
        loss = td_loss + alpha * cql_penalty
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = loss.item()
        if (step + 1) % 100 == 0:
            for p, tp in zip(q_net.parameters(), target_net.parameters()):
                tp.data.copy_(0.995 * tp.data + 0.005 * p.data)
    return q_net, final_loss

# ── Online evaluation ─────────────────────────────────────────────────────────

def evaluate(model, algo, n_episodes=EVAL_EPISODES):
    """Run greedy policy on EnvA_v2 and return performance metrics.

    Parameters
    ----------
    model  : trained MLP (BC policy or CQL Q-network)
    algo   : "bc" or "cql"
    Returns dict with avg_return, success_rate, avg_episode_length.
    """
    env = EnvA_v2()
    returns, successes, lengths = [], [], []
    model.eval()
    with torch.no_grad():
        for ep in range(n_episodes):
            obs_raw, _ = env.reset()
            r_total, ep_len, done, success = 0.0, 0, False, False
            while not done:
                s_t = encode_single(*obs_raw).unsqueeze(0)
                logits = model(s_t)
                action = int(logits.argmax(dim=1).item())
                obs_raw, r, terminated, truncated, _ = env.step(action)
                r_total += r
                ep_len  += 1
                done     = terminated or truncated
                if terminated:
                    success = True
            returns.append(r_total)
            successes.append(float(success))
            lengths.append(ep_len)
    model.train()
    return {
        "avg_return":         float(np.mean(returns)),
        "success_rate":       float(np.mean(successes)),
        "avg_episode_length": float(np.mean(lengths)),
    }
