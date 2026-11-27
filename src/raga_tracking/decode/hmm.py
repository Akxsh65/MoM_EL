"""D3: HMM with minimum-duration chained sub-states. Offline Viterbi, online forward filtering.

State (r, j), j = 0..m-1: inside a chain move forward w.p. 1; from (r, m-1) stay w.p. 1-eps or
jump to (r', 0), r' != r, w.p. eps/(K-1). Emission log-score for every sub-state of r is
log p(r|x) - log prior(r); silent windows give uniform (zero) emission. Log-space, numpy-vectorised.
The path must start at some (r, 0), so the first segment also lasts >= m windows.
"""
from __future__ import annotations

import numpy as np

from .common import labels_to_segments


def emissions(post: np.ndarray, valid: np.ndarray, prior: np.ndarray | None = None,
              beta: float = 1.0) -> np.ndarray:
    """(T,K) emission log-scores beta*log p - log prior; zero rows for invalid windows."""
    K = post.shape[1]
    prior = np.full(K, 1.0 / K) if prior is None else np.asarray(prior)
    e = beta * np.log(np.clip(post, 1e-12, 1)) - np.log(prior)[None, :]
    e[~np.asarray(valid, bool)] = 0.0
    return e


def _best_other(x: np.ndarray):
    """For each r: (max over r' != r of x[r'], argmax). Requires K >= 2."""
    order = np.argsort(x)
    top, sec = order[-1], order[-2]
    idx = np.arange(len(x))
    val = np.where(idx == top, x[sec], x[top])
    arg = np.where(idx == top, sec, top)
    return val, arg


def viterbi(em: np.ndarray, m: int, eps: float) -> np.ndarray:
    """Best-path raga label per window, shape (T,)."""
    T, K = em.shape
    ls, lj = np.log1p(-eps), np.log(eps / max(K - 1, 1))
    delta = np.full((K, m), -np.inf)
    delta[:, 0] = em[0]
    arg_from = np.zeros((T, K), dtype=np.int32)  # raga we jumped from when entering (r, 0)
    stayed = np.zeros((T, K), dtype=bool)        # last sub-state reached by staying
    for t in range(1, T):
        last = delta[:, m - 1]
        jb, arg = _best_other(last) if K > 1 else (np.full(1, -np.inf), np.zeros(1, int))
        arg_from[t] = arg
        new = np.full((K, m), -np.inf)
        stay = last + ls
        if m == 1:
            new[:, 0] = np.maximum(stay, jb + lj)
            stayed[t] = stay >= jb + lj
        else:
            new[:, 0] = jb + lj
            new[:, 1:] = delta[:, :-1]
            adv = delta[:, m - 2]
            stayed[t] = stay > adv
            new[:, m - 1] = np.maximum(stay, adv)
        delta = new + em[t][:, None]
    r, j = divmod(int(np.argmax(delta)), m)
    labels = np.zeros(T, dtype=int)
    for t in range(T - 1, -1, -1):
        labels[t] = r
        if t == 0:
            break
        if m == 1:
            if not stayed[t, r]:
                r = int(arg_from[t, r])
        elif j == 0:
            r, j = int(arg_from[t, r]), m - 1
        elif j == m - 1:
            j = m - 1 if stayed[t, r] else m - 2
        else:
            j -= 1
    return labels


def forward_filter(em: np.ndarray, m: int, eps: float) -> np.ndarray:
    """Causal forward filtering: row t of the (T,K) raga marginals uses em[:t+1] only."""
    T, K = em.shape
    ls, lj = np.log1p(-eps), np.log(eps / max(K - 1, 1))
    a = np.full((K, m), -np.inf)
    a[:, 0] = em[0]
    out = np.zeros((T, K))
    for t in range(T):
        if t > 0:
            last = a[:, m - 1]
            if K > 1:
                other = np.array([np.logaddexp.reduce(np.delete(last, r)) for r in range(K)])
            else:
                other = np.full(1, -np.inf)
            new = np.full((K, m), -np.inf)
            if m == 1:
                new[:, 0] = np.logaddexp(last + ls, other + lj)
            else:
                new[:, 0] = other + lj
                new[:, 1:] = a[:, :-1]
                new[:, m - 1] = np.logaddexp(new[:, m - 1], last + ls)
            a = new + em[t][:, None]
        a = a - np.logaddexp.reduce(a, axis=None)
        out[t] = np.exp(np.logaddexp.reduce(a, axis=1))
    return out


def decode_d3(post, valid, centres, hop, eps: float = 1e-2, min_dur: float = 45.0,
              prior=None, beta: float = 1.0, online: bool = False):
    """D3 decoder. Returns (intervals, labels)."""
    m = max(1, int(round(min_dur / hop)))
    em = emissions(post, valid, prior, beta)
    labels = forward_filter(em, m, eps).argmax(1) if online else viterbi(em, m, eps)
    return labels_to_segments(labels, centres, hop)
