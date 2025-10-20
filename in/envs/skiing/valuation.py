import torch as th

import torch as th
from nsfr.utils.common import bool_to_probs  # kept for compatibility (used in a few spots)

# --- helper: get x,y from a single-object tensor [B,F] ---
def _xy(z: th.Tensor):
    return z[:, -2], z[:, -1]  # x, y

# --- tunables (adjust to taste / scaling) ---
PAIR_Y_TOL   = 6.0     # how close in Y two flags must be to be considered one gate (same row)
AHEAD_NEAR   = 6.0     # minimal "ahead" distance (flag must be above player by at least this much)
AHEAD_FAR    = 80.0    # maximal "ahead" window
CENTER_TOL_X = 2.5     # deadband for being 'centered' on the gate
TEMP         = 3.0     # sigmoid sharpness (↑ = crisper decisions)
CENTER_GAIN  = 4.0     # strength of NOOP preference when centered

# -------------------------
# Basic pairwise predicates
# -------------------------
def flag_ahead(z_flag: th.Tensor, z_player: th.Tensor) -> th.Tensor:
    """
    True-ish if flag is ahead (above) of the player in the y-range window.
    NOTE: ALE coords usually increase downward. 'Ahead' → yf < yp.
    """
    _, yf = _xy(z_flag)
    _, yp = _xy(z_player)
    diff = yp - yf  # positive if flag is above (ahead)
    in_front = (diff > AHEAD_NEAR) & (diff < AHEAD_FAR)
    return bool_to_probs(in_front)

def same_row_flags(z_f1: th.Tensor, z_f2: th.Tensor) -> th.Tensor:
    """True-ish if two flags are roughly on the same horizontal row (a gate pair)."""
    _, y1 = _xy(z_f1)
    _, y2 = _xy(z_f2)
    return bool_to_probs((y1 - y2).abs() <= PAIR_Y_TOL)

# ---------------------------------
# Smooth steering with center bias
# ---------------------------------
def _center_margin(cx: th.Tensor, xp: th.Tensor,
                   tol: float = CENTER_TOL_X, temp: float = TEMP) -> th.Tensor:
    """
    Returns ~1.0 when |cx - xp| << tol (well-centered), ~0.0 when far.
    """
    return th.sigmoid(temp * (tol - (cx - xp).abs()))

def _damped_sigmoid(pos: th.Tensor, damp: th.Tensor, temp: float = TEMP) -> th.Tensor:
    """
    Sigmoid(pos) scaled by 'damp' to suppress lateral moves near center.
    """
    return th.sigmoid(temp * pos) * damp

def center_right_of_player(z_f1: th.Tensor, z_f2: th.Tensor, z_player: th.Tensor) -> th.Tensor:
    """
    Smooth 'right' preference: increases as (cx - xp - tol) grows,
    but is *suppressed* near center to let NOOP dominate.
    """
    x1, _ = _xy(z_f1); x2, _ = _xy(z_f2); xp, _ = _xy(z_player)
    cx = 0.5 * (x1 + x2)
    margin = _center_margin(cx, xp)                      # ~1 near center, 0 far
    damp   = 1.0 - margin                                # allow turns only when not centered
    return _damped_sigmoid((cx - xp) - CENTER_TOL_X, damp)

def center_left_of_player(z_f1: th.Tensor, z_f2: th.Tensor, z_player: th.Tensor) -> th.Tensor:
    """
    Smooth 'left' preference: increases as (xp - cx - tol) grows,
    but is *suppressed* near center to let NOOP dominate.
    """
    x1, _ = _xy(z_f1); x2, _ = _xy(z_f2); xp, _ = _xy(z_player)
    cx = 0.5 * (x1 + x2)
    margin = _center_margin(cx, xp)
    damp   = 1.0 - margin
    return _damped_sigmoid((xp - cx) - CENTER_TOL_X, damp)

def centered_on_gate(z_f1: th.Tensor, z_f2: th.Tensor, z_player: th.Tensor) -> th.Tensor:
    """
    Strong NOOP preference when centered:
    - near center: ~CENTER_GAIN (after later normalization in the rule head this wins)
    - far from center: ~0 (so left/right can take over)
    """
    x1, _ = _xy(z_f1); x2, _ = _xy(z_f2); xp, _ = _xy(z_player)
    cx = 0.5 * (x1 + x2)
    return CENTER_GAIN * _center_margin(cx, xp)

# ------------------------------------------------------
# Single-flag fallback (if only one flag is detectable)
# ------------------------------------------------------
def flag_right_of_player(z_flag: th.Tensor, z_player: th.Tensor) -> th.Tensor:
    """
    Prefer RIGHT if single flag is right of player beyond tolerance,
    damped near alignment to avoid jitter.
    """
    xf, _ = _xy(z_flag); xp, _ = _xy(z_player)
    margin = _center_margin(xf, xp)
    damp   = 1.0 - margin
    return _damped_sigmoid((xf - xp) - CENTER_TOL_X, damp)

def flag_left_of_player(z_flag: th.Tensor, z_player: th.Tensor) -> th.Tensor:
    """Prefer LEFT if single flag is left of player beyond tolerance, damped near center."""
    xf, _ = _xy(z_flag); xp, _ = _xy(z_player)
    margin = _center_margin(xf, xp)
    damp   = 1.0 - margin
    return _damped_sigmoid((xp - xf) - CENTER_TOL_X, damp)

def horizontally_aligned(z_flag: th.Tensor, z_player: th.Tensor) -> th.Tensor:
    """NOOP weight near alignment with single flag (center-biased)."""
    xf, _ = _xy(z_flag); xp, _ = _xy(z_player)
    return CENTER_GAIN * _center_margin(xf, xp)

# ------------------------------------------------------
# Gate validity (same row & ahead)
# ------------------------------------------------------
def valid_gate_pair(z_f1: th.Tensor, z_f2: th.Tensor, z_player: th.Tensor) -> th.Tensor:
    """
    Candidate 'next gate' for the player: both flags ahead and on the same row.
    (Uses boolean-to-prob to keep this strictly gating; steerings are smooth.)
    """
    return flag_ahead(z_f1, z_player) * flag_ahead(z_f2, z_player) * same_row_flags(z_f1, z_f2)
