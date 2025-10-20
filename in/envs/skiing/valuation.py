import torch as th

from nsfr.utils.common import bool_to_probs

# --- helper: get x,y from a single-object tensor [B,F] ---
def _xy(z: th.Tensor):
    return z[:, -2], z[:, -1]  # x, y

# --- visibility / geometry tunables (tweak to your OCAtari scaling) ---
PAIR_Y_TOL   = 6.0    # how close in Y two flags must be to be considered a pair (same gate)
AHEAD_NEAR   = 6.0    # minimal "ahead" distance: flag y must be at least this much above the player
AHEAD_FAR    = 80.0   # maximal "ahead" window
CENTER_TOL_X = 2.0    # if |player_x - center_x| <= this, we consider "centered"

# -------------------------
# Basic pairwise predicates
# -------------------------
def flag_ahead(z_flag: th.Tensor, z_player: th.Tensor) -> th.Tensor:
    """True if flag is slightly ahead (above) the player in the y-range window."""
    _, yf = _xy(z_flag)
    _, yp = _xy(z_player)
    diff = yf - yp  # >0 if flag below in screen coords? In ALE Skiing, larger y is lower.
    # We want "ahead" = smaller y than player (above), so invert if your axis differs.
    # If your coords are (0 at top, increasing downward), then 'ahead' means yf < yp.
    # Toggle the line below if your coord convention is the other way around.
    ahead = (yp - yf > AHEAD_NEAR) & (yp - yf < AHEAD_FAR)
    return bool_to_probs(ahead)

def same_row_flags(z_f1: th.Tensor, z_f2: th.Tensor) -> th.Tensor:
    """True if two flags are roughly on the same horizontal row (gate pair)."""
    _, y1 = _xy(z_f1)
    _, y2 = _xy(z_f2)
    return bool_to_probs((y1 - y2).abs() <= PAIR_Y_TOL)

# ---------------------------------
# Gate center vs player positioning
# ---------------------------------
def center_right_of_player(z_f1: th.Tensor, z_f2: th.Tensor, z_player: th.Tensor) -> th.Tensor:
    """True if gate center x is to the RIGHT of the player -> take RIGHT action."""
    x1, _ = _xy(z_f1)
    x2, _ = _xy(z_f2)
    xp, _ = _xy(z_player)
    cx = 0.5 * (x1 + x2)
    return bool_to_probs(cx > xp + CENTER_TOL_X)

def center_left_of_player(z_f1: th.Tensor, z_f2: th.Tensor, z_player: th.Tensor) -> th.Tensor:
    """True if gate center x is to the LEFT of the player -> take LEFT action."""
    x1, _ = _xy(z_f1)
    x2, _ = _xy(z_f2)
    xp, _ = _xy(z_player)
    cx = 0.5 * (x1 + x2)
    return bool_to_probs(cx < xp - CENTER_TOL_X)

def centered_on_gate(z_f1: th.Tensor, z_f2: th.Tensor, z_player: th.Tensor) -> th.Tensor:
    """True if player is already near the gate center -> NOOP (or slight trim)."""
    x1, _ = _xy(z_f1)
    x2, _ = _xy(z_f2)
    xp, _ = _xy(z_player)
    cx = 0.5 * (x1 + x2)
    return bool_to_probs((cx - xp).abs() <= CENTER_TOL_X)

# ------------------------------------------------------
# Single-flag fallback (if only one flag is detectable)
# ------------------------------------------------------
def flag_right_of_player(z_flag: th.Tensor, z_player: th.Tensor) -> th.Tensor:
    """If a single flag is to the right of player -> go RIGHT."""
    xf, _ = _xy(z_flag)
    xp, _ = _xy(z_player)
    return bool_to_probs(xf > xp + CENTER_TOL_X)

def flag_left_of_player(z_flag: th.Tensor, z_player: th.Tensor) -> th.Tensor:
    """If a single flag is to the left of player -> go LEFT."""
    xf, _ = _xy(z_flag)
    xp, _ = _xy(z_player)
    return bool_to_probs(xf < xp - CENTER_TOL_X)

def horizontally_aligned(z_flag: th.Tensor, z_player: th.Tensor) -> th.Tensor:
    """Already aligned with the single visible flag -> NOOP."""
    xf, _ = _xy(z_flag)
    xp, _ = _xy(z_player)
    return bool_to_probs((xf - xp).abs() <= CENTER_TOL_X)

# ------------------------------------------------------
# Convenience predicates to ensure we’re using NEXT gate
# (both flags ahead of player & on roughly same row)
# ------------------------------------------------------
def valid_gate_pair(z_f1: th.Tensor, z_f2: th.Tensor, z_player: th.Tensor) -> th.Tensor:
    """Two flags form a candidate 'next gate' for the player."""
    return flag_ahead(z_f1, z_player) * flag_ahead(z_f2, z_player) * same_row_flags(z_f1, z_f2)
