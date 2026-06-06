"""Historical PushT contact fallback used by Proposal 1 prelim scripts."""

from __future__ import annotations

from typing import Any


def contact_blocks_from_state(states: Any, frameskip: int) -> Any:
    import numpy as np

    if states is None:
        raise KeyError("Cannot compute geometric contacts without state")
    contacts_obs = geometric_contacts_from_state(np.asarray(states))
    if len(contacts_obs) < 2:
        return np.zeros(0, dtype=np.float32)
    blocks = np.maximum(contacts_obs[:-1], contacts_obs[1:])
    return blocks.astype(np.float32)


def geometric_contacts_from_state(states: Any, *, agent_radius: float = 15.0, margin: float = 2.0) -> Any:
    import numpy as np

    s = np.asarray(states, dtype=np.float64)
    agent = s[:, 0:2]
    block = s[:, 2:4]
    angle = s[:, 4]
    delta = agent - block
    c = np.cos(-angle)
    si = np.sin(-angle)
    x = c * delta[:, 0] - si * delta[:, 1]
    y = si * delta[:, 0] + c * delta[:, 1]

    rects = [
        (-60.0, 60.0, -45.0, -15.0),
        (-15.0, 15.0, -15.0, 75.0),
    ]
    min_dist = np.full(len(s), np.inf, dtype=np.float64)
    for xmin, xmax, ymin, ymax in rects:
        dx = np.maximum(np.maximum(xmin - x, 0.0), x - xmax)
        dy = np.maximum(np.maximum(ymin - y, 0.0), y - ymax)
        min_dist = np.minimum(min_dist, np.sqrt(dx * dx + dy * dy))
    return min_dist <= (agent_radius + margin)
