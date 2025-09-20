# sitecustomize.py — auto-imported by Python at startup
import time, random
try:
    import numpy as np
except Exception:
    np = None

seed = int(time.time()) & 0xFFFFFFFF  # new seed every run
random.seed(seed)
if np is not None:
    np.random.seed(seed)
print(f"[sitecustomize] Random seed set to {seed}")
