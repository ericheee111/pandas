import numpy as np

def group_prod_native_float(
    out: np.ndarray,
    counts: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    min_count: int,
    skipna: bool,
) -> bool: ...
