import numpy as np
import torch


def def_equbin(
    s_distr: torch.tensor, 
    binsize=None, 
    bin_num: int=None, 
    hist_edge_l: int=-14) -> np.ndarray:

    """
    Flat-s binning with overflow bin on the left side. Only the number of events counts, weights do not contribute.

    Args:
        s_distr (torch.tensor): Tensor representing all signal events.
        hist_edge_l (int): Integer representing the left most edge of the binned histogram

    Returns:
        1D array containing the exact bin edges from hist_edge_l to the maximum value.
    """

    s_distr_filtered = s_distr[s_distr > hist_edge_l + 1]
    distr_size = len(s_distr_filtered)

    bin_size = distr_size // bin_num
    odd_bin_size = distr_size % bin_num

    args = s_distr_filtered.argsort()

    # Die Limits starten am Anfang der regulären Bins (nach dem odd_bin)
    limit_indices = np.arange(bin_num) * bin_size + odd_bin_size
    
    # Setze das Array direkt zusammen: [hist_edge_l] + [reguläre Limits] + [Maximum]
    bins_limits = np.concatenate((
        [hist_edge_l], 
        s_distr_filtered[args[limit_indices]], 
        [s_distr_filtered[args[-1]]]
    ))

    return bins_limits