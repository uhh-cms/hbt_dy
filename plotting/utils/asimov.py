import numpy as np 

def asimov_no_background(
    s: np.ndarray,
    b: np.ndarray,
    eps_b: float=1e-9,
    eps_sqrt: float=1e-9,
    *args,
    **kwargs
    ) -> np.ndarray:
    """
    Asimov Significance for the case when background uncertainty is set to 0.
    Approximation coming from asimov for no background uncertainty: https://arxiv.org/abs/1806.00322 eq. 3.2
    This approximation is unstable for two cases, and thus certain epsilons are introduced to stabilize  it.:
    It is unstable for no-background (b=0) regions, where *eps_log* helps.
    But also when s > (s+b)(ln(1+s/b)), for which *eps_sqrt* increases stability and lower bound the significance.

    Args:
        s (np.ndarray): Array representing signal in bins.
        b (np.ndarray): Array representing background in bins.
        log (float, optional): _description_. Defaults to 0.
        eps_sqrt (float, optional): _description_. Defaults to 1e-9.

    Returns:
        np.ndarray: Asimov Significance without background uncertainty
    """

    # equivalent to:
    # np.sqrt(2 * ((s + b) * (np.log(s+b + eps_log / 2) - np.log(b + eps_log / 2) -s) + eps_sqrt)
    b = b + eps_b
    s = s + eps_sqrt
    # return np.sqrt(
    #     2 * ((s + b) * np.log(1 + s / (b + eps_log)) - s) + eps_sqrt
    #     )
    return np.sqrt(
        2 * ((s + b) * np.log(1 + s / b) - s)
        )