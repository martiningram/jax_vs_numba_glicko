import jax.numpy as np
from jax import jit
from typing import List, Tuple


Q = np.log(10.) / 400.


@jit
def calculate_g(sigma_sq):
    """
    Calculates g as defined in the Glicko paper.
    """

    return 1. / np.sqrt(1 + (3 * Q**2 * sigma_sq / np.pi**2))


@jit
def calculate_e(theta: float, mu_j: np.ndarray,
                sigma_j_sq: np.ndarray) -> np.ndarray:
    """
    Calculates E, a quantity used in the Glicko approximations related to
    the win probability.

    Args:
        theta: The skill of player a.
        mu_j: The mean skill of each opponent j.
        sigma_j_sq: The variance of each opponent j's skill.

    Returns:
        E, as defined in the Glicko paper.
    """

    g = calculate_g(sigma_j_sq)

    return 1. / (1. + 10**(-g * (theta - mu_j) / 400.))


@jit
def calculate_delta_sq(n_j: np.ndarray, sigma_j_sq: np.ndarray, mu: float,
                       mu_j: np.ndarray) -> float:
    """
    Calculates the approximate variance of the likelihood.

    Args:
        n_j: The number of matches played against each opponent j.
        sigma_j_sq: The variance of each opponent j's skill.
        mu: The prior mean of the player.
        mu_j: The mean of each opponent j's skill.
    """

    win_exp = calculate_e(mu, mu_j, sigma_j_sq)
    win_exp_component = win_exp * (1 - win_exp)
    g = calculate_g(sigma_j_sq)
    sum_components = n_j * g**2 * win_exp_component
    summed = np.sum(sum_components)

    return 1. / (Q**2 * summed)


@jit
def calculate_single_mu_update(mu: float, opp_sigma_sq: float, s: np.ndarray,
                               opp_mu: float) -> float:
    """
    Helper function computing part of the update to mu for a single opponent.

    Args:
        mu: Player skill.
        opp_sigma_sq: Opponent skill variance.
        s: Outcomes against this opponent.
        opp_mu: Opponent skill mean.

    Returns:
        The summed quantity required for a mu update.
    """

    g = calculate_g(opp_sigma_sq)

    return np.sum(g * (s - calculate_e(mu, opp_mu, opp_sigma_sq)))


@jit
def calculate_mu_prime(mu: float, sigma_sq: float, delta_sq: float,
                       s_jk: List[np.ndarray], mu_j: np.ndarray,
                       sigma_j_sq: np.ndarray) -> float:
    """Calculates a player's posterior mean given outcomes.

    Args:
        mu: Prior mean skill of player to update.
        sigma_sq: Prior variance of player to update.
        delta_sq: The variance of the normal approximation to the likelihood.
        s_jk: The outcomes. This is a list whose j-th entry contains the
            outcomes against opponent j as a numpy array of ones and zeros.
        mu_j: Opponent mean skills.
        sigma_j_sq: Opponent skill variance.

    Returns:
        The updated mean.
    """

    pre_factor = Q / ((1. / sigma_sq) + (1. / delta_sq))

    total_update = 0.

    # This is the pesky dynamic for loop
    for j in range(len(s_jk)):

        cur_s = s_jk[j]
        cur_opp_sigma_sq = sigma_j_sq[j]
        cur_opp_mu = mu_j[j]

        total_update = total_update + calculate_single_mu_update(
            mu, cur_opp_sigma_sq, cur_s, cur_opp_mu)

    total_update = total_update * pre_factor

    return mu + total_update


@jit
def calculate_approximate_likelihood(mu: float, sigma_sq: float,
                                     n_j: np.ndarray, mu_j: np.ndarray,
                                     sigma_j_sq: np.ndarray,
                                     s_jk: List[np.ndarray]) \
        -> Tuple[float, float]:
    """
    Calculates the approximate likelihood used in Glicko for one player.

    Args:
        mu: The player's prior mean.
        sigma_sq: The player's prior variance.
        n_j: The number of games played against each opponent j.
        mu_j: The skill of each opponent j.
        sigma_j_sq: The variance of each opponent j's skill.
        s_jk: The outcomes against each opponent j.

    Returns:
        The mean and variance of the normal likelihood approximation, as a
        tuple.
    """

    delta_sq = calculate_delta_sq(n_j, sigma_j_sq, mu, mu_j)
    mu_prime = calculate_mu_prime(mu, sigma_sq, delta_sq, s_jk, mu_j,
                                  sigma_j_sq)

    gamma = sigma_sq / (sigma_sq + delta_sq)
    theta_hat = (1. / gamma) * (mu_prime - mu * (1 - gamma))

    return theta_hat, delta_sq
