const Q = log(10.) / 400.

"""
Calculates g as defined in the Glicko paper.
"""
function calculate_g(sigma_sq)
    return 1. / sqrt(1 + (3 * Q^2 * sigma_sq / pi^2))
end

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
function calculate_e(theta, mu_j, sigma_j_sq)
    g = calculate_g(sigma_j_sq)

    return 1. / (1. + 10^(-g * (theta - mu_j) / 400.))
end

"""
Calculates the approximate variance of the likelihood.

Args:
    n_j: The number of matches played against each opponent j.
    sigma_j_sq: The variance of each opponent j's skill.
    mu: The prior mean of the player.
    mu_j: The mean of each opponent j's skill.
"""
function calculate_delta_sq(n_j, sigma_j_sq, mu, mu_j)
    win_exp = calculate_e.(mu, mu_j, sigma_j_sq)
    win_exp_component = win_exp .* (1 .- win_exp)
    g = calculate_g.(sigma_j_sq)
    sum_components = n_j .* g.^2 .* win_exp_component
    summed = sum(sum_components)

    return 1. / (Q^2 * summed)
end

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
function calculate_single_mu_update(mu, opp_sigma_sq, s, opp_mu)
    g = calculate_g.(opp_sigma_sq)

    return sum(g .* (s .- calculate_e.(mu, opp_mu, opp_sigma_sq)))
end

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
function calculate_mu_prime(mu, sigma_sq, delta_sq, s_jk, mu_j, sigma_j_sq)

    pre_factor = Q ./ ((1. ./ sigma_sq) .+ (1. ./ delta_sq))

    total_update = 0.

    # This is the pesky dynamic for loop
    for j in 1:length(s_jk)
        cur_s = s_jk[j]
        cur_opp_sigma_sq = sigma_j_sq[j]
        cur_opp_mu = mu_j[j]

        total_update += calculate_single_mu_update(
            mu, cur_opp_sigma_sq, cur_s, cur_opp_mu)
    end

    total_update *= pre_factor

    return mu + total_update
end

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
function calculate_approximate_likelihood(mu, sigma_sq, n_j, mu_j, sigma_j_sq, s_jk)

    delta_sq = calculate_delta_sq(n_j, sigma_j_sq, mu, mu_j)
    mu_prime = calculate_mu_prime(mu, sigma_sq, delta_sq, s_jk, mu_j,
                                  sigma_j_sq)

    gamma = sigma_sq / (sigma_sq + delta_sq)
    theta_hat = (1. / gamma) * (mu_prime - mu * (1 - gamma))

    return theta_hat, delta_sq
end
