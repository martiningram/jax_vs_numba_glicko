using BenchmarkTools

include("glicko.jl")

function generate_random_observations()
    # Draw a random number of opponents
    num_opponents = rand(1:5)
    #num_opponents = 5

    # Draw a random number of matches played against each opponent
    n_j = rand(1:5, num_opponents)
    #n_j = fill(5, num_opponents)

    # Draw random mean for each opponent
    mu_j = randn(num_opponents) .* 50 .+ 1500

    # Draw random variances for each opponent
    sigma_j_sq = (randn(num_opponents) .* 5 .+ 20) .^ 2

    # Draw random outcomes (win/loss) for each opponent
    s_jk = [rand(1:2, x) for x in n_j]

    return n_j, mu_j, sigma_j_sq, s_jk
end

mu = 1500.
sigma_sq = 100. ^ 2

@btime n_j, mu_j, sigma_j_sq, s_jk = generate_random_observations()

@btime begin
    n_j, mu_j, sigma_j_sq, s_jk = generate_random_observations()
    calculate_approximate_likelihood(mu, sigma_sq, n_j, mu_j, sigma_j_sq, s_jk)
end
