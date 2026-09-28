# ENSEMBLE DISCRETIZATION

function bin_means_and_probs(dist, edges)    # 把一个连续分布离散化为若干个bin，计算每个bin的平均值和概率
    length(edges) >= 2 || error("edges must contain at least two values.")
    all(isfinite, edges) || error("All bin edges must be finite.")
    all(diff(edges) .> 0) || error("Bin edges must be strictly increasing.")

    Mloc = length(edges) - 1
    probs = zeros(Float64, Mloc)
    means = zeros(Float64, Mloc)

    for j in 1:Mloc
        low, high = edges[j], edges[j+1]
        pj = cdf(dist, high) - cdf(dist, low)
        probs[j] = pj

        if pj > 0
            num, _ = quadgk(x -> x * pdf(dist, x), low, high)
            means[j] = num / pj
        else
            means[j] = 0.0
        end
    end

    return means, probs
end

function build_2d_bins(N, delta_b_1d, p_delta, g_b_1d, p_g)
    M_delta = length(delta_b_1d)
    M_g = length(g_b_1d)

    M_delta > 0 || error("delta_b_1d cannot be empty.")
    M_g > 0 || error("g_b_1d cannot be empty.")
    length(p_delta) == M_delta ||
        error("length(p_delta) must equal length(delta_b_1d).")
    length(p_g) == M_g ||
        error("length(p_g) must equal length(g_b_1d).")
    N >= 0 || error("N must be non-negative.")
    all(isfinite, delta_b_1d) || error("All detuning bins must be finite.")
    all(isfinite, g_b_1d) || error("All coupling bins must be finite.")
    all(x -> isfinite(x) && x >= 0, p_delta) ||
        error("All detuning probabilities must be finite and non-negative.")
    all(x -> isfinite(x) && x >= 0, p_g) ||
        error("All coupling probabilities must be finite and non-negative.")

    # Physical assumption: detuning and coupling are statistically independent,
    # so their joint probability factorizes as p(delta, g) = p_delta * p_g.
    weights_2d = Float64.(p_delta) * transpose(Float64.(p_g))
    Nj_2d = Float64(N) .* weights_2d

    # g varies fastest within each detuning bin.
    delta_flat = repeat(Float64.(delta_b_1d); inner=M_g)
    g_flat = repeat(Float64.(g_b_1d); outer=M_delta)
    Nj_flat = vec(permutedims(Nj_2d))

    return Nj_flat, delta_flat, g_flat, sum(Nj_flat), Nj_2d
end


function build_ensemble(N, freq_inhomogeneity, g_inhomogeneity, M_delta, M_g)
    edges_delta, delta_b_1d, p_delta, freq_info =
        build_frequency_bins(freq_inhomogeneity, M_delta)

    edges_g, g_b_1d, p_g, g_mean, g_std, g2_avg, g_info =
        build_coupling_bins(g_inhomogeneity, M_g)

    Nj, delta_b, g_b, N_total, Nj_2d =
        build_2d_bins(N, delta_b_1d, p_delta, g_b_1d, p_g)

    return (
        M_delta = M_delta,
        M_g = M_g,
        M = M_delta * M_g,
        freq_inhomogeneity = freq_inhomogeneity,
        freq_info = freq_info,
        FWHM = get(freq_info, :FWHM, 0.0),
        g_inhomogeneity = g_inhomogeneity,
        g_info = g_info,
        g_mean = g_mean,
        g_std = g_std,
        g2_avg = g2_avg,
        N = N,
        N_total = N_total,
        Nj = Nj,
        delta_b = delta_b,
        g_b = g_b,
        edges_delta = edges_delta,
        delta_b_1d = delta_b_1d,
        p_delta = p_delta,
        edges_g = edges_g,
        g_b_1d = g_b_1d,
        p_g = p_g,
        Nj_2d = Nj_2d,
    )
end

function _truncated_quantile_grid(distribution, low, high, count)
    count isa Integer && count > 0 || error("Sample count must be positive.")
    low isa Real && high isa Real && isfinite(low) && isfinite(high) && low < high ||
        error("Sampling bounds must be finite with low < high.")
    p_low, p_high = cdf(distribution, low), cdf(distribution, high)
    p_high > p_low || error("Sampling range has zero probability.")
    edge_probabilities = range(p_low, p_high; length=count + 1)
    midpoint_probabilities = range(p_low, p_high; length=2count + 1)[2:2:end]
    edges = quantile.(Ref(distribution), edge_probabilities)
    values = quantile.(Ref(distribution), midpoint_probabilities)
    return Float64.(edges), Float64.(values)
end

function _equal_weight_frequency_samples(config, count)
    validate_frequency_inhomogeneity(config)
    if config.kind == :constant
        count == 1 || error("Constant frequency inhomogeneity requires M_delta = 1.")
        value = Float64(config.delta_value)
        return [value, value], [value]
    elseif config.kind == :gaussian
        distribution = build_frequency_distribution(config)
        sigma = config.FWHM / (2sqrt(2log(2)))
        bound = config.span_sigma * sigma
        return _truncated_quantile_grid(distribution, -bound, bound, count)
    elseif config.kind == :lorentzian
        distribution = build_frequency_distribution(config)
        bound = config.span_gamma * config.FWHM / 2
        return _truncated_quantile_grid(distribution, -bound, bound, count)
    else
        values = Float64.(config.delta_values)
        length(values) == count || error("length(delta_values) must equal M_delta.")
        return copy(values), values
    end
end

function _equal_weight_coupling_samples(config, count)
    validate_coupling_inhomogeneity(config)
    if config.kind == :constant
        count == 1 || error("Constant coupling requires M_g = 1.")
        value = Float64(config.g_value)
        return [prevfloat(value), nextfloat(value)], [value]
    elseif config.kind == :gaussian
        sigma = config.FWHM / (2sqrt(2log(2)))
        low = max(0.0, config.mean - config.span_sigma * sigma)
        high = config.mean + config.span_sigma * sigma
        return _truncated_quantile_grid(Normal(config.mean, sigma), low, high, count)
    elseif hasproperty(config, :g_values)
        values = Float64.(config.g_values)
        length(values) == count || error("length(g_values) must equal M_g.")
        return copy(values), values
    else
        error("Equal-weight sampling of file-based custom coupling is not supported.")
    end
end

"""Build a deterministic quantile ensemble of equally weighted representative atoms."""
function build_equal_weight_ensemble(freq_config, g_config, M_delta, M_g)
    M_delta isa Integer && M_delta > 0 || error("M_delta must be positive.")
    M_g isa Integer && M_g > 0 || error("M_g must be positive.")
    edges_delta, delta_b_1d = _equal_weight_frequency_samples(freq_config, M_delta)
    edges_g, g_b_1d = _equal_weight_coupling_samples(g_config, M_g)
    p_delta = fill(1 / M_delta, M_delta)
    p_g = fill(1 / M_g, M_g)
    atom_count = M_delta * M_g
    Nj, delta_b, g_b, N_total, Nj_2d =
        build_2d_bins(atom_count, delta_b_1d, p_delta, g_b_1d, p_g)
    g_mean, g_std, g2_avg = weighted_g_stats_from_bins(g_b_1d, p_g)
    return (M_delta=M_delta, M_g=M_g, M=atom_count,
        sampling=:equal_weight_quantile, freq_inhomogeneity=freq_config,
        g_inhomogeneity=g_config, N=atom_count, N_total=N_total,
        Nj=Nj, delta_b=delta_b, g_b=g_b, edges_delta=edges_delta,
        delta_b_1d=delta_b_1d, p_delta=p_delta, edges_g=edges_g,
        g_b_1d=g_b_1d, p_g=p_g, Nj_2d=Nj_2d,
        g_mean=g_mean, g_std=g_std, g2_avg=g2_avg)
end

function prepare_derived(CONFIG)
    return build_ensemble(
        CONFIG.N,
        CONFIG.freq_inhomogeneity,
        CONFIG.g_inhomogeneity,
        CONFIG.M_delta,
        CONFIG.M_g,
    )
end
