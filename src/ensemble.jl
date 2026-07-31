# ENSEMBLE DISCRETIZATION

function bin_means_and_probs(dist, edges)    # 把一个连续分布离散化为若干个bin，计算每个bin的平均值和概率
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

function second_moment_from_bins(x_b_1d, p_x)   # 计算离散化后的二阶矩
    p_sum = sum(p_x)
    if p_sum <= 0
        error("Cannot compute second moment because sum(p_x) <= 0.")
    end

    return sum((x_b_1d .^ 2) .* p_x) / p_sum
end

function build_2d_bins(N, delta_b_1d, p_delta, g_b_1d, p_g)   # 计算二维bin的平均值和概率
    M_delta = length(delta_b_1d)
    M_g     = length(g_b_1d)
    Nj_2d    = zeros(Float64, M_delta, M_g)
    delta_2d = zeros(Float64, M_delta, M_g)
    g_2d     = zeros(Float64, M_delta, M_g)

    for i in 1:M_delta
        for k in 1:M_g
            pij = p_delta[i] * p_g[k]
            Nj_2d[i, k]    = N * pij
            delta_2d[i, k] = delta_b_1d[i]
            g_2d[i, k]     = g_b_1d[k]
        end
    end

    Nj_flat    = vec(Nj_2d)
    delta_flat = vec(delta_2d)
    g_flat     = vec(g_2d)
    N_total = sum(Nj_2d)

    return Nj_flat, delta_flat, g_flat, N_total, Nj_2d
end


function prepare_derived(CONFIG)    # 计算衍生的参数
    M_delta = CONFIG.M_delta
    M_g     = CONFIG.M_g
    M       = M_delta * M_g

    # -----------------------------------------------
    # Frequency inhomogeneity
    # -----------------------------------------------
    freq_cfg = CONFIG.freq_inhomogeneity
    if freq_cfg.kind == :constant

        delta_b_1d = [Float64(freq_cfg.delta_value)]
        p_delta    = [1.0]

        edges_delta = [freq_cfg.delta_value, freq_cfg.delta_value]

        freq_info = (
            kind = :constant,
            delta_value = freq_cfg.delta_value,
            FWHM = 0.0,
        )

        FWHM = 0.0

    else

        dist_delta = build_frequency_distribution(freq_cfg)
        edges_delta = build_frequency_edges(freq_cfg, M_delta)
        delta_b_1d, p_delta = bin_means_and_probs(dist_delta, edges_delta)

        maybe_renormalize_frequency_probs!(p_delta, freq_cfg)

        freq_info = build_frequency_info(freq_cfg, edges_delta)
        FWHM = freq_info.FWHM

    end

    # -----------------------------------------------
    # Coupling inhomogeneity
    # -----------------------------------------------
    g_inhomogeneity = CONFIG.g_inhomogeneity
    edges_g, g_b_1d, p_g, g_mean, g_std, g2_avg, g_info = build_coupling_bins(g_inhomogeneity, M_g,)

    # -----------------------------------------------
    # Total spin number
    # -----------------------------------------------
    N = CONFIG.N
    println("Total spin number N = $N")

    # -----------------------------------------------
    # Build 2D bins
    # -----------------------------------------------
    Nj, delta_b, g_b, N_total, Nj_2d = build_2d_bins(N, delta_b_1d, p_delta, g_b_1d, p_g,)

    # -----------------------------------------------
    # Time grid
    # -----------------------------------------------
    timespan = (0.0, CONFIG.Ttotal)
    t_save = collect(range(0, CONFIG.Ttotal; length = CONFIG.Nt_save,))
    Nt = length(t_save)

    return (M_delta = M_delta, M_g = M_g, M = M,
        freq_inhomogeneity = freq_cfg, freq_info = freq_info, FWHM = FWHM,
        g_inhomogeneity = g_inhomogeneity, g_info = g_info,
        g_mean = g_mean, g_std = g_std, g2_avg = g2_avg,
        N = N, N_total = N_total, Nj = Nj, delta_b = delta_b, g_b = g_b,
        delta_b_1d = delta_b_1d, p_delta = p_delta,
        edges_g = edges_g, g_b_1d = g_b_1d, p_g = p_g,
        Nj_2d = Nj_2d, timespan = timespan, t_save = t_save, Nt = Nt,)
end