# ============================================================
# Coupling inhomogeneity
#
# Supported choices:
#     :constant
#     :gaussian
#     :custom
#
# For :custom, the JLD2 file must contain:
#     edges_g
#     g_b_1d_tmp
#     p_g_tmp
# ============================================================

# ============================================================
# Utilities
# ============================================================
function coupling_renormalize_enabled(g_inhomogeneity)
    if hasproperty(g_inhomogeneity, :renormalize)
        return g_inhomogeneity.renormalize
    else
        return true
    end
end


function maybe_renormalize_coupling_probs!(p_g, g_inhomogeneity)
    if coupling_renormalize_enabled(g_inhomogeneity)
        s = sum(p_g)
        if s <= 0
            error("Cannot renormalize g probabilities because sum(p_g) <= 0.")
        end

        p_g ./= s
    end

    return p_g
end


function weighted_g_stats_from_bins(g_b_1d, p_g)
    p_sum = sum(p_g)
    if p_sum <= 0
        error("Cannot compute g statistics because sum(p_g) <= 0.")
    end

    w = p_g ./ p_sum
    g_mean = sum(w .* g_b_1d)
    g2_avg = sum(w .* g_b_1d.^2)
    g_std  = sqrt(sum(w .* (g_b_1d .- g_mean).^2))

    return g_mean, g_std, g2_avg
end


# ============================================================
# Custom JLD2 loader
# ============================================================
function load_custom_g_distribution(filename; renormalize=true)
    filename isa AbstractString && !isempty(filename) ||
        error("filename must be a non-empty string.")
    isfile(filename) || error("Cannot find custom g-distribution file: $(filename)")
    renormalize isa Bool || error("renormalize must be true or false.")
    data = load(filename)
    for key in ("edges_g", "g_b_1d_tmp", "p_g_tmp")
        haskey(data, key) || error("Custom g-distribution file is missing key: $(key)")
    end

    edges_g    = Float64.(data["edges_g"])
    g_b_1d_tmp = Float64.(data["g_b_1d_tmp"])
    p_g_tmp    = Float64.(data["p_g_tmp"])

    all(isfinite, edges_g) || error("All custom g-bin edges must be finite.")
    all(diff(edges_g) .> 0) || error("Custom g-bin edges must be strictly increasing.")
    all(x -> isfinite(x) && x > 0, g_b_1d_tmp) ||
        error("All custom coupling values must be finite and positive.")
    all(x -> isfinite(x) && x >= 0, p_g_tmp) ||
        error("All custom coupling probabilities must be finite and non-negative.")

    if length(g_b_1d_tmp) != length(p_g_tmp)
        error("Loaded g_b_1d_tmp and p_g_tmp have different lengths: " *
              "length(g_b_1d_tmp) = $(length(g_b_1d_tmp)), " *
              "length(p_g_tmp) = $(length(p_g_tmp)).")
    end

    if length(edges_g) != length(g_b_1d_tmp) + 1
        error("Loaded edges_g has inconsistent length. " *
            "Expected length(edges_g) = length(g_b_1d_tmp) + 1.")
    end

    p_sum = sum(p_g_tmp)

    if p_sum <= 0
        error("Cannot use custom g distribution because sum(p_g_tmp) <= 0.")
    end

    if renormalize
        p_g_tmp = p_g_tmp ./ p_sum
    end

    g_mean_tmp, g_std_tmp, g2_avg_tmp = weighted_g_stats_from_bins(
        g_b_1d_tmp, p_g_tmp,)

    return edges_g, g_b_1d_tmp, p_g_tmp, g_mean_tmp, g_std_tmp, g2_avg_tmp
end


# ============================================================
# Validation
# ============================================================
function validate_coupling_inhomogeneity(g_inhomogeneity)
    if !hasproperty(g_inhomogeneity, :kind)
        error("g_inhomogeneity must contain a kind field.")
    end

    kind = g_inhomogeneity.kind

    allowed_kinds = (
        :constant,
        :gaussian,
        :custom,)

    if !(kind in allowed_kinds)
        error(
            "Unknown g_inhomogeneity.kind = $(kind). " *
            "Use :constant, :gaussian, or :custom.")
    end

    # ========================================================
    # Constant coupling
    # ========================================================

    if kind == :constant
        if !hasproperty(g_inhomogeneity, :g_value)
            error(
                "For constant coupling, provide g_value. For example:\n" *
                "g_inhomogeneity = (\n" *
                "    kind = :constant,\n" *
                "    g_value = 2*pi*100,\n" *
                ")")
        end

        g_value = g_inhomogeneity.g_value

        if !(g_value isa Real)
            error("For constant coupling, g_value must be a real number.")
        end

        if !isfinite(g_value)
            error("For constant coupling, g_value must be finite.")
        end

        if g_value <= 0
            error("For constant coupling, g_value must be positive.")
        end

    # ========================================================
    # Gaussian coupling distribution
    # ========================================================
    elseif kind == :gaussian
        if !hasproperty(g_inhomogeneity, :mean)
            error("For Gaussian g inhomogeneity, provide mean.")
        end

        if !hasproperty(g_inhomogeneity, :FWHM)
            error("For Gaussian g inhomogeneity, provide FWHM.")
        end

        if !hasproperty(g_inhomogeneity, :span_sigma)
            error("For Gaussian g inhomogeneity, provide span_sigma.")
        end

        mean = g_inhomogeneity.mean
        FWHM = g_inhomogeneity.FWHM
        span_sigma = g_inhomogeneity.span_sigma

        if !(mean isa Real) || !isfinite(mean) || mean <= 0
            error("For Gaussian coupling, mean must be a finite positive real number.")
        end

        if !(FWHM isa Real) || !isfinite(FWHM) || FWHM <= 0
            error("For Gaussian coupling, FWHM must be a finite positive real number. Use :constant when there is no coupling inhomogeneity.")
        end

        if !(span_sigma isa Real) || !isfinite(span_sigma) || span_sigma <= 0
            error("For Gaussian coupling, span_sigma must be a finite positive real number.")
        end

    # ========================================================
    # Custom coupling distribution
    # ========================================================
    elseif kind == :custom
        if !hasproperty(g_inhomogeneity, :g_values) &&
           !hasproperty(g_inhomogeneity, :filename)

            error("For custom g inhomogeneity, provide g_values or filename.")
        end

        if hasproperty(g_inhomogeneity, :g_values)
            if isempty(g_inhomogeneity.g_values)
                error("g_values cannot be empty.")
            end

            if any(x -> !(x isa Real) || !isfinite(x) || x <= 0,
               g_inhomogeneity.g_values)
                error("All g_values must be finite positive numbers.")
            end
        end  

        if hasproperty(g_inhomogeneity, :filename)
            if !isfile(g_inhomogeneity.filename)
                error(
                "Cannot find custom g-distribution file: " *
                "$(g_inhomogeneity.filename)"
                )
            end
        end
    end

    # ========================================================
    # Optional common setting
    # ========================================================
    if hasproperty(g_inhomogeneity, :renormalize)
        if !(g_inhomogeneity.renormalize isa Bool)
            error("g_inhomogeneity.renormalize must be true or false.")
        end
    end

    return nothing
end


# ============================================================
# Gaussian g distribution
# ============================================================
function build_gaussian_coupling_bins(g_inhomogeneity, M_g)
    g_mean_input = g_inhomogeneity.mean
    g_FWHM_input = g_inhomogeneity.FWHM
    g_std_input  = g_FWHM_input / (2 * sqrt(2 * log(2)))

    span_sigma = g_inhomogeneity.span_sigma

    g_low  = max(0.0, g_mean_input - span_sigma * g_std_input)
    g_high = g_mean_input + span_sigma * g_std_input

    dist_g = Normal(g_mean_input, g_std_input)

    edges_g = collect(range(g_low, g_high; length = M_g + 1,))

    g_b_1d, p_g = bin_means_and_probs(dist_g, edges_g,)

    maybe_renormalize_coupling_probs!( p_g, g_inhomogeneity,)

    g_mean, g_std, g2_avg = weighted_g_stats_from_bins(g_b_1d, p_g,)

    g_info = (kind = :gaussian,
        mean_input = g_mean_input,
        FWHM_input = g_FWHM_input,
        span_sigma = span_sigma,
        renormalize = coupling_renormalize_enabled(g_inhomogeneity),
        p_sum = sum(p_g),
        g_low = g_low, g_high = g_high,
        g_mean = g_mean, g_std = g_std, g2_avg = g2_avg,)
    return edges_g, g_b_1d, p_g, g_mean, g_std, g2_avg, g_info
end


# ============================================================
# Constant g
# ============================================================
function build_constant_coupling_bins(g_inhomogeneity, M_g)
    if M_g != 1
        error("Constant coupling requires M_g = 1, but received M_g = $(M_g).")
    end

    g_value = Float64(g_inhomogeneity.g_value)

    # One bin containing the full ensemble probability.
    g_b_1d = [g_value]
    p_g    = [1.0]

    # Degenerate distribution centered at g_value.
    # prevfloat/nextfloat keep the two edges strictly ordered.
    edges_g = [prevfloat(g_value), nextfloat(g_value),]

    # Distribution statistics.
    g_mean = g_value
    g_std  = 0.0
    g2_avg = abs2(g_value)

    g_info = (kind = :constant, g_value = g_value, M_g = 1,
        renormalize = true,)

    return (edges_g, g_b_1d, p_g, g_mean, g_std, g2_avg, g_info,)
end

# ============================================================
# Custom g distribution
# ============================================================
function build_custom_coupling_bins(g_inhomogeneity, M_g)
    renormalize = coupling_renormalize_enabled(g_inhomogeneity)

    if hasproperty(g_inhomogeneity, :g_values)
        g_b_1d = Float64.(g_inhomogeneity.g_values)

        if length(g_b_1d) != M_g
            error("SIM_SETTING.M_g = $(M_g), but length(g_values) = $(length(g_b_1d)).")
        end

        p_g = fill(1 / M_g, M_g)
        edges_g = g_b_1d
        g_mean = sum(p_g .* g_b_1d)
        g_std = sqrt(sum(p_g .* (g_b_1d .- g_mean).^2))
        g2_avg = sum(p_g .* g_b_1d.^2)

        g_info = (kind = :custom, g_values = g_b_1d, renormalize = renormalize,
            p_sum = sum(p_g), g_low = minimum(g_b_1d), g_high = maximum(g_b_1d),
            g_mean = g_mean, g_std = g_std, g2_avg = g2_avg)

        return edges_g, g_b_1d, p_g, g_mean, g_std, g2_avg, g_info
    end

    filename = g_inhomogeneity.filename
    edges_g, g_b_1d, p_g, g_mean, g_std, g2_avg = load_custom_g_distribution(filename; renormalize=renormalize)

    if length(g_b_1d) != M_g
        error("SIM_SETTING.M_g = $(M_g), but the custom g distribution contains $(length(g_b_1d)) bins. Set SIM_SETTING.M_g = $(length(g_b_1d)).")
    end

    g_info = (kind = :custom, filename = filename, renormalize = renormalize,
        p_sum = sum(p_g), g_low = minimum(g_b_1d), g_high = maximum(g_b_1d),
        g_mean = g_mean, g_std = g_std, g2_avg = g2_avg)

    return edges_g, g_b_1d, p_g, g_mean, g_std, g2_avg, g_info
end

# ============================================================
# Main selector
# ============================================================
function build_coupling_bins(g_inhomogeneity, M_g)
    validate_coupling_inhomogeneity(g_inhomogeneity)

    kind = g_inhomogeneity.kind

    if kind == :gaussian
        return build_gaussian_coupling_bins(g_inhomogeneity, M_g)

    elseif kind == :constant
        return build_constant_coupling_bins(g_inhomogeneity, M_g)

    elseif kind == :custom
        return build_custom_coupling_bins(g_inhomogeneity, M_g)

    else
        error("Unknown g_inhomogeneity.kind = $(kind).")
    end
end