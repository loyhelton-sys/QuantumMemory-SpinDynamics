# ============================================================
# Frequency inhomogeneity
#
# Supported choices:
#     :gaussian
#     :lorentzian
#     :constant
#     :custom
#
# This file handles:
#     1. validation
#     2. frequency distribution construction
#     3. detuning-bin edge construction
#     4. metadata construction
# ============================================================

# ============================================================
# Validation
# ============================================================
function _validate_positive_finite_field(config, field, kind)
    hasproperty(config, field) ||
        error("$(kind) frequency inhomogeneity requires $(field).")
    value = getproperty(config, field)
    value isa Real && isfinite(value) && value > 0 ||
        error("freq_inhomogeneity.$(field) must be a positive finite real number.")
    return nothing
end

function validate_frequency_inhomogeneity(freq_inhomogeneity)
    hasproperty(freq_inhomogeneity, :kind) ||
        error("freq_inhomogeneity must contain a kind field.")

    kind = freq_inhomogeneity.kind
    if !(kind in (:constant, :gaussian, :lorentzian, :custom))
        error("Unknown frequency kind: $(kind). Use :constant, :gaussian, :lorentzian, or :custom.")
    end

    if hasproperty(freq_inhomogeneity, :renormalize) &&
       !(freq_inhomogeneity.renormalize isa Bool)
        error("freq_inhomogeneity.renormalize must be true or false.")
    end

    if kind == :gaussian
        _validate_positive_finite_field(freq_inhomogeneity, :FWHM, :gaussian)
        _validate_positive_finite_field(freq_inhomogeneity, :span_sigma, :gaussian)

    elseif kind == :lorentzian
        _validate_positive_finite_field(freq_inhomogeneity, :FWHM, :lorentzian)
        _validate_positive_finite_field(freq_inhomogeneity, :span_gamma, :lorentzian)

    elseif kind == :constant
        hasproperty(freq_inhomogeneity, :delta_value) ||
            error("Constant frequency inhomogeneity requires delta_value.")
        value = freq_inhomogeneity.delta_value
        value isa Real && isfinite(value) ||
            error("delta_value must be a finite real number.")
            
    else
        hasproperty(freq_inhomogeneity, :delta_values) ||
            error("Custom frequency inhomogeneity requires delta_values.")
        values = freq_inhomogeneity.delta_values
        values isa AbstractVector || error("delta_values must be a vector.")
        isempty(values) && error("delta_values cannot be empty.")
        all(x -> x isa Real && isfinite(x), values) ||
            error("All delta_values must be finite real numbers.")
    end

    return nothing
end

# ============================================================
# Build detuning distribution
# ============================================================
function build_frequency_distribution(freq_inhomogeneity)
    validate_frequency_inhomogeneity(freq_inhomogeneity)

    kind = freq_inhomogeneity.kind
    FWHM = get(freq_inhomogeneity, :FWHM, 0.0)

    if kind == :constant
        return nothing

    elseif kind == :gaussian
        σ = FWHM / (2 * sqrt(2 * log(2)))
        return Normal(0.0, σ)

    elseif kind == :lorentzian
        γL = FWHM / 2
        return Cauchy(0.0, γL)

    elseif kind == :custom
        return freq_inhomogeneity.delta_values
    end
end

# ============================================================
# Build detuning bin edges
# ============================================================
function build_frequency_edges(freq_inhomogeneity, M_delta)
    validate_frequency_inhomogeneity(freq_inhomogeneity)

    kind = freq_inhomogeneity.kind
    FWHM = get(freq_inhomogeneity, :FWHM, 0.0)

    if kind == :constant
        δ = freq_inhomogeneity.delta_value
        return [δ, δ]

    elseif kind == :gaussian
        σ = FWHM / (2 * sqrt(2 * log(2)))
        span_sigma = freq_inhomogeneity.span_sigma

        return range(-span_sigma * σ, span_sigma * σ;
            length = M_delta + 1,)

    elseif kind == :lorentzian
        γL = FWHM / 2
        span_gamma = freq_inhomogeneity.span_gamma

        return range(-span_gamma * γL, span_gamma * γL;
            length = M_delta + 1,)

    elseif kind == :custom
        δ = freq_inhomogeneity.delta_values
        return δ
    end
end

# ============================================================
# Optional probability renormalization
#
# If renormalize = true, the probability inside the chosen
# finite simulation range is normalized to 1.
#
# If renormalize = false, the probability outside the finite
# range is excluded, so sum(p_delta) may be less than 1.
# ============================================================
function renormalize_frequency_probs_enabled(freq_inhomogeneity)
    if hasproperty(freq_inhomogeneity, :renormalize)
        return freq_inhomogeneity.renormalize
    else
        return true
    end
end

function maybe_renormalize_frequency_probs!(p_delta, freq_inhomogeneity)
    if renormalize_frequency_probs_enabled(freq_inhomogeneity)
        s = sum(p_delta)
        if s <= 0
            error("Cannot renormalize detuning probabilities because sum(p_delta) <= 0.")
        end

        p_delta ./= s
    end

    return p_delta
end

function build_frequency_bins(freq_inhomogeneity, M_delta)
    validate_frequency_inhomogeneity(freq_inhomogeneity)
    M_delta > 0 || error("M_delta must be positive.")

    kind = freq_inhomogeneity.kind

    if kind == :constant
        M_delta == 1 || error("Constant frequency inhomogeneity requires M_delta = 1.")
        delta = Float64(freq_inhomogeneity.delta_value)
        edges = [delta, delta]
        values = [delta]
        probabilities = [1.0]

    elseif kind == :custom
        values = Float64.(freq_inhomogeneity.delta_values)
        length(values) == M_delta ||
            error("length(delta_values) must equal M_delta.")
        edges = copy(values)
        probabilities = fill(1 / M_delta, M_delta)

    else
        distribution = build_frequency_distribution(freq_inhomogeneity)
        edges = collect(build_frequency_edges(freq_inhomogeneity, M_delta))
        values, probabilities = bin_means_and_probs(distribution, edges)
        maybe_renormalize_frequency_probs!(probabilities, freq_inhomogeneity)
    end

    info = build_frequency_info(freq_inhomogeneity, edges)

    return edges, values, probabilities, info
end

# ============================================================
# Frequency info for saving / debugging
# ============================================================
function build_frequency_info(freq_inhomogeneity, edges_delta)
    validate_frequency_inhomogeneity(freq_inhomogeneity)

    kind = freq_inhomogeneity.kind
    FWHM = get(freq_inhomogeneity, :FWHM, 0.0)

    if kind == :constant
        return (kind = :constant,
            delta_value = freq_inhomogeneity.delta_value,
            FWHM = FWHM,)

    elseif kind == :gaussian
        σ = FWHM / (2 * sqrt(2 * log(2)))

        return (kind = :gaussian, FWHM = FWHM,
            sigma = σ,
            span_sigma = freq_inhomogeneity.span_sigma,
            edges_min = first(edges_delta),
            edges_max = last(edges_delta),
            renormalize = renormalize_frequency_probs_enabled(freq_inhomogeneity),)

    elseif kind == :lorentzian
        γL = FWHM / 2

        return (kind = :lorentzian, FWHM = FWHM,
            gammaL = γL,
            span_gamma = freq_inhomogeneity.span_gamma,
            edges_min = first(edges_delta),
            edges_max = last(edges_delta),
            renormalize = renormalize_frequency_probs_enabled(freq_inhomogeneity),)
    
    elseif kind == :custom
        return (kind = :custom,
            delta_values = freq_inhomogeneity.delta_values,)
    
    end
end