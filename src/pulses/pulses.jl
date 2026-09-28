# ============================================================
# PULSE CONSTRUCTORS
# ============================================================
function piecewise_drive(E_control, t_start, dt)
    isempty(E_control) && error("E_control cannot be empty.")
    t_start isa Real && isfinite(t_start) || error("t_start must be finite.")
    dt isa Real && isfinite(dt) && dt > 0 || error("dt must be positive and finite.")

    N = length(E_control)
    t_end = t_start + N * dt

    function E_of_t(t)
        if t < t_start || t >= t_end
            return 0.0 + 0.0im
        end

        k = floor(Int, (t - t_start) / dt) + 1
        k = clamp(k, 1, N)

        return E_control[k]
    end

    return E_of_t
end
"""
    gaussian_pulse_amplitude(; area=π, sigma, g)

Return the Gaussian electric-field amplitude required to obtain
the target pulse area.

The Bloch-rotation area is defined by

    ∫ 2g E(t) dt = area.

This convention matches the factor `2gE` in the Bloch equations.
"""
function gaussian_pulse_amplitude(; area=π, sigma, g)
    sigma isa Real && isfinite(sigma) && sigma > 0 || error("sigma must be positive and finite.")
    g isa Real && isfinite(g) && g != 0 || error("g must be nonzero and finite.")

    return area / (2 * g * sqrt(2π) * sigma)

end

function gaussian_drive(;t0, sigma, amp,
    omega=0.0, phase=0.0, cutoff=0.0,)

    t0_f    = Float64(t0)
    sigma_f = Float64(sigma)
    amp_c   = ComplexF64(amp)
    omega_f = Float64(omega)
    phase_f = Float64(phase)
    cutoff_f = Float64(cutoff)

    isfinite(t0_f) || error("Gaussian t0 must be finite.")
    isfinite(sigma_f) && sigma_f > 0 || error("Gaussian sigma must be positive and finite.")
    isfinite(amp_c) || error("Gaussian amp must be finite.")
    isfinite(omega_f) || error("Gaussian omega must be finite.")
    isfinite(phase_f) || error("Gaussian phase must be finite.")
    isfinite(cutoff_f) && cutoff_f >= 0 || error("Gaussian cutoff must be non-negative and finite.")

    return function(t)
        τ = t - t0_f
        envelope = abs(amp_c) * exp(-(τ^2) / (2*sigma_f^2))

        if envelope < cutoff_f
            return 0.0 + 0.0im
        end

        return amp_c * exp(-(τ^2) / (2*sigma_f^2)) *
            exp(1im * (omega_f * τ + phase_f))
    end
end

function wurst_drive(; t_center, duration, amp, bandwidth, n=20.0,
    omega0=0.0, chirp_sign=+1.0, phase0=0.0, edge_frac=1e-4,)
    
    t_center_f  = Float64(t_center)
    duration_f  = Float64(duration)
    amp_c       = ComplexF64(amp)
    bandwidth_f = Float64(bandwidth)
    n_f         = Float64(n)
    omega0_f    = Float64(omega0)
    chirp_s     = Float64(chirp_sign)
    phase0_f    = Float64(phase0)
    edge_frac_f = Float64(edge_frac)

    isfinite(t_center_f) || error("WURST t_center must be finite.")
    isfinite(duration_f) && duration_f > 0 || error("WURST duration must be positive and finite.")
    isfinite(amp_c) || error("WURST amp must be finite.")
    isfinite(bandwidth_f) && bandwidth_f > 0 || error("WURST bandwidth must be positive and finite.")
    isfinite(n_f) && n_f > 0 || error("WURST n must be positive and finite.")
    isfinite(omega0_f) || error("WURST omega0 must be finite.")
    isfinite(chirp_s) && abs(chirp_s) == 1 || error("WURST chirp_sign must be +1 or -1.")
    isfinite(phase0_f) || error("WURST phase0 must be finite.")
    isfinite(edge_frac_f) && edge_frac_f > 0 || error("WURST edge_frac must be positive and finite.")

    t_start = t_center_f - duration_f/2
    edge    = max(duration_f * edge_frac_f, eps(Float64))

    return function (t)
        τ = t - t_start

        gate = 0.5 * (tanh((t - t_start) / edge) -
            tanh((t - (t_start + duration_f)) / edge))

        envelope = amp_c * (1 - abs(sin(pi * (τ - duration_f/2) / duration_f))^n_f)

        phase = phase0_f + (omega0_f - chirp_s * bandwidth_f/2) * τ +
                0.5 * chirp_s * (bandwidth_f / duration_f) * τ^2

        return gate * envelope * exp(1im * phase)
    end
end
 
function _three_wurst_values(value)
    if value isa Number
        return (value, value, value)
    end
    length(value) == 3 || error("A three_wurst parameter must be scalar or contain three values.")
    return Tuple(value)
end

"""Construct three consecutive WURST pulses."""
function three_wurst_drive(; t_start, durations, amp, bandwidth,
    n=20.0, omega0=0.0, chirp_sign=+1.0, phase0=0.0, edge_frac=1e-4,)

    t_start isa Real && isfinite(t_start) || error("three_wurst t_start must be finite.")
    length(durations) == 3 || error("three_wurst durations must contain three values.")
    all(x -> x isa Real && isfinite(x) && x > 0, durations) ||
        error("All three_wurst durations must be finite and positive.")

    durations_f = Float64.(durations)
    amps = _three_wurst_values(amp)
    bandwidths = _three_wurst_values(bandwidth)
    ns = _three_wurst_values(n)
    omega0s = _three_wurst_values(omega0)
    chirp_signs = _three_wurst_values(chirp_sign)
    phase0s = _three_wurst_values(phase0)
    edge_fracs = _three_wurst_values(edge_frac)

    starts = Float64(t_start) .+ (0.0, durations_f[1], durations_f[1] + durations_f[2])
    centers = starts .+ durations_f .* 0.5
    pulses = ntuple(3) do k
        wurst_drive(t_center=centers[k], duration=durations_f[k], amp=amps[k],
            bandwidth=bandwidths[k], n=ns[k], omega0=omega0s[k],
            chirp_sign=chirp_signs[k], phase0=phase0s[k], edge_frac=edge_fracs[k],)
    end

    return t -> pulses[1](t) + pulses[2](t) + pulses[3](t)
end

function constant_drive(; amp)
    amp_c = ComplexF64(amp)
    isfinite(amp_c) || error("Constant amp must be finite.")
    return t -> amp_c
end

function custom_drive(; f)
    f isa Function || error("Custom pulse f must be a function.")
    return f
end

function build_drive_pulse(cfg)   # 根据脉冲配置构建驱动脉冲
    if cfg.kind == :gaussian
        return gaussian_drive(
            t0    = cfg.t0,
            sigma = cfg.sigma,
            amp   = cfg.amp,
            omega = get(cfg, :omega, 0.0),
            phase = get(cfg, :phase, 0.0),
            cutoff = get(cfg, :cutoff, 0.0),)

    elseif cfg.kind == :wurst
        return wurst_drive(
            t_center   = cfg.t_center,
            duration   = cfg.duration,
            amp        = cfg.amp,
            bandwidth  = cfg.bandwidth,
            n          = get(cfg, :n, 20.0),
            omega0     = get(cfg, :omega0, 0.0),
            chirp_sign = get(cfg, :chirp_sign, +1.0),
            phase0     = get(cfg, :phase0, 0.0),
            edge_frac  = get(cfg, :edge_frac, 1e-4),)

    elseif cfg.kind == :three_wurst
        return three_wurst_drive(
            t_start   = cfg.t_start,
            durations = cfg.durations,
            amp       = cfg.amp,
            bandwidth = cfg.bandwidth,
            n          = get(cfg, :n, 20.0),
            omega0     = get(cfg, :omega0, 0.0),
            chirp_sign = get(cfg, :chirp_sign, +1.0),
            phase0     = get(cfg, :phase0, 0.0),
            edge_frac  = get(cfg, :edge_frac, 1e-4),
        )

    elseif cfg.kind == :constant
        return constant_drive(
            amp = cfg.amp,)
            
    elseif cfg.kind == :custom
        return custom_drive(f = cfg.f)

    else
        error("Unknown pulse kind: $(cfg.kind)")
    end
end

function build_E_of_t(PULSE_CONFIG)    # 总驱动场
    drive_pulses = tuple((build_drive_pulse(cfg) for cfg in PULSE_CONFIG)...)

    return function E_of_t(t)
        E_t = 0.0 + 0.0im
        @inbounds for pulse in drive_pulses
            E_t += pulse(t)
        end

        return E_t
    end
end