# ============================================================
# Pulse editing utilities
# ============================================================
using Printf

function unwrap_phase(ϕ)
    out = float.(ϕ)
    correction = 0.0

    for k in 2:length(out)
        correction -= 2π * round((ϕ[k] - ϕ[k-1]) / (2π))
        out[k] += correction
    end

    return out
end


function taper_edges!(E; n_edge=8, amp_floor_frac=0.05)
    N = length(E)
    n_edge isa Integer && n_edge > 0 || error("n_edge must be a positive integer.")
    N >= 2n_edge + 1 || error("Pulse length must be at least 2n_edge + 1.")
    amp_floor_frac isa Real && isfinite(amp_floor_frac) && 0 <= amp_floor_frac <= 1 ||
        error("amp_floor_frac must lie between 0 and 1.")

    x = range(0, 1; length=n_edge)
    shape = 0.5 .* (1 .- cos.(π .* x))

    amp_left = abs(E[n_edge+1])
    amp_right = abs(E[N-n_edge])

    left_amp = amp_floor_frac * amp_left .+
               (amp_left - amp_floor_frac * amp_left) .* shape

    right_amp = amp_floor_frac * amp_right .+
                (amp_right - amp_floor_frac * amp_right) .* reverse(shape)

    E[1:n_edge] .= left_amp .* exp.(1im .* angle.(E[1:n_edge]))
    E[end-n_edge+1:end] .= right_amp .* exp.(1im .* angle.(E[end-n_edge+1:end]))

    return E
end


function smooth_phase_region!(E, k1, k2)
    N = length(E)
    k1 isa Integer && k2 isa Integer || error("k1 and k2 must be integers.")
    2 <= k1 <= k2 <= N - 1 ||
        error("Indices must satisfy 2 <= k1 <= k2 <= length(E)-1.")

    phase = unwrap_phase(angle.(E))
    ϕ1 = phase[k1-1]
    ϕ2 = phase[k2+1]
    new_phase = range(ϕ1, ϕ2; length=k2-k1+3)

    E[k1:k2] .= abs.(E[k1:k2]) .* exp.(1im .* new_phase[2:end-1])
    return E
end


"""
    smooth_amplitude!(E; strength=0.5, passes=1)

Smooth pulse amplitudes while preserving every sample phase. Endpoint
amplitudes remain fixed.
"""
function smooth_amplitude!(E; strength=0.5, passes=1)
    length(E) >= 3 || error("E must contain at least three samples.")
    strength isa Real && isfinite(strength) && 0 <= strength <= 1 ||
        error("strength must lie between 0 and 1.")
    passes isa Integer && passes >= 0 ||
        error("passes must be a non-negative integer.")

    phase = angle.(E)
    amplitude = abs.(E)
    smoothed = similar(amplitude)

    for _ in 1:passes
        copyto!(smoothed, amplitude)
        for k in 2:length(amplitude)-1
            local_average = 0.25 * (amplitude[k-1] + 2amplitude[k] + amplitude[k+1])
            smoothed[k] = (1 - strength) * amplitude[k] + strength * local_average
        end
        amplitude, smoothed = smoothed, amplitude
    end

    E .= amplitude .* exp.(1im .* phase)
    return E
end

function limit_phase_slew!(E, dt; f_max=2e6)
    isempty(E) && error("E cannot be empty.")
    dt isa Real && isfinite(dt) && dt > 0 || error("dt must be positive and finite.")
    f_max isa Real && isfinite(f_max) && f_max >= 0 ||
        error("f_max must be non-negative and finite.")

    amp = abs.(E)
    phase = unwrap_phase(angle.(E))
    dphi_max = 2π * f_max * dt

    for k in 2:length(E)
        dphi = phase[k] - phase[k-1]
        phase[k] = phase[k-1] + clamp(dphi, -dphi_max, dphi_max)
    end

    E .= amp .* exp.(1im .* phase)
    return E
end

function resample_pulse(E, dt; factor=2)
    N_old = length(E)
    dt isa Real && isfinite(dt) && dt > 0 || error("dt must be positive and finite.")
    N_old >= 2 || error("E must contain at least two samples.")
    factor isa Real && isfinite(factor) && factor > 0 ||
        error("factor must be finite and positive.")
    N_new = round(Int, factor * N_old)
    N_new >= 2 || error("factor produces fewer than two samples.")
    T = N_old * dt

    amp = abs.(E)
    phase = unwrap_phase(angle.(E))

    dt_new = T / N_new
    # Treat E[k] as the value on the kth control interval and interpolate
    # between interval centers while keeping the total duration fixed.
    t_new = ((0:N_new-1) .+ 0.5) .* dt_new

    amp_new = similar(t_new)
    phase_new = similar(t_new)

    for k in eachindex(t_new)
        x = t_new[k] / dt + 0.5
        i = clamp(floor(Int, x), 1, N_old-1)
        α = clamp(x - i, 0.0, 1.0)

        amp_new[k] = (1-α)*amp[i] + α*amp[i+1]
        phase_new[k] = (1-α)*phase[i] + α*phase[i+1]
    end

    E_new = amp_new .* exp.(1im .* phase_new)

    return ComplexF64.(E_new), dt_new
end

function inspect_pulse(E, dt; k1=1, k2=length(E))
    isempty(E) && error("E cannot be empty.")
    dt isa Real && isfinite(dt) && dt > 0 || error("dt must be positive and finite.")
    k1 isa Integer && k2 isa Integer || error("k1 and k2 must be integers.")
    1 <= k1 <= k2 <= length(E) || error("Indices must satisfy 1 <= k1 <= k2 <= length(E).")

    phase = unwrap_phase(angle.(E))

    println("k\t time(μs)\t amp\t\t phase(rad)\t freq(MHz)")
    println("-"^70)

    for k in k1:k2
        t = (k-1) * dt * 1e6
        amp = abs(E[k])

        if k == 1
            @printf("%d\t %.3f\t\t %.6f\t %.6f\t ---\n",
                    k, t, amp, phase[k])
        else
            freq = (phase[k] - phase[k-1]) / (2π * dt) / 1e6
            @printf("%d\t %.3f\t\t %.6f\t %.6f\t %.6f\n",
                    k, t, amp, phase[k], freq)
        end
    end
end

function frequency_smooth_penalty(E; λ_freq=1e-4, control_mask=nothing)
    λ_freq isa Real && isfinite(λ_freq) && λ_freq >= 0 ||
        error("λ_freq must be non-negative and finite.")
    N = length(E)
    N < 3 && return 0.0
    control_mask === nothing || length(control_mask) == N ||
        error("control_mask must have the same length as E.")
    control_mask === nothing || all(w -> w isa Real && isfinite(w) && 0 <= w <= 1,
        control_mask) || error("control_mask weights must lie in [0, 1].")

    φ = unwrap_phase(angle.(E))
    dφ = diff(φ)
    ddφ = diff(dφ)

    if control_mask === nothing
        return -λ_freq * sum(abs2, ddφ)
    end

    triplet_weight = control_mask[1:end-2] .* control_mask[2:end-1] .* control_mask[3:end]
    return -λ_freq * sum(abs2.(ddφ) .* triplet_weight)
end


function frequency_smooth_gradient(E; λ_freq=1e-4, eps_phase=1e-8,
    control_mask=nothing)
    λ_freq isa Real && isfinite(λ_freq) && λ_freq >= 0 ||
        error("λ_freq must be non-negative and finite.")
    eps_phase isa Real && isfinite(eps_phase) && eps_phase > 0 ||
        error("eps_phase must be positive and finite.")
    N = length(E)
    control_mask === nothing || length(control_mask) == N ||
        error("control_mask must have the same length as E.")
    control_mask === nothing || all(w -> w isa Real && isfinite(w) && 0 <= w <= 1,
        control_mask) || error("control_mask weights must lie in [0, 1].")

    gx = zeros(Float64, N)
    gy = zeros(Float64, N)

    φ = unwrap_phase(angle.(E))
    dφ = diff(φ)
    ddφ = diff(dφ)

    gφ = zeros(Float64, N)

    for k in 1:N-2
        q = ddφ[k]
        triplet_weight = control_mask === nothing ? 1.0 :
            control_mask[k] * control_mask[k+1] * control_mask[k+2]
        c = -2λ_freq * triplet_weight * q

        gφ[k]   += c
        gφ[k+1] += -2c
        gφ[k+2] += c
    end

    for k in 1:N
        x = real(E[k])
        y = imag(E[k])
        r2 = x^2 + y^2 + eps_phase^2

        gx[k] = gφ[k] * (-y / r2)
        gy[k] = gφ[k] * ( x / r2)
    end

    return gx .+ 1im .* gy
end

"""
    taper_tail!(E, first_sample)

Replace the tail after an unchanged anchor sample with a monotone cubic
amplitude taper to zero. Continue the incoming phase slope through the tail.
Returns E; applying it again with the same anchor leaves the tail unchanged.
"""
function taper_tail!(E, first_sample)
    first_sample isa Integer && 2 <= first_sample < length(E) ||
        error("first_sample must lie between 2 and length(E)-1.")
    all(isfinite, E) || error("Pulse samples must be finite.")
    k = first_sample
    width = length(E) - k
    amplitude = abs(E[k])
    # Limit the incoming amplitude slope to preserve a monotone, nonnegative
    # Hermite segment, with zero slope at its zero-amplitude endpoint.
    slope = clamp(width * (amplitude - abs(E[k-1])), -3amplitude, 0.0)
    phase = angle(E[k])
    phase_step = angle(E[k] * conj(E[k-1]))
    for j in k+1:length(E)
        u = (j - k) / width
        a = (1-u)^2 * (amplitude * (1+2u) + slope * u)
        E[j] = a * cis(phase + (j-k) * phase_step)
    end
    E[end] = 0
    return E
end
