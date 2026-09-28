using Test
using SpinDynamics

@testset "pulse constructors" begin
    gaussian = gaussian_drive(t0=1.0, sigma=0.2, amp=0.5)
    @test gaussian(1.0) ≈ 0.5 + 0im
    @test abs(gaussian(0.0)) > 0
    @test_throws ErrorException gaussian_drive(t0=0.0, sigma=0.0, amp=1.0)

    piecewise = piecewise_drive(ComplexF64[1, 2], 1.0, 0.5)
    @test piecewise(0.9) == 0im
    @test piecewise(1.0) == 1 + 0im
    @test piecewise(1.5) == 2 + 0im
    @test piecewise(2.0) == 0im
    @test_throws ErrorException piecewise_drive(ComplexF64[], 0.0, 1.0)
    @test_throws ErrorException piecewise_drive(ComplexF64[1], 0.0, 0.0)

    @test constant_drive(amp=2im)(3.0) == 2im
    f = t -> 1 + 2im * t
    @test custom_drive(f=f) === f
end

@testset "three WURST pulse" begin
    durations = (1.0, 2.0, 1.0)
    combined = three_wurst_drive(t_start=0.0, durations=durations,
        amp=(1.0, 2.0, 3.0), bandwidth=10.0, edge_frac=0.01)

    centers = (0.5, 2.0, 3.5)
    pulses = ntuple(3) do k
        wurst_drive(t_center=centers[k], duration=durations[k], amp=k,
            bandwidth=10.0, edge_frac=0.01)
    end

    for t in (0.25, 1.5, 3.75)
        @test combined(t) ≈ sum(pulse(t) for pulse in pulses)
    end

    cfg = ((kind=:three_wurst, t_start=0.0, durations=durations,
        amp=1.0, bandwidth=10.0),)
    @test isnothing(SpinDynamics.validate_pulse_config(cfg))
    total = SpinDynamics.build_E_of_t(cfg)
    @test total(2.0) isa Complex
    @test_throws ErrorException SpinDynamics.validate_pulse_config(
        ((kind=:three_wurst, t_start=0.0, durations=(1.0, 2.0),
          amp=1.0, bandwidth=10.0),))
    @test_throws ErrorException SpinDynamics.validate_pulse_config(
        ((kind=:single_reset_chirp,),))
end


@testset "pulse editing" begin
    wrapped = [0.0, 0.9π, -0.9π, -0.8π]
    unwrapped = unwrap_phase(wrapped)
    @test maximum(abs.(diff(unwrapped))) <= π

    E = ComplexF64[1, 1im, -1, -1im]
    refined, dt_new = resample_pulse(E, 1.0; factor=2)
    @test length(refined) == 8
    @test dt_new == 0.5

    reduced, dt_reduced = resample_pulse(E, 1.0; factor=0.5)
    @test length(reduced) == 2
    @test dt_reduced == 2.0
    @test_throws ErrorException resample_pulse(E, 1.0; factor=0.0)
    @test_throws ErrorException taper_edges!(copy(E); n_edge=2)
    @test_throws ErrorException smooth_phase_region!(copy(E), 1, 2)
    @test_throws ErrorException limit_phase_slew!(copy(E), 0.0)
    @test_throws ErrorException frequency_smooth_penalty(E; λ_freq=-1.0)
    masked = Bool[true, false, true, true]
    @test frequency_smooth_penalty(E; control_mask=masked) == 0
    @test frequency_smooth_gradient(E; control_mask=masked) == zeros(ComplexF64, 4)
    @test_throws ErrorException frequency_smooth_penalty(E; control_mask=trues(3))

    rough = ComplexF64[1, 3im, -1, -3im, 1]
    phase_before = angle.(rough)
    roughness_before = sum(abs2, diff(abs.(rough)))
    smooth_amplitude!(rough; strength=1.0, passes=2)
    @test angle.(rough) ≈ phase_before
    @test sum(abs2, diff(abs.(rough))) < roughness_before
    @test abs(rough[1]) ≈ 1.0
    @test abs(rough[end]) ≈ 1.0
    @test_throws ErrorException smooth_amplitude!(copy(rough); strength=1.5)

end

@testset "continuous tail cleanup" begin
    E = ComplexF64[(2 + sin(k)) * cis(0.2k) for k in 1:100]
    original = copy(E)
    SpinDynamics.taper_tail!(E, 80)
    @test E[1:80] == original[1:80]
    @test E[end] == 0
    @test all(diff(abs.(E[80:end])) .<= 0)
    @test all(isfinite, E)
    @test all(isapprox.(angle.(E[81:99] .* conj.(E[80:98])), 0.2; atol=1e-12))
    cleaned = copy(E)
    SpinDynamics.taper_tail!(E, 80)
    @test E == cleaned
    @test_throws ErrorException SpinDynamics.taper_tail!(E, 1)
    @test_throws ErrorException SpinDynamics.taper_tail!(E, 100)
end
