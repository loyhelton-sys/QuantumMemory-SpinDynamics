using Test
using SpinDynamics

@testset "frequency inhomogeneity" begin
    gaussian = (
        kind=:gaussian,
        FWHM=2.0,
        span_sigma=3.0,
        renormalize=true,
    )
    edges, values, probabilities, info =
        SpinDynamics.build_frequency_bins(gaussian, 5)

    @test length(edges) == 6
    @test length(values) == 5
    @test length(probabilities) == 5
    @test sum(probabilities) ≈ 1.0
    @test values ≈ -reverse(values)
    @test info.sigma ≈ 2.0 / (2 * sqrt(2 * log(2)))

    default_normalized = (
        kind=:gaussian,
        FWHM=2.0,
        span_sigma=1.0,
    )
    _, _, probabilities, info =
        SpinDynamics.build_frequency_bins(default_normalized, 5)
    @test sum(probabilities) ≈ 1.0
    @test info.renormalize === true

    not_normalized = merge(default_normalized, (renormalize=false,))
    _, _, probabilities, info =
        SpinDynamics.build_frequency_bins(not_normalized, 5)
    @test sum(probabilities) < 1.0
    @test info.renormalize === false

    constant = (kind=:constant, delta_value=1.25)
    _, values, probabilities, info =
        SpinDynamics.build_frequency_bins(constant, 1)
    @test values == [1.25]
    @test probabilities == [1.0]
    @test info.FWHM == 0.0

    custom = (kind=:custom, delta_values=[-1.0, 0.0, 2.0])
    _, values, probabilities, _ =
        SpinDynamics.build_frequency_bins(custom, 3)
    @test values == [-1.0, 0.0, 2.0]
    @test probabilities == fill(1 / 3, 3)
end

@testset "frequency configuration validation" begin
    @test_throws ErrorException SpinDynamics.validate_frequency_inhomogeneity((FWHM=1.0,))
    @test_throws ErrorException SpinDynamics.validate_frequency_inhomogeneity((kind=:unknown,))
    @test_throws ErrorException SpinDynamics.validate_frequency_inhomogeneity(
        (kind=:gaussian, FWHM=NaN, span_sigma=3.0))
    @test_throws ErrorException SpinDynamics.validate_frequency_inhomogeneity(
        (kind=:gaussian, FWHM=1.0, span_sigma=Inf))
    @test_throws ErrorException SpinDynamics.validate_frequency_inhomogeneity(
        (kind=:lorentzian, FWHM=1.0, span_gamma=0.0))
    @test_throws ErrorException SpinDynamics.validate_frequency_inhomogeneity(
        (kind=:constant, delta_value=Inf))
    @test_throws ErrorException SpinDynamics.validate_frequency_inhomogeneity(
        (kind=:custom, delta_values=[0.0, NaN]))
    @test_throws ErrorException SpinDynamics.validate_frequency_inhomogeneity(
        (kind=:gaussian, FWHM=1.0, span_sigma=3.0, renormalize=1))

    @test isnothing(SpinDynamics.validate_frequency_inhomogeneity(
        (kind=:lorentzian, FWHM=2.0, span_gamma=10.0, renormalize=false)))
end
