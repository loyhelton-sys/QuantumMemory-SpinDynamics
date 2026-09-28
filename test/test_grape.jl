using Test
using LinearAlgebra
using CUDA
using SpinDynamics

@testset "GRAPE public API and validation" begin
    for name in (:build_ensemble, :grape_forward_manual, :grape_objective_manual,
        :grape_gradient_manual, :grape_step_manual, :grape_objective_coherence,
        :grape_gradient_coherence, :rase_objective, :rase_gradient,
        :rase_coherence_timeline, :optimize_rase!)
        @test name in names(SpinDynamics)
    end

    delta = [-1.0, 0.2, 1.1]
    coupling = [0.8, 1.0, 1.2]
    weights = [0.2, 0.5, 0.3]
    control = ComplexF64[0.2 + 0.1im, -0.1 + 0.15im, 0.12 - 0.08im]
    target = [0.4, 0.4, -0.3]
    dt = 0.1
    n_sub = 8

    Sx, Sy, Sz = grape_forward_manual(control, delta, coupling, dt; n_sub=n_sub)
    @test size(Sx) == (3, 4)
    @test size(Sy) == (3, 4)
    @test size(Sz) == (3, 4)
    @test all(isfinite, Sx) && all(isfinite, Sy) && all(isfinite, Sz)

    objective(E) = grape_objective_manual(E, delta, coupling, dt;
        target_Sz=target, spin_weight=weights, n_sub=n_sub)
    analytic = grape_gradient_manual(control, delta, coupling, dt;
        target_Sz=target, spin_weight=weights, n_sub=n_sub)
    epsilon = 1e-6
    finite_difference = similar(control)
    for k in eachindex(control)
        plus = copy(control); minus = copy(control)
        plus[k] += epsilon; minus[k] -= epsilon
        gx = (objective(plus) - objective(minus)) / (2epsilon)
        plus = copy(control); minus = copy(control)
        plus[k] += im * epsilon; minus[k] -= im * epsilon
        gy = (objective(plus) - objective(minus)) / (2epsilon)
        finite_difference[k] = gx + im * gy
    end
    @test norm(analytic - finite_difference) / norm(finite_difference) < 2e-3

    @test_throws UndefKeywordError grape_gradient_manual(control, delta, coupling, dt)
    @test_throws ErrorException grape_forward_manual(control, delta, coupling, 0.0)
    @test_throws ErrorException grape_forward_manual(control, delta, coupling[1:2], dt)
    @test_throws ErrorException grape_objective_manual(control, delta, coupling, dt;
        target_Sz=target, spin_weight=zeros(3))
    @test_throws ErrorException grape_objective_coherence(control, delta, coupling, dt;
        spin_weight=weights, A_target=0.0)
end

@testset "RASE GPU smoke test" begin
    if CUDA.functional()
        delta = CuArray([-1.0, -0.3, 0.3, 1.0])
        coupling = CuArray(fill(1.0, 4))
        weights = CuArray(fill(0.25, 4))
        control = ComplexF64[0.2 + 0.1im, -0.1 + 0.1im, 0.1 - 0.1im]
        result = rase_objective(control, delta, coupling, weights, 0.05;
            delta_window=0.5, t_dephase=0.1, t_echo_free=0.1,
            w_pop_inside=0.5, w_pop_outside=0.5, w_coh=1.0, w_smooth=1e-3, n_sub=2)
        @test all(isfinite, result[1:5])
        @test size(result[7]) == (4,)
        @test size(result[7]) == (4,)
    else
        @test_skip CUDA.functional()
    end
end

function finite_difference_complex(objective, control; epsilon=1e-6)
    gradient = similar(control)
    for k in eachindex(control)
        plus = copy(control); minus = copy(control)
        plus[k] += epsilon; minus[k] -= epsilon
        gx = (objective(plus) - objective(minus)) / (2epsilon)
        plus = copy(control); minus = copy(control)
        plus[k] += im * epsilon; minus[k] -= im * epsilon
        gy = (objective(plus) - objective(minus)) / (2epsilon)
        gradient[k] = gx + im * gy
    end
    return gradient
end

@testset "GRAPE CPU coherence gradient" begin
    delta = [-1.2, -0.1, 0.9]
    coupling = [0.8, 1.0, 1.1]
    weights = [0.2, 0.5, 0.3]
    control = ComplexF64[0.2 + 0.1im, -0.1 + 0.15im, 0.12 - 0.08im]
    kwargs = (spin_weight=weights, n_sub=16, Sx0=0.45, Sy0=0.1,
        Sz0=-0.15, phi_target=0.37, A_target=0.5)
    objective(E) = grape_objective_coherence(E, delta, coupling, 0.08; kwargs...)
    analytic = grape_gradient_coherence(control, delta, coupling, 0.08; kwargs...)
    numerical = finite_difference_complex(objective, control)
    @test norm(analytic - numerical) / norm(numerical) < 2e-3
end

@testset "GRAPE GPU gradients" begin
    if CUDA.functional()
        delta = CuArray([-1.4, -0.5, 0.2, 0.9, 1.6])
        coupling = CuArray([0.8, 1.0, 1.1, 0.9, 1.2])
        weights = CuArray(fill(0.2, 5))
        target = CuArray([0.35, 0.40, -0.20, -0.35, -0.45])
        control = ComplexF64[0.2 + 0.1im, -0.12 + 0.16im, 0.08 - 0.14im]
        objective(E) = grape_objective_manual(E, delta, coupling, 0.07;
            target_Sz=target, spin_weight=weights, n_sub=16)
        analytic = grape_gradient_manual(control, delta, coupling, 0.07;
            target_Sz=target, spin_weight=weights, n_sub=16)
        numerical = finite_difference_complex(objective, control)
        @test norm(analytic - numerical) / norm(numerical) < 1.5e-3

        rase_control = ComplexF64[0.2 + 0.1im, -0.12 + 0.16im, 0.08 - 0.14im, 0.15 + 0.04im]
        rase_kwargs = (delta_window=1.0, t_dephase=0.13, t_echo_free=0.11,
            w_pop_inside=0.8, w_pop_outside=0.6, w_coh=2.0, w_smooth=0.03, n_sub=8)
        rase_obj(E) = rase_objective(E, delta, coupling, weights, 0.07; rase_kwargs...)[1]
        rase_analytic = rase_gradient(rase_control, delta, coupling, weights, 0.07; rase_kwargs...)
        rase_numerical = finite_difference_complex(rase_obj, rase_control)
        @test norm(rase_analytic - rase_numerical) / norm(rase_numerical) < 5e-3
    else
        @test_skip CUDA.functional()
        @test_skip CUDA.functional()
    end
end

@testset "RASE parameter validation" begin
    if CUDA.functional()
        delta = CuArray([-1.0, 0.0, 1.0])
        coupling = CuArray(ones(3))
        weights = CuArray(fill(1 / 3, 3))
        control = ComplexF64[0.1 + 0.05im, -0.05 + 0.1im]
        base = (delta_window=0.5, t_dephase=0.1, t_echo_free=0.1)

        @test_throws ErrorException rase_objective(control, delta, coupling, weights, 0.05; base..., t_dephase=-0.1)
        @test_throws ErrorException rase_objective(control, delta, coupling, weights, 0.05; base..., t_echo_free=Inf)
        @test_throws ErrorException rase_objective(control, delta, coupling, weights, 0.05; base..., Sz_init=-0.6)
        @test_throws ErrorException rase_objective(control, delta, coupling, weights, 0.05; base..., Sz_target=0.6)
        @test_throws ErrorException rase_gradient(control, delta, coupling, weights, 0.05; base..., n_edge=3)
        @test_throws ErrorException rase_coherence_timeline(control, delta, coupling, weights, 0.05;
            delta_window=0.5, t_dephase=0.1, t_free_after=0.1, Nt_free=1)
        @test_throws ErrorException rase_coherence_timeline(control, delta, coupling, weights, 0.05;
            delta_window=0.5, t_dephase=0.1, t_free_after=-0.1)

        for kwargs in ((η=0.0,), (max_iter=-1,), (coh_tol=-1.0,),
            (checkpoint_every=0,), (checkpoint_file="",))
            @test_throws ErrorException optimize_rase!(control, delta, coupling, weights, 0.05;
                base..., kwargs...)
        end
    else
        for _ in 1:12
            @test_skip CUDA.functional()
        end
    end
end


@testset "RASE population components" begin
    Sz = [0.4, 0.5, -0.4, -0.5]
    delta = [0.0, 0.1, 1.0, -1.0]
    weights = fill(0.25, 4)
    inside, outside = rase_population_objectives(
        Sz, delta, weights; delta_window=0.5, Sz_target=0.5)
    @test inside ≈ -0.005
    @test outside ≈ -0.005

    inside_all, outside_all = rase_population_objectives(
        Sz, delta, weights; delta_window=2.0, Sz_target=0.5)
    @test outside_all == 0
    @test_throws ErrorException rase_population_objectives(
        Sz, delta[1:3], weights; delta_window=0.5)
end

@testset "RASE adaptive learning rate" begin
    if CUDA.functional()
        mktempdir() do dir
            delta = CuArray([-1.4, -0.5, 0.2, 0.9, 1.6])
            coupling = CuArray([0.8, 1.0, 1.1, 0.9, 1.2])
            weights = CuArray(fill(0.2, 5))
            seed = ComplexF64[0.2 + 0.1im, -0.12 + 0.16im, 0.08 - 0.14im, 0.15 + 0.04im]
            kw = (delta_window=1.0, t_dephase=0.13, t_echo_free=0.11,
                w_pop_inside=0.8, w_pop_outside=0.6, w_coh=2.0, w_smooth=0.03, n_sub=8)
            initial = rase_objective(seed, delta, coupling, weights, 0.07; kw...)[1]
            file = joinpath(dir, "checkpoint.jld2")
            # An oversized step must backtrack and improve within ONE iteration.
            result = optimize_rase!(copy(seed), delta, coupling, weights, 0.07;
                kw..., η=1e4, max_iter=1, iteration_offset=10, checkpoint_file=file)
            saved = SpinDynamics.load(file)
            @test result[2] > initial
            @test saved["iter"] == 11
            @test saved["η"] < 1e4
            @test saved["E"] == result[1]
            @test saved["J"] == result[2]
            # Successful small steps recover, stopping at the specified cap.
            result = optimize_rase!(copy(seed), delta, coupling, weights, 0.07;
                kw..., η=1e-5, η_max=1.3e-5, max_iter=2, checkpoint_file=file)
            saved = SpinDynamics.load(file)
            @test result[2] > initial
            @test saved["η"] ≈ 1.3e-5
            # The fixed tail must survive accepted updates with its nonzero values.
            result = optimize_rase!(copy(seed), delta, coupling, weights, 0.07;
                kw..., η=1e-5, n_tail=2, max_iter=2, checkpoint_file=file)
            @test result[1][3:4] == seed[3:4]
            @test result[1][1:2] != seed[1:2]
            @test result[2] > initial
            # A frozen control cannot improve: stop without corrupting it.
            result = optimize_rase!(copy(seed), delta, coupling, weights, 0.07;
                kw..., η=1e-8, n_edge=2, max_iter=3, checkpoint_file=file)
            @test result[1] == seed
            @test result[2] == initial
            @test SpinDynamics.load(file)["iter"] == 1
            for extra in ((η_max=0.0,), (η_growth=1.0,), (η_growth=Inf,))
                @test_throws ErrorException optimize_rase!(copy(seed), delta, coupling, weights, 0.07;
                    kw..., extra..., checkpoint_file=file)
            end
        end
    else
        @test_skip CUDA.functional()
    end
end
