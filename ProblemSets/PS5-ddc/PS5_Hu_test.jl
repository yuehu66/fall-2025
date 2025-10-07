using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, GLM, Distributions, HTTP, CSV

cd(@__DIR__)

include("PS5_Hu_source.jl")

# Set random seed for reproducibility
Random.seed!(123)

@testset "PS5 source unit tests" begin
    @testset "Data Loading Tests" begin
        # Test load_static_data
        @testset "load_static_data" begin
            df_static = load_static_data()
            
            # Structure tests
            @test df_static isa DataFrame
            @test all(in.([:bus_id, :time, :Odometer, :Branded, :Y], Ref(names(df_static))))
            @test nrow(df_static) > 0
            
            # Data validity tests
            @test all(df_static.time .>= 1)
            @test all(df_static.time .<= 20)
            @test all(x -> x isa Bool, df_static.Y)
            @test issorted(df_static, [:bus_id, :time])
            @test all(x -> x >= 0, df_static.Odometer)  # Odometer should be non-negative
            
            # Consistency tests
            @test length(unique(df_static.bus_id)) == nrow(df_static) ÷ 20  # Each bus should have 20 time periods
            @test allunique(tuple.(df_static.bus_id, df_static.time))  # No duplicate bus-time combinations
            
            # Data completeness tests
            @test !any(ismissing, df_static.Odometer)
            @test !any(ismissing, df_static.Branded)
            @test !any(ismissing, df_static.Y)
        end

        # Test load_dynamic_data
        @testset "load_dynamic_data" begin
            d = load_dynamic_data()
            
            # Structure tests
            @test d isa NamedTuple
            required_fields = [:Y, :X, :Xstate, :Zstate, :B, :N, :T, :xbin, :zbin]
            @test all(haskey.((d,), required_fields))
            
            # Dimension tests
            @test size(d.Y, 1) == d.N  # N buses
            @test size(d.Y, 2) == d.T  # T time periods
            @test size(d.X, 1) == d.N
            @test size(d.Xstate, 1) == d.xbin
            @test size(d.Zstate, 1) == d.zbin
            
            # Data type and range tests
            @test all(x -> x isa Bool, d.Y)
            @test all(x -> x >= 0, d.X)
            @test all(x -> x >= 0, d.Xstate)
            @test all(x -> x >= 0, d.Zstate)
            @test d.N > 0 && d.T > 0
            @test d.xbin > 0 && d.zbin > 0
            
            # State space consistency
            @test size(d.B, 1) == d.xbin * d.zbin
            @test size(d.B, 2) == 2  # Two actions (replace or keep)
        end
    end

    @testset "Static Model Tests" begin
        @testset "estimate_static_model" begin
            # Create synthetic datasets for testing
            N = [10, 20]  # Test different sample sizes
            T = [4, 8]   # Test different time periods
            
            for n in N, t in T
                # Create synthetic data
                bus_id = repeat(1:n, inner=t)
                time = repeat(1:t, outer=n)
                Odometer = Float64.(time) .+ randn(n*t) .* 0.01
                Branded = repeat(rand(Bool, n), inner=t)
                η = -1.0 .+ 0.5 .* Odometer .+ 0.8 .* Float64.(Branded)
                p = 1 ./(1 .+ exp.(-η))
                Y = rand.(Bernoulli.(p))
                df_test = DataFrame(bus_id=bus_id, time=time, Odometer=Odometer, Branded=Branded, Y=Y)
                
                # Model estimation tests
                mdl = estimate_static_model(df_test)
                
                # Basic structure tests
                @test mdl isa GLM.GeneralizedLinearModel
                @test length(coef(mdl)) == 3  # Intercept + Odometer + Branded
                @test StatsBase.nobs(mdl) == n * t
                
                # Coefficient tests
                coefs = coef(mdl)
                @test coefs[2] > 0  # Odometer coefficient should be positive (by construction)
                @test !any(isnan, coefs)  # No NaN coefficients
                @test !any(isinf, coefs)  # No infinite coefficients
                
                # Model fit tests
                @test deviance(mdl) >= 0  # Deviance should be non-negative
                @test 0 <= r2(mdl) <= 1   # R² should be between 0 and 1
                
                # Prediction tests
                probs = predict(mdl)
                @test all(0 .<= probs .<= 1)  # Predicted probabilities should be between 0 and 1
            end
            
            # Edge cases
            # Test with empty dataset
            @test_throws Exception estimate_static_model(DataFrame(bus_id=Int[], time=Int[], 
                                                                Odometer=Float64[], Branded=Bool[], Y=Bool[]))
            
            # Test with missing values
            df_missing = DataFrame(bus_id=[1,1], time=[1,2], Odometer=[1.0, missing], 
                                 Branded=[true, true], Y=[false, true])
            @test_throws Exception estimate_static_model(df_missing)
        end
    end

    @testset "Dynamic Model Components" begin
        @testset "compute_future_value!" begin
            # Test different grid sizes and time periods
            test_configs = [
                (zbin=2, xbin=3, T=3, N=5),
                (zbin=4, xbin=5, T=5, N=10),
                (zbin=3, xbin=4, T=4, N=8)
            ]
            
            for d in test_configs
                # Test with different parameter values
                for θ in [[0.5, -0.3], [0.0, 0.0], [1.0, 1.0]]
                    FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
                    compute_future_value!(FV, θ, d)
                    
                    # Dimension tests
                    @test size(FV) == (d.zbin * d.xbin, 2, d.T + 1)
                    
                    # Numerical stability tests
                    @test !any(isnan.(FV))  # No NaN values
                    @test !any(isinf.(FV))  # No Inf values
                    
                    # Terminal value tests
                    @test FV[:, :, end] ≈ zeros(d.zbin * d.xbin, 2)  # Terminal values should be zero
                    
                    # Monotonicity tests (future values should be non-increasing)
                    for t in 1:(d.T-1)
                        @test all(FV[:, :, t] .>= FV[:, :, t+1])
                    end
                    
                    # Value bounds tests
                    max_possible_value = maximum(abs.(θ)) * d.T
                    @test all(-max_possible_value .<= FV .<= max_possible_value)
                end
            end
        end

        @testset "log_likelihood_dynamic" begin
            test_configs = [
                (zbin=2, xbin=3, T=3, N=5),
                (zbin=4, xbin=5, T=5, N=10)
            ]
            
            for d in test_configs
                # Test with different parameter values
                for θ in [[0.5, -0.3], [0.0, 0.0], [1.0, 1.0]]
                    ll = log_likelihood_dynamic(θ, d)
                    
                    # Basic tests
                    @test ll isa Number
                    @test !isnan(ll)
                    @test !isinf(ll)
                    
                    # Log-likelihood should be non-positive
                    @test ll <= 0
                    
                    # Test likelihood changes with parameters
                    ll_perturbed = log_likelihood_dynamic(θ .+ 0.1, d)
                    @test ll ≠ ll_perturbed  # Should be sensitive to parameter changes
                end
            end
            
            # Edge case tests
            d = (zbin=2, xbin=3, T=3, N=5)
            @test_throws Exception log_likelihood_dynamic([Inf, 0.0], d)  # Test with invalid parameters
        end

        @testset "estimate_dynamic_model" begin
            test_configs = [
                (zbin=2, xbin=3, T=3, N=5),
                (zbin=3, xbin=4, T=4, N=8)
            ]
            
            for d in test_configs
                # Test with different starting values
                for θ_start in [nothing, [0.0, 0.0], [0.5, -0.3]]
                    result = estimate_dynamic_model(d; θ_start=θ_start)
                    
                    if !isnothing(result)
                        # Basic structure tests
                        @test result isa Union{Optim.OptimizationResults, NamedTuple}
                        
                        if result isa Optim.OptimizationResults
                            # Optimization results tests
                            @test Optim.converged(result)  # Should converge
                            @test !any(isnan.(Optim.minimizer(result)))  # No NaN parameters
                            @test !any(isinf.(Optim.minimizer(result)))  # No infinite parameters
                            
                            # Likelihood improvement test
                            if !isnothing(θ_start)
                                initial_ll = -log_likelihood_dynamic(θ_start, d)
                                final_ll = -Optim.minimum(result)
                                @test final_ll <= initial_ll  # Should improve or maintain likelihood
                            end
                        end
                    end
                end
            end
        end
    end

    @testset "Main Function Test" begin
        @testset "main" begin
            # Test execution and output
            result = main()
            
            # Capture any console output if needed
            # TODO: Add specific output validation tests based on expected behavior
            
            # Test workspace state after main execution
            # 1. Check if required data files exist
            @test isfile("busdata.csv") || isfile("busdataBeta0.csv")
            
            # 2. Verify data loading still works after main
            try
                df = load_static_data()
                @test df isa DataFrame
                
                d = load_dynamic_data()
                @test d isa NamedTuple
            catch e
                @test false  # Should not throw errors
            end
            
            # 3. Test state preservation
            # Add specific tests based on what main() is expected to modify
            # For example, if main creates output files, test their existence
            # if main modifies global state, test those modifications
        end
    end
end

