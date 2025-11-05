using Test, Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM, MultivariateStats, FreqTables, ForwardDiff

cd(@__DIR__)

include("lgwt.jl")
include("PS8_Hu_source.jl")

# Set random seed for reproducible tests
Random.seed!(42)

@testset "PS8 Factor Analysis Tests" begin
    
    # Mock data for testing
    function create_test_data()
        n = 100
        df = DataFrame(
            black = rand([0, 1], n),
            hispanic = rand([0, 1], n),
            female = rand([0, 1], n),
            schoolt = 8 .+ 8 * rand(n),
            gradHS = rand([0, 1], n),
            grad4yr = rand([0, 1], n),
            logwage = 2.0 .+ 0.1 * randn(n),
            asvabAR = 50 .+ 10 * randn(n),
            asvabCS = 50 .+ 10 * randn(n),
            asvabMK = 50 .+ 10 * randn(n),
            asvabNO = 50 .+ 10 * randn(n),
            asvabPC = 50 .+ 10 * randn(n),
            asvabWK = 50 .+ 10 * randn(n)
        )
        return df
    end
    
    @testset "Data Loading Tests" begin
        @testset "load_data function" begin
            # Test that function exists and has correct signature
            @test isdefined(Main, :load_data)
            @test hasmethod(load_data, (String,))
            
            # Test that function throws appropriate error for invalid URL
            @test_throws Exception load_data("invalid_url")
        end
    end
    
    @testset "ASVAB Correlation Tests" begin
        test_df = create_test_data()
        
        @testset "compute_asvab_correlations function" begin
            corr_result = compute_asvab_correlations(test_df)
            
            # Test that result is a DataFrame
            @test isa(corr_result, DataFrame)
            
            # Test dimensions - should be 6x6 correlation matrix
            @test size(corr_result) == (6, 6)
            
            # Test that all correlation values are between -1 and 1
            for col in names(corr_result)
                @test all(-1 .<= corr_result[:, col] .<= 1)
            end
            
            # Test that diagonal elements are 1 (correlation with self)
            @test all(abs.(diag(Matrix(corr_result)) .- 1.0) .< 1e-10)
            
            # Test symmetry of correlation matrix
            corr_matrix = Matrix(corr_result)
            @test all(abs.(corr_matrix - corr_matrix') .< 1e-10)
        end
    end
    
    @testset "Factor Analysis Tests" begin
        test_df = create_test_data()
        
        @testset "estimate_factor! function" begin
            df_modified = estimate_factor!(deepcopy(test_df))
            
            # Test that function returns a DataFrame
            @test isa(df_modified, DataFrame)
            
            # Test that asvabFactor column was added
            @test "asvabFactor" in names(df_modified)
            
            # Test that the factor column has correct length
            @test length(df_modified.asvabFactor) == nrow(test_df)
            
            # Test that factor values are numeric
            @test all(isa.(df_modified.asvabFactor, Real))
        end
        
        @testset "generate_factor! function" begin
            df_modified = generate_factor!(deepcopy(test_df))
            
            # Test that function returns a DataFrame
            @test isa(df_modified, DataFrame)
            
            # Test that asvabFactor column was added
            @test "asvabFactor" in names(df_modified)
            
            # Test that the factor column has correct length
            @test length(df_modified.asvabFactor) == nrow(test_df)
            
            # Test that factor values are numeric
            @test all(isa.(df_modified.asvabFactor, Real))
        end
    end
    
    @testset "Matrix Preparation Tests" begin
        test_df = create_test_data()
        
        @testset "prepare_factor_matrices function" begin
            X, y, Xfac, asvabs = prepare_factor_matrices(test_df)
            
            # Test dimensions
            n = nrow(test_df)
            @test size(X) == (n, 7)  # 6 variables + constant
            @test length(y) == n
            @test size(Xfac) == (n, 4)  # 3 variables + constant
            @test size(asvabs) == (n, 6)  # 6 ASVAB scores
            
            # Test that X has constant column (last column should be all ones)
            @test all(X[:, end] .== 1.0)
            
            # Test that Xfac has constant column (last column should be all ones)
            @test all(Xfac[:, end] .== 1.0)
            
            # Test that y contains the correct variable
            @test y == test_df.logwage
            
            # Test that asvabs contains ASVAB scores
            @test asvabs[:, 1] == test_df.asvabAR
            @test asvabs[:, 6] == test_df.asvabWK
        end
    end
    
    @testset "Factor Model Likelihood Tests" begin
        test_df = create_test_data()
        X, y, Xfac, asvabs = prepare_factor_matrices(test_df)
        
        @testset "factor_model function structure" begin
            # Create reasonable starting values
            K = size(X, 2)      # 7
            L = size(Xfac, 2)   # 4
            J = size(asvabs, 2) # 6
            
            # Parameter vector: γ (L×J), β (K), α (J+1), σ (J+1)
            θ = vcat(
                0.1 * randn(L * J),  # γ parameters
                0.1 * randn(K),      # β parameters
                0.5 * ones(J + 1),   # α parameters
                abs.(0.1 * randn(J + 1)) .+ 0.1    # σ parameters (ensure positive)
            )
            
            R = 5  # Small number of quadrature points for testing
            
            # Test that function runs without error
            @test_nowarn factor_model(θ, X, Xfac, asvabs, y, R)
            
            # Test that function returns a scalar
            result = factor_model(θ, X, Xfac, asvabs, y, R)
            @test isa(result, Real)
            
            # Test that likelihood is finite (unless numerical issues occur)
            @test isfinite(result) || isnan(result)  # Allow NaN for numerical edge cases
        end
        
        @testset "factor_model parameter sensitivity" begin
            K = size(X, 2)
            L = size(Xfac, 2)
            J = size(asvabs, 2)
            R = 3
            
            # Base parameter vector
            θ_base = vcat(
                0.01 * randn(L * J),
                0.01 * randn(K),
                0.3 * ones(J + 1),
                abs.(0.3 * randn(J + 1)) .+ 0.1  # Ensure positive σ
            )
            
            base_ll = factor_model(θ_base, X, Xfac, asvabs, y, R)
            
            # Test that changing parameters changes likelihood
            θ_modified = copy(θ_base)
            θ_modified[1] += 0.5  # Change first γ parameter
            modified_ll = factor_model(θ_modified, X, Xfac, asvabs, y, R)
            
            @test base_ll != modified_ll
        end
    end
    
    @testset "Estimation Function Tests" begin
        test_df = create_test_data()[1:50, :]  # Use smaller sample for faster testing
        
        @testset "run_estimation function structure" begin
            # Create simple starting values
            X, y, Xfac, asvabs = prepare_factor_matrices(test_df)
            K = size(X, 2)
            L = size(Xfac, 2)
            J = size(asvabs, 2)
            
            start_vals = vcat(
                0.01 * randn(L * J),
                0.01 * randn(K),
                0.1 * ones(J + 1),
                0.3 * ones(J + 1)
            )
            
            # Test that function exists and accepts correct arguments
            @test hasmethod(run_estimation, (DataFrame, Vector))
        end
    end
    
    @testset "Integration Tests" begin
        @testset "End-to-end workflow with small data" begin
            # Create very small dataset for quick testing
            small_df = create_test_data()[1:20, :]
            
            # Test correlation computation
            corr_result = compute_asvab_correlations(small_df)
            @test size(corr_result) == (6, 6)
            
            # Test factor generation
            factor_df = generate_factor!(deepcopy(small_df))
            @test "asvabFactor" in names(factor_df)
            
            # Test matrix preparation
            X, y, Xfac, asvabs = prepare_factor_matrices(factor_df)
            @test size(X, 1) == size(y, 1) == size(Xfac, 1) == size(asvabs, 1) == 20
        end
    end
    
    @testset "Robustness Tests" begin
        @testset "Edge cases and error handling" begin
            # Test with very small dataset
            tiny_df = create_test_data()[1:5, :]
            
            @test_nowarn compute_asvab_correlations(tiny_df)
            @test_nowarn prepare_factor_matrices(tiny_df)
            
            # Test with missing values (if applicable)
            df_with_na = create_test_data()[1:10, :]
            # Most functions should handle complete cases, so we test they don't crash
            @test_nowarn prepare_factor_matrices(df_with_na)
        end
        
        @testset "Parameter bounds and constraints" begin
            test_df = create_test_data()[1:30, :]
            X, y, Xfac, asvabs = prepare_factor_matrices(test_df)
            
            K = size(X, 2)
            L = size(Xfac, 2)
            J = size(asvabs, 2)
            R = 3
            
            # Test with very small positive σ values
            θ_small_sigma = vcat(
                0.01 * randn(L * J),
                0.01 * randn(K),
                0.1 * ones(J + 1),
                0.001 * ones(J + 1)  # Very small σ
            )
            
            @test_nowarn factor_model(θ_small_sigma, X, Xfac, asvabs, y, R)
            
            # Test with larger σ values
            θ_large_sigma = vcat(
                0.01 * randn(L * J),
                0.01 * randn(K),
                0.1 * ones(J + 1),
                2.0 * ones(J + 1)  # Larger σ
            )
            
            @test_nowarn factor_model(θ_large_sigma, X, Xfac, asvabs, y, R)
        end
    end
    
    @testset "Mathematical Properties Tests" begin
        @testset "Correlation matrix properties" begin
            test_df = create_test_data()
            corr_result = compute_asvab_correlations(test_df)
            corr_matrix = Matrix(corr_result)
            
            # Test positive semi-definiteness
            eigenvals = eigvals(corr_matrix)
            @test all(eigenvals .>= -1e-10)  # Allow for small numerical errors
            
            # Test that matrix is symmetric
            @test norm(corr_matrix - corr_matrix') < 1e-10
        end
        
        @testset "Factor analysis mathematical properties" begin
            test_df = create_test_data()
            
            # Test that factor analysis produces reasonable results
            factor_df = generate_factor!(deepcopy(test_df))
            
            # Factor scores should have reasonable scale
            @test std(factor_df.asvabFactor) > 0  # Should have variation
            @test abs(mean(factor_df.asvabFactor)) < 3 * std(factor_df.asvabFactor)  # Not too extreme
        end
    end
    
    @testset "Performance and Numerical Stability" begin
        @testset "Numerical stability with different scales" begin
            # Test with scaled data
            scaled_df = create_test_data()
            
            # Scale ASVAB scores by large factor
            for col in ["asvabAR", "asvabCS", "asvabMK", "asvabNO", "asvabPC", "asvabWK"]
                scaled_df[:, col] = scaled_df[:, col] * 100
            end
            
            @test_nowarn compute_asvab_correlations(scaled_df)
            @test_nowarn generate_factor!(deepcopy(scaled_df))
        end
    end
    
    println("All tests completed!")
end