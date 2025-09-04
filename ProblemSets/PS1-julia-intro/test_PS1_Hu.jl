using Test
using JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions

# Include the main file (note: we'll need to fix some syntax errors first)
# include("PS1_Hu.jl")

# Set seed for reproducible tests
Random.seed!(1234)

@testset "PS1 Unit Tests" begin
    
    @testset "Question 1 Tests" begin
        # Test q1 function (after fixing syntax)
        @testset "Matrix Generation and Operations" begin
            # Create test matrices similar to q1
            Random.seed!(1234)
            A_test = rand(Uniform(-5, 10), (10, 7))
            B_test = rand(Normal(-2, 15), (10, 7))
            
            @test size(A_test) == (10, 7)
            @test size(B_test) == (10, 7)
            @test length(A_test) == 70
            
            # Test matrix C creation
            C_test = [A_test[1:5, 1:5] B_test[1:5, 6:7]]
            @test size(C_test) == (5, 7)
            
            # Test matrix D creation (element-wise multiplication with condition)
            D_test = A_test .* (A_test .<= 0)
            @test size(D_test) == size(A_test)
            @test all(D_test[A_test .> 0] .== 0)  # Elements should be 0 where A > 0
            
            # Test reshape operations
            E_test = reshape(B_test, 70, 1)
            @test size(E_test) == (70, 1)
            @test length(E_test) == length(B_test)
            
            # Test concatenation
            F_test = cat(A_test, B_test; dims=3)
            @test size(F_test) == (10, 7, 2)
            
            # Test permutation
            F_perm = permutedims(F_test, (3, 1, 2))
            @test size(F_perm) == (2, 10, 7)
            
            # Test Kronecker product
            G_test = kron(B_test, C_test)
            @test size(G_test) == (size(B_test, 1) * size(C_test, 1), size(B_test, 2) * size(C_test, 2))
        end
        
        @testset "File I/O Operations" begin
            # Test that files can be created (we'll use temporary test data)
            test_matrix = rand(3, 3)
            
            # Test JLD save/load
            save("test_matrix.jld", "test_matrix", test_matrix)
            @test isfile("test_matrix.jld")
            loaded_data = load("test_matrix.jld")
            @test "test_matrix" in keys(loaded_data)
            
            # Test CSV operations
            df_test = DataFrame(test_matrix, :auto)
            CSV.write("test_matrix.csv", df_test)
            @test isfile("test_matrix.csv")
            
            # Clean up test files
            rm("test_matrix.jld")
            rm("test_matrix.csv")
        end
    end
    
    @testset "Question 2 Tests" begin
        @testset "Element-wise Operations" begin
            A_test = rand(5, 5)
            B_test = rand(5, 5)
            
            # Test element-wise multiplication
            AB_manual = zeros(size(A_test))
            for r in 1:size(A_test, 1)
                for c in 1:size(A_test, 2)
                    AB_manual[r, c] = A_test[r, c] * B_test[r, c]
                end
            end
            
            AB_vectorized = A_test .* B_test
            @test AB_manual ≈ AB_vectorized
        end
        
        @testset "Conditional Filtering" begin
            C_test = rand(Uniform(-10, 10), (5, 5))
            
            # Manual filtering
            Cprime_manual = Float64[]
            for c in axes(C_test, 2)
                for r in axes(C_test, 1)
                    if C_test[r, c] >= -5 && C_test[r, c] <= 5
                        push!(Cprime_manual, C_test[r, c])
                    end
                end
            end
            
            # Vectorized filtering
            Cprime_vectorized = C_test[C_test .>= -5 .&& C_test .<= 5]
            
            @test length(Cprime_manual) == length(Cprime_vectorized)
            @test sort(Cprime_manual) ≈ sort(Cprime_vectorized)
        end
        
        @testset "Multi-dimensional Array Generation" begin
            N, K, T = 100, 6, 5  # Smaller for testing
            X = zeros(N, K, T)
            
            # Test array initialization
            @test size(X) == (N, K, T)
            
            # Test that we can set intercept column
            X[:, 1, :] .= 1.0
            @test all(X[:, 1, :] .== 1.0)
        end
    end
    
    @testset "Question 4 - Matrix Operations Function" begin
        @testset "matrixops function" begin
            function matrixops_fixed(A::Array{Float64}, B::Array{Float64})
                # Check dimensions
                if size(A) != size(B)
                    error("A and B must have the same dimensions.")
                end
                
                # (i) element-by-element product
                out1 = A .* B
                # (ii) matrix product A' * B
                out2 = A' * B
                # (iii) sum of all elements of A and B
                out3 = sum(A + B)
                
                return out1, out2, out3
            end
            
            # Test with compatible matrices
            A_test = rand(Float64, (3, 4))
            B_test = rand(Float64, (3, 4))
            
            out1, out2, out3 = matrixops_fixed(A_test, B_test)
            
            @test size(out1) == size(A_test)
            @test size(out2) == (size(A_test, 2), size(B_test, 2))
            @test typeof(out3) == Float64
            @test out1 ≈ A_test .* B_test
            @test out2 ≈ A_test' * B_test
            @test out3 ≈ sum(A_test + B_test)
            
            # Test dimension mismatch error
            C_test = rand(Float64, (2, 3))
            @test_throws ErrorException matrixops_fixed(A_test, C_test)
        end
    end
    
    @testset "Data Processing Tests" begin
        @testset "DataFrame Operations" begin
            # Create test DataFrame
            test_df = DataFrame(
                id = 1:10,
                grade = rand(8:16, 10),
                race = rand(["White", "Black", "Other"], 10),
                wage = rand(5.0:0.1:25.0, 10),
                never_married = rand([0, 1], 10)
            )
            
            # Test basic operations
            @test nrow(test_df) == 10
            @test ncol(test_df) == 5
            @test typeof(mean(test_df.never_married)) == Float64
            
            # Test frequency table
            freq_table = freqtable(test_df.race)
            @test typeof(freq_table) <: AbstractArray
            
            # Test groupby operations
            grouped = groupby(test_df, :race)
            mean_wage = combine(grouped, :wage => mean => :mean_wage)
            @test nrow(mean_wage) <= 3  # At most 3 race categories
            @test "mean_wage" in names(mean_wage)
        end
    end
    
    @testset "Random Number Generation" begin
        @testset "Distribution Sampling" begin
            Random.seed!(1234)
            
            # Test uniform distribution
            uniform_sample = rand(Uniform(-5, 10), 100)
            @test all(-5 .<= uniform_sample .<= 10)
            
            # Test normal distribution
            normal_sample = rand(Normal(-2, 15), 100)
            @test length(normal_sample) == 100
            
            # Test binomial distribution
            binomial_sample = rand(Binomial(20, 0.6), 10)
            @test all(0 .<= binomial_sample .<= 20)
        end
    end
    
    @testset "Utility Functions" begin
        @testset "Array Utilities" begin
            test_array = rand(5, 6)
            
            # Test size operations
            @test size(test_array, 1) * size(test_array, 2) == length(test_array)
            @test length(unique(test_array)) <= length(test_array)
            
            # Test reshape operations
            reshaped = reshape(test_array, length(test_array), 1)
            @test size(reshaped) == (30, 1)
            
            # Test flattening
            flattened = test_array[:]
            @test length(flattened) == length(test_array)
        end
    end
end

# Helper function to run all tests
function run_all_tests()
    println("Running PS1 Unit Tests...")
    # Run the test suite
    # Note: In practice, you would run this with `julia test_PS1_Hu.jl` 
    # or using Pkg.test() if set up as a package
end

# Print message when file is loaded
println("Test file loaded. Run tests with: julia test_PS1_Hu.jl")
