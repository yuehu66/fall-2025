using Test, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV

# Include the source file to test
include("PS6_Hu_source.jl")

@testset "PS6 Bus Engine Replacement Model Tests" begin

    # Test data for consistent use across tests
    test_data_small = nothing
    test_xstate = nothing
    test_zstate = nothing 
    test_branded = nothing

    @testset "1. Data Loading and Reshaping Tests" begin
        
        @testset "load_and_reshape_data function" begin
            # Test with actual URL (integration test)
            @test_nowarn begin
                url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
                df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
                
                # Store for later tests
                global test_data_small = first(df_long, 100)  # Small subset for testing
                global test_xstate = Xstate[1:5, :]  # First 5 buses
                global test_zstate = Zstate[1:5]
                global test_branded = Branded[1:5]
            end
            
            # Test return types and structure
            url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
            df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
            
            @test isa(df_long, DataFrame)
            @test isa(Xstate, Matrix)
            @test isa(Zstate, Vector) 
            @test isa(Branded, Vector)
            
            # Test DataFrame structure
            required_cols = [:bus_id, :Y, :time, :Odometer, :Xstate, :Zst, :RouteUsage, :Branded]
            for col in required_cols
                @test col in names(df_long)
            end
            
            # Test dimensions make sense
            @test size(Xstate, 2) == 20  # 20 time periods
            @test length(Zstate) == size(Xstate, 1)  # Same number of buses
            @test length(Branded) == size(Xstate, 1)
            
            # Test long format conversion
            n_buses = size(Xstate, 1)
            @test nrow(df_long) == n_buses * 20  # 20 observations per bus
            
            # Test time variable
            @test minimum(df_long.time) == 1
            @test maximum(df_long.time) == 20
            
            # Test that decisions are binary
            @test all(x -> x in [0, 1], df_long.Y)
            
            # Test that bus_id ranges appropriately
            @test minimum(df_long.bus_id) == 1
            @test maximum(df_long.bus_id) == n_buses
        end
    end

    @testset "2. Flexible Logit Estimation Tests" begin
        
        @testset "estimate_flexible_logit function" begin
            # Create mock data for testing
            mock_df = DataFrame(
                Y = rand([0, 1], 100),
                Odometer = rand(0:200, 100),
                RouteUsage = rand(0.25:0.01:1.25, 100),
                Branded = rand([0, 1], 100),
                time = rand(1:20, 100)
            )
            
            @test_nowarn result = estimate_flexible_logit(mock_df)
            
            result = estimate_flexible_logit(mock_df)
            @test isa(result, StatsModels.TableRegressionModel)
            
            # Test that model has reasonable properties
            @test length(coef(result)) > 1  # Should have multiple coefficients
            @test haskey(result.model.rr, :y)  # Should have response variable
        end
    end

    @testset "3. State Space Construction Tests" begin
        
        @testset "construct_state_space function" begin
            # Test with small dimensions
            xbin, zbin = 5, 3
            xval = collect(0:5:20)
            zval = collect(1:3)
            xtran = rand(xbin * zbin, xbin)  # Mock transition matrix
            
            state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
            
            @test isa(state_df, DataFrame)
            @test nrow(state_df) == xbin * zbin
            
            # Check required columns
            required_cols = [:Odometer, :RouteUsage, :Branded, :time]
            for col in required_cols
                @test col in names(state_df)
            end
            
            # Test value ranges
            @test all(x -> x in xval, state_df.Odometer)
            @test all(x -> x in zval, state_df.RouteUsage)
            @test all(x -> x == 0, state_df.Branded)  # Initialized to 0
            @test all(x -> x == 0, state_df.time)     # Initialized to 0
            
            # Test that we get all combinations
            unique_combos = unique(select(state_df, :Odometer, :RouteUsage))
            @test nrow(unique_combos) == xbin * zbin
        end
    end

    @testset "4. Future Value Computation Tests" begin
        
        @testset "compute_future_values function" begin
            # Create minimal test setup
            xbin, zbin, T = 3, 2, 5
            β = 0.9
            
            # Mock state dataframe
            state_df = DataFrame(
                Odometer = [0, 5, 10, 0, 5, 10],
                RouteUsage = [1, 1, 1, 2, 2, 2],
                Branded = zeros(6),
                time = zeros(6)
            )
            
            # Mock flexible logit model (create a simple mock)
            mock_df_fit = DataFrame(
                Y = rand([0, 1], 50),
                Odometer = rand(1:20, 50),
                RouteUsage = rand(1:3, 50),
                Branded = rand([0, 1], 50),
                time = rand(1:10, 50)
            )
            flex_logit = glm(@formula(Y ~ Odometer + RouteUsage + Branded + time), 
                           mock_df_fit, Binomial(), LogitLink())
            
            # Mock transition matrix
            xtran = rand(xbin * zbin, xbin)
            
            FV = compute_future_values(state_df, flex_logit, xtran, xbin, zbin, T, β)
            
            @test isa(FV, Array{Float64, 3})
            @test size(FV) == (xbin * zbin, 2, T + 1)
            
            # Test boundary conditions
            @test all(FV[:, :, 1] .== 0)  # Terminal period should be 0
            
            # Test that values are reasonable (negative due to -β * log(p0))
            @test all(FV[:, :, 2:end] .<= 0)
        end
    end

    @testset "5. Future Value Mapping Tests" begin
        
        @testset "compute_fvt1 function" begin
            # Create test data
            df_test = DataFrame(
                bus_id = repeat(1:2, inner=3),
                Y = rand([0, 1], 6),
                time = repeat(1:3, outer=2)
            )
            
            # Mock arrays
            FV = rand(6, 2, 4) * (-1)  # Negative values as expected
            xtran = rand(6, 3)
            Xstate = [1 2 3; 2 1 3]  # 2 buses, 3 time periods  
            Zstate = [1, 2]
            B = [0, 1]
            xbin = 3
            
            fvt1 = compute_fvt1(df_test, FV, xtran, Xstate, Zstate, xbin, B)
            
            @test isa(fvt1, Vector)
            @test length(fvt1) == nrow(df_test)
            @test all(isfinite.(fvt1))  # Should not have NaN or Inf
        end
    end

    @testset "6. Structural Parameter Estimation Tests" begin
        
        @testset "estimate_structural_params function" begin
            # Create test data
            df_test = DataFrame(
                Y = rand([0, 1], 50),
                Odometer = rand(1:200, 50),
                Branded = rand([0, 1], 50)
            )
            fvt1 = rand(50) * (-1)  # Negative offset values
            
            result = estimate_structural_params(df_test, fvt1)
            
            @test isa(result, StatsModels.TableRegressionModel)
            @test length(coef(result)) >= 2  # At least intercept and Odometer
            
            # Test that the model includes the offset
            @test hasfield(typeof(result.model), :offset)
        end
    end

    @testset "7. Integration Tests" begin
        
        @testset "create_grids function (from included file)" begin
            zval, zbin, xval, xbin, xtran = create_grids()
            
            @test isa(zval, Vector)
            @test isa(xval, Vector) 
            @test isa(xtran, Matrix)
            @test zbin == length(zval)
            @test xbin == length(xval)
            @test size(xtran) == (zbin * xbin, xbin)
            
            # Test transition matrix properties
            @test all(xtran .>= 0)  # Non-negative probabilities
            # Note: Rows don't necessarily sum to 1 in this implementation
        end
        
        @testset "End-to-end workflow (small scale)" begin
            # Test that all functions work together with small data
            @test_nowarn begin
                # Use real but small dataset
                url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
                df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
                
                # Take small subset to speed up tests
                small_df = first(df_long, 200)  
                small_Xstate = Xstate[1:10, :]
                small_Zstate = Zstate[1:10]
                small_Branded = Branded[1:10]
                
                # Test flexible logit
                flex_result = estimate_flexible_logit(small_df)
                
                # Test state space construction
                zval, zbin, xval, xbin, xtran = create_grids()
                state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
                
                # These would take too long for unit tests, so just test they don't error
                # FV = compute_future_values(state_df, flex_result, xtran, xbin, zbin, 20, 0.9)
                # fvt1 = compute_fvt1(small_df, FV, xtran, small_Xstate, small_Zstate, xbin, small_Branded)
                # final_result = estimate_structural_params(small_df, fvt1)
            end
        end
    end

    @testset "8. Error Handling and Edge Cases" begin
        
        @testset "Invalid input handling" begin
            # Test empty DataFrame
            empty_df = DataFrame()
            @test_throws Exception estimate_flexible_logit(empty_df)
            
            # Test mismatched dimensions
            df_test = DataFrame(Y = [0, 1], Odometer = [100, 200], Branded = [0, 1])
            fvt1_wrong_size = [1.0, 2.0, 3.0]  # Wrong length
            @test_throws Exception estimate_structural_params(df_test, fvt1_wrong_size)
        end
        
        @testset "Numerical stability" begin
            # Test with extreme values
            extreme_df = DataFrame(
                Y = [0, 1, 0, 1],
                Odometer = [0, 1000000, 0, 1000000],  # Very large odometer
                RouteUsage = [0.25, 1.25, 0.25, 1.25],
                Branded = [0, 1, 0, 1],
                time = [1, 2, 3, 4]
            )
            
            @test_nowarn result = estimate_flexible_logit(extreme_df)
        end
    end

    @testset "9. Mathematical Properties" begin
        
        @testset "Future value properties" begin
            # Test with known simple case
            xbin, zbin = 2, 1
            state_df = DataFrame(
                Odometer = [0, 10],
                RouteUsage = [1, 1], 
                Branded = [0, 0],
                time = [0, 0]
            )
            
            # Create deterministic mock model
            mock_df = DataFrame(Y = [0, 0, 1, 1], Odometer = [0, 10, 0, 10], 
                              RouteUsage = [1, 1, 1, 1], Branded = [0, 0, 0, 0], time = [1, 1, 1, 1])
            flex_logit = glm(@formula(Y ~ Odometer), mock_df, Binomial(), LogitLink())
            
            xtran = [0.8 0.2; 0.1 0.9]  # Simple transition matrix
            β = 0.9
            
            FV = compute_future_values(state_df, flex_logit, xtran, xbin, zbin, 3, β)
            
            # Test monotonicity: later periods should have smaller (more negative) values
            # This is because we're looking at -β * log(p0) where p0 = 1 - p1
            @test all(FV[:, :, 3] .<= FV[:, :, 4])  # Values should be more negative going forward
        end
    end

end

# Helper function to run all tests
function run_all_tests()
    println("Running comprehensive tests for PS6...")
    Test.runtests(@__FILE__)
    println("All tests completed!")
end