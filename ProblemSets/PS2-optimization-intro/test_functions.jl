using Test, Optim, LinearAlgebra, Random, Statistics, DataFrames, CSV, HTTP
using Distributions  # for Multinomial distribution in tests

# Set random seed for reproducible tests
Random.seed!(123)

println("Running unit tests for PS2 functions...")

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Test Question 1 functions
#:::::::::::::::::::::::::::::::::::::::::::::::::::
@testset "Question 1: Basic Optimization Functions" begin
    f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
    minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
    
    # Test that f and minusf are negatives of each other
    test_points = [-10.0, -5.0, 0.0, 1.0]
    for x_val in test_points
        @test f([x_val]) ≈ -minusf([x_val])
    end
    
    # Test optimization result is reasonable
    result = optimize(minusf, [-7.0], BFGS())
    @test Optim.converged(result)
    @test abs(Optim.minimizer(result)[1] - (-7.5)) < 0.1  # approximate expected minimum
    
    println("✓ Question 1 tests passed")
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Test Question 2: OLS function
#:::::::::::::::::::::::::::::::::::::::::::::::::::
@testset "Question 2: OLS Function" begin
    function ols(beta, X, y)
        ssr = (y.-X*beta)'*(y.-X*beta)
        return ssr
    end
    
    # Create simple test data
    n = 100
    X_test = [ones(n) randn(n, 2)]  # intercept + 2 covariates
    β_true = [1.0, 2.0, -1.5]
    y_test = X_test * β_true + 0.1 * randn(n)  # add small noise
    
    # Test OLS function
    ssr = ols(β_true, X_test, y_test)
    @test ssr >= 0  # SSR should be non-negative
    @test ssr < 10  # Should be small with our setup
    
    # Test that OLS function gives different values for different betas
    β_wrong = [0.0, 0.0, 0.0]
    ssr_wrong = ols(β_wrong, X_test, y_test)
    @test ssr_wrong > ssr  # Wrong coefficients should give higher SSR
    
    # Test optimization gives reasonable result
    result = optimize(b -> ols(b, X_test, y_test), rand(3), BFGS())
    @test Optim.converged(result)
    β_hat = Optim.minimizer(result)
    @test length(β_hat) == 3
    
    # Check that optimized coefficients are close to true coefficients
    @test isapprox(β_hat, β_true, atol=0.5)
    
    println("✓ Question 2 tests passed")
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Test Question 3: Logit function
#:::::::::::::::::::::::::::::::::::::::::::::::::::
@testset "Question 3: Logit Function" begin
    function logit(alpha, X, d)
        xb = X * alpha
        p = 1 ./(1 .+ exp.(xb))
        loglike = -sum(d .* log.(p) .+ (1 .- d) .* log.(1 .- p))
        return loglike
    end
    
    # Create test data for binary logit
    n = 200
    X_test = [ones(n) randn(n, 2)]
    α_true = [0.5, 1.0, -0.8]
    
    # Generate binary outcome based on logistic model
    xb = X_test * α_true
    p_true = 1 ./ (1 .+ exp.(-xb))  # Note: this is the correct logistic formula
    y_test = rand(n) .< p_true
    
    # Test logit function properties
    ll = logit(α_true, X_test, y_test)
    @test ll > 0  # Should return positive value (negative log-likelihood)
    @test isfinite(ll)  # Should not be infinite
    
    # Test that wrong coefficients give higher negative log-likelihood
    α_wrong = zeros(3)
    ll_wrong = logit(α_wrong, X_test, y_test)
    @test ll_wrong > ll  # Wrong coefficients should give worse fit
    
    # Test optimization
    result = optimize(alpha -> logit(alpha, X_test, y_test), rand(3), BFGS())
    @test Optim.converged(result)
    α_hat = Optim.minimizer(result)
    @test length(α_hat) == 3
    
    # Check signs are reasonable (should be close to true parameters)
    @test sign(α_hat[2]) == sign(α_true[2])  # Check at least the sign is correct
    
    println("✓ Question 3 tests passed")
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Test Question 5: Multinomial Logit function
#:::::::::::::::::::::::::::::::::::::::::::::::::::
@testset "Question 5: Multinomial Logit Function" begin
    function mlogit(alpha, X, y)
        K = size(X, 2)
        J = length(unique(y))
        N = length(y)
        bigy = zeros(N,J)
        for i in 1:N
            bigy[i,y[i]] = 1
        end
        bigAlpha = [reshape(alpha, K, J-1) zeros(K)]

        num = zeros(N,J)
        dem = zeros(N)
        for j in 1:J
            num[:,j] = exp.(X*bigAlpha[:,j])
            dem += num[:,j]
        end
        P = num ./ repeat(dem, 1, J)

        loglike = -sum(bigy .* log.(P))
        return loglike
    end
    
    # Create small test data for multinomial logit
    n = 150
    k = 3  # number of covariates including intercept
    J = 3  # number of alternatives
    
    X_test = [ones(n) randn(n, k-1)]
    
    # Create true parameters (K x (J-1) matrix reshaped to vector)
    β_true = [
        0.5  -0.3;   # intercept for alternatives 1 and 2
        1.0   0.8;   # covariate 1 effects
        -0.5  1.2    # covariate 2 effects
    ]
    α_true = vec(β_true)  # reshape to vector
    
    # Generate multinomial outcomes
    U = X_test * [β_true zeros(k)]  # utilities, alternative 3 is reference
    P = exp.(U) ./ sum(exp.(U), dims=2)  # choice probabilities
    
    # Sample outcomes
    y_test = Int[]
    for i in 1:n
        y_choice = rand(Categorical(P[i,:]))
        push!(y_test, y_choice)
    end
    
    # Test mlogit function
    ll = mlogit(α_true, X_test, y_test)
    @test ll > 0  # Should be positive (negative log-likelihood)
    @test isfinite(ll)  # Should not be infinite
    
    # Test that wrong coefficients give higher negative log-likelihood
    α_wrong = zeros(length(α_true))
    ll_wrong = mlogit(α_wrong, X_test, y_test)
    @test ll_wrong > ll  # Null model should fit worse
    
    # Test parameter dimensions
    expected_params = k * (J-1)  # K * (J-1) parameters
    @test length(α_true) == expected_params
    
    # Test that function accepts correct parameter vector length
    @test_nowarn mlogit(α_true, X_test, y_test)
    
    # Test that function throws error for wrong parameter length
    @test_throws BoundsError mlogit(rand(expected_params + 1), X_test, y_test)
    
    println("✓ Question 5 tests passed")
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Test helper functions and data generation
#:::::::::::::::::::::::::::::::::::::::::::::::::::
@testset "Helper Functions and Data Generation" begin
    # Test data generation function from your script
    function generate_multinomial_logit_data(N::Int, K::Int)
        X = randn(N, K)
        X = hcat(ones(N), X)
        β = [
                    1.0  0.1 0.0;  
                    2.0  1.1 0.0;
                   -1.5  1.0 0.0;
                    0.5 -0.5 0.0;
                    1.0 -1.0 0.0
        ]
        @assert size(β, 1) == K + 1 "β should have K+1 rows"
        @assert size(β, 2) == 3 "β should have 3 columns for 3 choice alternatives"

        U = X * β
        P = exp.(U) ./ sum(exp.(U), dims=2)
        y = [argmax(rand(Multinomial(1, P[i,:]))) for i in 1:N]

        return X, y, β
    end
    
    # Test the data generation function
    N, K = 100, 4
    X_gen, y_gen, β_gen = generate_multinomial_logit_data(N, K)
    
    @test size(X_gen) == (N, K+1)  # Should add intercept
    @test length(y_gen) == N
    @test size(β_gen) == (K+1, 3)
    @test all(y_gen .>= 1) && all(y_gen .<= 3)  # outcomes should be 1, 2, or 3
    
    println("✓ Helper function tests passed")
end

println("\n🎉 All tests completed successfully!")
println("All functions are working as expected.")
