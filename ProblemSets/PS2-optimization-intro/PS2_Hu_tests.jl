using Test, Optim, LinearAlgebra, Random, Statistics

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
    n = 50
    X_test = [ones(n) randn(n, 2)]  # intercept + 2 covariates
    β_true = [1.0, 2.0, -1.5]
    y_test = X_test * β_true + 0.1 * randn(n)  # add small noise
    
    # Test OLS function
    ssr = ols(β_true, X_test, y_test)
    @test ssr >= 0  # SSR should be non-negative
    
    # Test that OLS function gives different values for different betas
    β_wrong = [0.0, 0.0, 0.0]
    ssr_wrong = ols(β_wrong, X_test, y_test)
    @test ssr_wrong > ssr  # Wrong coefficients should give higher SSR
    
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
    n = 50
    X_test = [ones(n) randn(n, 2)]
    α_test = [0.5, 1.0, -0.8]
    
    # Generate binary outcome
    y_test = rand(n) .> 0.5  # simple binary outcome
    
    # Test logit function properties
    ll = logit(α_test, X_test, y_test)
    @test ll > 0  # Should return positive value (negative log-likelihood)
    @test isfinite(ll)  # Should not be infinite
    
    # Test that different coefficients give different likelihood
    α_different = [0.0, 0.0, 0.0]
    ll_different = logit(α_different, X_test, y_test)
    @test ll_different != ll  # Different coefficients should give different likelihood
    
    println("✓ Question 3 tests passed")
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Test Question 5: Multinomial Logit function structure
#:::::::::::::::::::::::::::::::::::::::::::::::::::
@testset "Question 5: Multinomial Logit Function Structure" begin
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
    
    # Create simple test data
    n = 30
    k = 3  # number of covariates including intercept
    J = 3  # number of alternatives
    
    X_test = [ones(n) randn(n, k-1)]
    y_test = rand(1:J, n)  # random outcomes from 1 to J
    
    # Test parameter dimensions
    expected_params = k * (J-1)  # K * (J-1) parameters
    α_test = randn(expected_params)
    
    # Test mlogit function runs without error
    @test_nowarn ll = mlogit(α_test, X_test, y_test)
    
    # Test that function returns finite positive value
    ll = mlogit(α_test, X_test, y_test)
    @test ll > 0  # Should be positive (negative log-likelihood)
    @test isfinite(ll)  # Should not be infinite
    
    println("✓ Question 5 tests passed")
end

println("\n🎉 All basic tests completed successfully!")
println("All functions have correct structure and basic functionality.")
