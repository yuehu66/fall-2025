# Manual unit tests for PS2 functions
println("Running manual unit tests for PS2 functions...")

# Set random seed
using Random
Random.seed!(123)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Test 1: Basic optimization functions
#:::::::::::::::::::::::::::::::::::::::::::::::::::
println("\n=== Test 1: Basic Optimization Functions ===")

f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2

# Test that f and minusf are negatives
test_vals = [-5.0, 0.0, 1.0]
for val in test_vals
    f_val = f([val])
    minusf_val = minusf([val])
    if abs(f_val + minusf_val) < 1e-10
        println("✓ f and minusf are negatives at x = $val")
    else
        println("✗ ERROR: f and minusf are not negatives at x = $val")
    end
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Test 2: OLS function
#:::::::::::::::::::::::::::::::::::::::::::::::::::
println("\n=== Test 2: OLS Function ===")

function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

# Create test data
n = 50
X_test = [ones(n) randn(n, 2)]
β_true = [1.0, 2.0, -1.5]
y_test = X_test * β_true + 0.01 * randn(n)

# Test OLS function
ssr_true = ols(β_true, X_test, y_test)
ssr_zero = ols([0.0, 0.0, 0.0], X_test, y_test)

if ssr_true >= 0
    println("✓ OLS returns non-negative SSR")
else
    println("✗ ERROR: OLS returns negative SSR")
end

if ssr_zero > ssr_true
    println("✓ True coefficients give lower SSR than zero coefficients")
else
    println("✗ ERROR: True coefficients don't give lower SSR")
end

println("SSR with true coefficients: $(round(ssr_true, digits=6))")
println("SSR with zero coefficients: $(round(ssr_zero, digits=6))")

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Test 3: Logit function
#:::::::::::::::::::::::::::::::::::::::::::::::::::
println("\n=== Test 3: Logit Function ===")

function logit(alpha, X, d)
    xb = X * alpha
    p = 1 ./(1 .+ exp.(xb))
    loglike = -sum(d .* log.(p) .+ (1 .- d) .* log.(1 .- p))
    return loglike
end

# Create test data
n = 100
X_logit = [ones(n) randn(n, 1)]
α_test = [0.5, 1.0]
y_binary = rand(n) .> 0.4  # binary outcome

# Test logit function
ll = logit(α_test, X_logit, y_binary)
ll_zero = logit([0.0, 0.0], X_logit, y_binary)

if isfinite(ll) && ll > 0
    println("✓ Logit returns finite positive value")
else
    println("✗ ERROR: Logit doesn't return finite positive value")
end

println("Log-likelihood with test coefficients: $(round(ll, digits=4))")
println("Log-likelihood with zero coefficients: $(round(ll_zero, digits=4))")

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Test 4: Multinomial Logit function structure
#:::::::::::::::::::::::::::::::::::::::::::::::::::
println("\n=== Test 4: Multinomial Logit Function ===")

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

# Create test data
n = 50
k = 3
J = 3
X_mlogit = [ones(n) randn(n, k-1)]
y_mlogit = rand(1:J, n)

# Calculate expected parameter count
expected_params = k * (J-1)
α_mlogit = randn(expected_params)

try
    ll_mlogit = mlogit(α_mlogit, X_mlogit, y_mlogit)
    if isfinite(ll_mlogit) && ll_mlogit > 0
        println("✓ Multinomial logit returns finite positive value")
        println("Log-likelihood: $(round(ll_mlogit, digits=4))")
    else
        println("✗ ERROR: Multinomial logit doesn't return finite positive value")
    end
    
    println("✓ Function runs without error")
    println("Parameter vector length: $(length(α_mlogit))")
    println("Expected parameter count (K*(J-1)): $expected_params")
    
catch e
    println("✗ ERROR: Multinomial logit function failed with error: $e")
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Test 5: Parameter dimensions and consistency
#:::::::::::::::::::::::::::::::::::::::::::::::::::
println("\n=== Test 5: Parameter Dimensions ===")

# Test parameter reshaping in mlogit
K, J = 4, 3
total_params = K * (J-1)
test_alpha = 1:total_params
reshaped = reshape(test_alpha, K, J-1)

if size(reshaped) == (K, J-1)
    println("✓ Parameter reshaping works correctly")
    println("Original vector length: $(length(test_alpha))")
    println("Reshaped matrix size: $(size(reshaped))")
else
    println("✗ ERROR: Parameter reshaping failed")
end

println("\n🎉 Manual testing completed!")
println("Check the output above for any errors marked with ✗")
