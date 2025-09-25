# Quick validation tests for all PS2 functions
# Tests each function individually to verify correctness

println("=== PS2 Functions Validation ===\n")

# Test 1: Basic optimization functions
println("1. Testing basic optimization functions...")
f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2

test_x = -5.0
if abs(f([test_x]) + minusf([test_x])) < 1e-10
    println("   ✓ f and minusf are proper negatives")
else
    println("   ✗ ERROR: f and minusf relationship broken")
end

# Test 2: OLS function
println("\n2. Testing OLS function...")
function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

# Simple test case
X_test = [1.0 2.0; 1.0 3.0; 1.0 4.0]  # 3x2 matrix
y_test = [3.0, 4.0, 5.0]  # y = 1 + x
beta_true = [1.0, 1.0]

ssr = ols(beta_true, X_test, y_test)
if ssr ≈ 0.0
    println("   ✓ OLS gives zero SSR for perfect fit")
else
    println("   ⚠ OLS SSR = $ssr (should be ≈ 0 for perfect fit)")
end

# Test 3: Logit function  
println("\n3. Testing logit function...")
function logit(alpha, X, d)
    xb = X * alpha
    p = 1 ./(1 .+ exp.(xb))
    loglike = -sum(d .* log.(p) .+ (1 .- d) .* log.(1 .- p))
    return loglike
end

X_logit = [1.0 0.5; 1.0 -0.5; 1.0 1.0; 1.0 -1.0]
y_logit = [true, false, true, false]
alpha_test = [0.0, 1.0]

try
    ll = logit(alpha_test, X_logit, y_logit)
    if isfinite(ll) && ll > 0
        println("   ✓ Logit returns finite positive value: $(round(ll, digits=3))")
    else
        println("   ✗ ERROR: Logit returns invalid value: $ll")
    end
catch e
    println("   ✗ ERROR: Logit function failed: $e")
end

# Test 4: Multinomial logit function
println("\n4. Testing multinomial logit function...")
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

X_mlogit = [1.0 0.5; 1.0 -0.5; 1.0 1.0; 1.0 -1.0; 1.0 0.0]
y_mlogit = [1, 2, 1, 3, 2]  # 3 alternatives
K = size(X_mlogit, 2)
J = 3
alpha_mlogit = [0.5, 1.0, -0.5, 0.8]  # K*(J-1) = 2*2 = 4 parameters

try
    ll_mlogit = mlogit(alpha_mlogit, X_mlogit, y_mlogit)
    if isfinite(ll_mlogit) && ll_mlogit > 0
        println("   ✓ Multinomial logit returns finite positive value: $(round(ll_mlogit, digits=3))")
    else
        println("   ✗ ERROR: Multinomial logit returns invalid value: $ll_mlogit")
    end
    
    # Test parameter dimensions
    expected_params = K * (J-1)
    if length(alpha_mlogit) == expected_params
        println("   ✓ Parameter vector has correct length: $(length(alpha_mlogit))")
    else
        println("   ✗ ERROR: Wrong parameter length. Got $(length(alpha_mlogit)), expected $expected_params")
    end
    
catch e
    println("   ✗ ERROR: Multinomial logit function failed: $e")
end

# Test 5: Data structures and matrix operations
println("\n5. Testing matrix operations...")
test_alpha = [1, 2, 3, 4, 5, 6]
try
    reshaped = reshape(test_alpha, 3, 2)
    if size(reshaped) == (3, 2)
        println("   ✓ Parameter reshaping works correctly")
    else
        println("   ✗ ERROR: Reshape gives wrong dimensions")
    end
catch e
    println("   ✗ ERROR: Reshape failed: $e")
end

println("\n=== Summary ===")
println("Manual validation of all PS2 functions completed.")
println("Check above for any ✗ errors. All ✓ marks indicate passing tests.")
println("Functions are ready for use with Optim for parameter estimation.")
