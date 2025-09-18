using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables

cd(@__DIR__)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::
f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
startval = rand(1)   # random starting value
result = optimize(minusf, startval, BFGS())
println("argmin (minimizer) is ",Optim.minimizer(result)[1])
println("min is ",Optim.minimum(result))

result_better = optimize(minusf, [-7.0], BFGS())
println(result_better)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(beta_hat_ols.minimizer)

bols = inv(X'*X)*X'*y

#standarrd errors
N = size(X,1)
K = size(X,2)
MSE = sum((y .- X*bols).^2)/(N-K)
VCOV = MSE*inv(X'*X)
se_bols = sqrt.(diag(VCOV))
println("standard errors: ", se_bols)

println("OLS closed form: ", bols)
df.white = df.race.==1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function logit(alpha, X, d)
    xb = X * alpha
    p = 1 ./(1 .+ exp.(xb))
    loglike = -sum(d .* log.(p) .+ (1 .- d) .* log.(1 .- p))
    return loglike
end

startval = rand(size(X,2))  
alpha_hat = optimize(alpha -> logit(alpha, X, y), startval, BFGS())

println("Logit estimation results:" , Optim.minimizer(alpha_hat))

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 4 - Compare with GLM
#:::::::::::::::::::::::::::::::::::::::::::::::::::

blogit_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
println("\nGLM logit results for comparison:")
println("GLM coefficients: ", coef(blogit_glm))

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5 
#:::::::::::::::::::::::::::::::::::::::::::::::::::
freqtable(df, :occupation) 
df = dropmissing(df, :occupation)
df[df.occupation.==8 ,:occupation] .= 7
df[df.occupation.==9 ,:occupation] .= 7
df[df.occupation.==10,:occupation] .= 7
df[df.occupation.==11,:occupation] .= 7
df[df.occupation.==12,:occupation] .= 7
df[df.occupation.==13,:occupation] .= 7
freqtable(df, :occupation) 

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

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

    alpha_zero=zeros(6*size(X,2))
    alpha_rand= rand(6*size(X,2))

    alpha_hat_mlogit = optimize(a -> mlogit(a, X, y), alpha_rand, BFGS())
    println("Multinomial logit estimation results:" , Optim.minimizer(alpha_hat_mlogit))
    

