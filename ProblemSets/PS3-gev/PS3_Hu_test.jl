using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

cd(@__DIR__)

include("PS3_Hu_source.jl")
"""
Unit tests for multinomial logit workflow
"""

@testset "load_data function" begin
	url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS3-gev/nlsw88w.csv"
	X, Z, y = load_data(url)
	@test size(X, 2) == 3
	@test size(Z, 2) == 8
	@test length(y) == size(X, 1)
	@test length(y) == size(Z, 1)
end

@testset "mlogit_with_Z function" begin
	url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS3-gev/nlsw88w.csv"
	X, Z, y = load_data(url)
	# Construct a random theta of correct length
	K = size(X, 2)
	J = length(unique(y))
	theta = [randn(K*(J-1)); 0.1]
	loglike = mlogit_with_Z(theta, X, Z, y)
	@test isfinite(loglike)
	@test loglike > 0
end

@testset "optimize_mlogit function" begin
	url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS3-gev/nlsw88w.csv"
	X, Z, y = load_data(url)
	theta_hat = optimize_mlogit(X, Z, y)
	@test length(theta_hat) == 22
	@test all(isfinite, theta_hat)
end

"""
Unit tests for nested logit workflow
"""

@testset "nested_logit_with_Z function" begin
	url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS3-gev/nlsw88w.csv"
	X, Z, y = load_data(url)
	K = size(X, 2)
	J = length(unique(y))
	nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]
	# theta: 6 alphas + 2 lambdas + 1 gamma
	theta = [randn(2*K); 0.5; 0.5; 0.1]
	loglike = nested_logit_with_Z(theta, X, Z, y, nesting_structure)
	@test isfinite(loglike)
	@test loglike > 0
end

@testset "optimize_nested_logit function" begin
	url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS3-gev/nlsw88w.csv"
	X, Z, y = load_data(url)
	nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]
	theta_hat = optimize_nested_logit(X, Z, y, nesting_structure)
	@test length(theta_hat) == 7
	@test all(isfinite, theta_hat)
end

