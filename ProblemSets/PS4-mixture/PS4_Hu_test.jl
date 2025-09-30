using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions

cd(@__DIR__)

Random.seed!(1234)  

include("PS4_Hu_source.jl")

"""
Comprehensive unit tests for PS4 source code
"""

@testset "load_data function" begin
	df, X, Z, y = load_data()
	@test size(X, 2) == 3
	@test size(Z, 2) == 8
	@test length(y) == size(X, 1)
	@test length(y) == size(Z, 1)
	@test isa(df, DataFrame)
end

@testset "mlogit_with_Z function" begin
	df, X, Z, y = load_data()
	K = size(X, 2)
	J = length(unique(y))
	theta = [randn(K*(J-1)); 0.1]
	loglike = mlogit_with_Z(theta, X, Z, y)
	@test isfinite(loglike)
	@test loglike > 0
end

@testset "optimize_mlogit function" begin
	df, X, Z, y = load_data()
	theta_hat, theta_se = optimize_mlogit(X, Z, y)
	K = size(X, 2)
	J = length(unique(y))
	@test length(theta_hat) == K*(J-1)+1
	@test length(theta_se) == K*(J-1)+1
	@test all(isfinite, theta_hat)
	@test all(isfinite, theta_se)
end

@testset "practice_quadrature function" begin
	try
		practice_quadrature()
		@test true
	catch e
		@test false "practice_quadrature errored: $e"
	end
end

@testset "variance_quadrature function" begin
	try
		variance_quadrature()
		@test true
	catch e
		@test false "variance_quadrature errored: $e"
	end
end

@testset "practice_monte_carlo function" begin
	try
		practice_monte_carlo()
		@test true
	catch e
		@test false "practice_monte_carlo errored: $e"
	end
end

@testset "mixed_logit_quad function (setup only)" begin
	df, X, Z, y = load_data()
	K = size(X, 2)
	J = length(unique(y))
	theta = [randn(K*(J-1)); 0.1; 1.0]
	nodes, weights = lgwt(7, -4, 4)
	try
		mixed_logit_quad(theta, X, Z, y, nodes, weights)
		@test true
	catch e
		@test false "mixed_logit_quad errored: $e"
	end
end

@testset "mixed_logit_mc function (setup only)" begin
	df, X, Z, y = load_data()
	K = size(X, 2)
	J = length(unique(y))
	theta = [randn(K*(J-1)); 0.1; 1.0]
	D = 10  # keep small for test speed
	try
		mixed_logit_mc(theta, X, Z, y, D)
		@test true
	catch e
		@test false "mixed_logit_mc errored: $e"
	end
end
