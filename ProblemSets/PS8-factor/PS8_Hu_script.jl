using Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM, MultivariateStats, FreqTables, ForwardDiff, LineSearches

# Set working directory
cd(@__DIR__)

# Input source code
include("PS8_Hu_source.jl")

main()