using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

cd(@__DIR__)

include("PS3_Hu_source.jl")

allwrap()

# Questioon: Interpret the estimated coefﬁcient γ
# Estimate gamma is -0.094
# Gamma represents te change in latent utlity
# with 1 unit increase in the log wage
# in occupations j(relative to other)
# It's surprising that gamma is negative, because I thoughtpeople liked earning more money