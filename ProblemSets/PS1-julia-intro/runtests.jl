#!/usr/bin/env julia

"""
Test Runner for PS1 Julia Assignment

This script runs the unit tests for PS1_Hu.jl functions.
Usage: julia runtests.jl
"""

using Pkg

# Check if Test package is available
try
    using Test
catch
    println("Installing Test package...")
    Pkg.add("Test")
    using Test
end

# Run the tests
println("="^60)
println("Running PS1 Unit Tests")
println("="^60)

include("test_PS1_Hu.jl")

println("\n" * "="^60)
println("Test run completed!")
println("="^60)
