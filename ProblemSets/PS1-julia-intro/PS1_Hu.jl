using JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions

# Set the seed
Random.seed!(1234)

function.q1()
#--------------------------------------------------------------------------------------------------------------------------
# Question 1, part (a)
#--------------------------------------------------------------------------------------------------------------------------
# Draw uniform random numbers
A = -5 .+ 15*rand(10,7)
A = rand(Uniform(-5, 10), (10, 7))

# Draw uniform random numbers
B = -2 .+ 15*rand(10,7)
B = rand(Normal(-2, 15), (10, 7))

# Indexing
C =[A[1:5, 1:5] B[1:5, 6:7]]

# Bit Array / dummy variable
D =A.*(A .<= 0)

#--------------------------------------------------------------------------------------------------------------------------
# Question 1, part (b)
#--------------------------------------------------------------------------------------------------------------------------
size (A)
size (A, 1) * size(A, 2)
length(A)
size(A,[:])

#--------------------------------------------------------------------------------------------------------------------------
# Question 1, part (c)
#--------------------------------------------------------------------------------------------------------------------------
size (D)
length(unique(D))

#--------------------------------------------------------------------------------------------------------------------------
# Question 1, part (d)
#--------------------------------------------------------------------------------------------------------------------------
E = reshape(B, 70, 1)
E = reshape(B, (70, 1))
E = reshape(B, length(B), 1)
E = reshape(B, size(B,1)*size(B,2), 1)
E = B [:]

#--------------------------------------------------------------------------------------------------------------------------
# Question 1, part (e)
#--------------------------------------------------------------------------------------------------------------------------
F = cat(A, B; dims=3)

#--------------------------------------------------------------------------------------------------------------------------
# Question 1, part (f)
#--------------------------------------------------------------------------------------------------------------------------
F = permutedims(F, (3, 1, 2))

#--------------------------------------------------------------------------------------------------------------------------
# Question 1, part (g)
#--------------------------------------------------------------------------------------------------------------------------
G = kron(B, C)
#G = kron(C, F) # this does not work because of dimension mismatch

#--------------------------------------------------------------------------------------------------------------------------
# Question 1, part (h)
#--------------------------------------------------------------------------------------------------------------------------
# save the matrces A, B, C, D, E, F, G in a file called "matrixpractice.jld"
save("matrixpractice.jld", "A", A, "B", B, "C", C, "D", D, "E", E, "F", F, "G", G)

save("firstmatrix.jld", "A", A, "B", B, "C", C, "D", D)


save("Cmatrix.jld", "C", C)# Save the matrix C in a file called "Cmatrix.jld"


# Save the matrix C in a CSV file called "Cmatrix.csv"
CSV.write("Cmatrix.csv", DataFrame(C, :auto))

df_D = DataFrame(D, :auto)
CSV.write("Dmatrix.dat", df_D, delim="\t")

DataFrame(D, :auto) |> x -> (df -> CSV.write("Dmatrix2.dat", df, delim="\t"))

return A, B, C, D
end

#call the function from q1
A, B, C, D = q1()

function q2(A,B)
#part a
    AB = zeros(size(A))
    for r in 1:eachindex(A[:, 1])
        for c in 1:eachindex(A[1: , 2])
            AB[r, c] = A[r, c] * B[r, c]
        end
    end
    AB = A .* B

function q2(A, B, C, D)
#part b
#find all elements in C that are between -5 and 5 (inclusive)
    Cprime = Float64[]
    for c in axes(C, 2)
        for r in axes(C,1])
            if C[r, c] >= -5 && C[r, c] <= 5
                push!(Cprime, C[r, c])
            end
        end
    end

Cprime = C[C .>= -5 .& (C .<= 5)]

#compare the two vectors
Cprime == Cprime2
if Cprime !== Cprime2
    @show Cprime
    @show Cprime2
    @show Cprime == Cprime2
    error("Cprime and Cprime2 are not the same.")
end

#part c
N = 15_169
K = 6
T = 5
#ordering of 2nd dimension
#intercept
#dummy variable
#continuous variable (normal)
#normal
#binomial
#another binomial
for i in axes(X,1)
    X[i, 1, :] .= 1.0
    X[i, 5, :] .= rand(Binomial(20, 0.6))
    X[i, 6, :] .= rand(Binomial(20, 0.5))
    for t in axes(X,3)
        X[i, 2, t] .= rand() <= .75* (6-t)/5
        X[i, 3, t] .= rand(Normal(15 + t - 1, 5*(t-1)))
        X[i, 4, t] .= rand(Normal(π*(6-t), 1/exp(1)))
    end
end

#part d
    β =zeros(K, T)
    β[1, :]=[1+0.25*(t-1) for t in 1:T]
    β[2, :]=[log(t) for t in 1:T]
    β[3, :]=[-sqrt(t) - exp(t+1) for t in 1:T]
    β[4, :]=[exp(t) - exp(t+1) for t in 1:T]
    β[5, :]=[t for t in 1:T]
    β[6, :]=[t/3 for t in 1:T]

    #part e
    Y = [X[:, :, t] .* β[:, t] .+ rand(Normal(0, 0.36), N) for t in 1:T]

    return nothing
end

function q3()
#Question 3, part (a)
    df = DataFrame(CSV.File("nlsw88.csv") )
    @show df[1:5, :]
    @show type(df[:, :grade])
    CSV.write("nlsw88_cleaned.csv", df)


#part b
    @show mean(df[:, never_married])

#part c
    @show freqtable(df[:, :race]) 

#part d
    vars = names(df)
    summarystats = describe(df)
    @show summarystats

#part e
#cross tabulation of race and grade
    @show freqtable(df[:, :rindustry], df[:, :occupation])

#part f
    df_sub = df[:, [:industry, :occupation, :wage]]
    grouped = groupby(df_sub, [:industry, :occupation])
    mean_wage = combine(grouped, :wage => mean => :mean_wage)
    @show mean_wage

function q3()
    mean_wage = combine(grouped, :occupation], :wage => mean => :mean_wage)
    @show mean_wage

    return nothing
end


# part (b) of question 4
"""
matrixops(A, B)

"""

function matrixops(A::Array{Float64}, B::Array{Float64})
    #part (e) of question 4: check dimensions
    if size(A) != size(B)
        error("A and B must have the same dimensions.")
    end
    # (i) the element-by-element product of A and
    out1 = A .* B
    #(ii) the product A′ and B
    out2 = A' * B
    #(iii) the sum of all the elements of A and B.
    out3 = sum(A + B)
    return nothing
end

function q4()
    #part (a)
    #three ways to load the  .jld file
    #load("matrixpractice.jld", "A", "B", "C", "D", "E", "F", "G")
    @load "matrixpractice.jld"
    #part (d)
    matrixops(A, B)

    #part (f)
    try
        matrixops(C,D)
    catch e
        printhln("Trying matrixops(C, D):")
        println(e)
    end 

    # part (g) of question 4
    # read in processed CSV file
    nlsw88 = DataFrame(CSV.File("nlsw88_cleaned.csv"))
    ttl_exp = convert(Array, nlsw88.ttl_exp)
    wage = convert(Array, nlsw88.wage)
    matrixops(ttl_exp, wage)

    return nothing
end

#call the function from q1
A, B, C, D = q1()

#call the function from q2
q2(A, B, C)

#call the function from q3

#call the function from q3
q3()

#call the function from 4


