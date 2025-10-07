# Read in function to create state transitions for dynamic model
include("create_grids.jl")

################################################################################
# PART 1: DATA LOADING AND PREPARATION (Pre-implemented)
################################################################################

"""
    load_static_data()

Load and reshape data for static estimation (Questions 1-2).
Returns a long-format DataFrame ready for GLM estimation.

This is pre-implemented so you can focus on the modeling rather than 
data wrangling. The key is understanding that we're converting from:
  - Wide format: Y1, Y2, ..., Y20 (20 columns per bus)
  - Long format: One row per bus-time observation (N×T rows total)
"""
function load_static_data()
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    
    # Create bus id variable
    df = @transform(df, :bus_id = 1:size(df,1))
    
    #---------------------------------------------------
    # Reshape from wide to long (done twice because 
    # stack() requires doing one variable at a time)
    #---------------------------------------------------
    
    # First reshape the decision variable (Y1-Y20)
    dfy = @select(df, :bus_id, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10,
                      :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20,
                      :RouteUsage, :Branded)
    dfy_long = DataFrames.stack(dfy, Not([:bus_id, :RouteUsage, :Branded]))
    rename!(dfy_long, :value => :Y)
    dfy_long = @transform(dfy_long, :time = kron(collect(1:20), ones(size(df,1))))
    select!(dfy_long, Not(:variable))
    
    # Next reshape the odometer variable (Odo1-Odo20)
    dfx = @select(df, :bus_id, :Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7, 
                      :Odo8, :Odo9, :Odo10, :Odo11, :Odo12, :Odo13, :Odo14, :Odo15,
                      :Odo16, :Odo17, :Odo18, :Odo19, :Odo20)
    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    rename!(dfx_long, :value => :Odometer)
    dfx_long = @transform(dfx_long, :time = kron(collect(1:20), ones(size(df,1))))
    select!(dfx_long, Not(:variable))
    
    # Join reshaped dataframes back together
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id, :time])
    sort!(df_long, [:bus_id, :time])
    
    return df_long
end

function load_dynamic_data()
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    
    # Extract matrices and vectors
    Y = Matrix(df[:, r"^Y\d+"])       # Decision variables (N × T)
    X = Matrix(df[:, r"^Odo\d+"])     # Odometer readings (N × T)
    Xstate = Matrix(df[:, r"^Xst"])   # Discretized mileage state (N × T)
    Zstate = Vector(df[:, :Zst])      # Discretized route usage state (N × 1)
    B = Vector(df[:, :Branded])       # Brand indicator (N × 1)
    
    N, T = size(Y)
    
    # Create state grids
    zval, zbin, xval, xbin, xtran = create_grids()
    
    # Bundle everything into a named tuple for clean code
    # This is much better than passing 10+ separate arguments!
    return (
        # Data
        Y = Y,
        X = X,
        B = B,
        Xstate = Xstate,
        Zstate = Zstate,
        # Dimensions
        N = N,
        T = T,
        # State space
        xval = xval,
        xbin = xbin,
        zbin = zbin,
        xtran = xtran,
        # Parameters
        β = 0.9
    )
end

################################################################################
# PART 2: STATIC ESTIMATION (Question 2)
################################################################################

function estimate_static_model(df_long)
    # Estimate the logit model
    θ̂_static = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink())
    println(θ̂_static)
    return θ̂_static
end

################################################################################
# PART 3: DYNAMIC ESTIMATION - FUTURE VALUE COMPUTATION (Question 3c)
################################################################################

@views @inbounds function compute_future_value!(FV, θ, d)
    # FV is already initialized to zeros in the calling function
    # FV[T+1] stays at zero (terminal condition)
    
    # Loop backward over time
    for t in d.T:-1:1
        # Loop over brand states
        for b in 0:1
            # Loop over route usage states (permanent characteristic)
            for z in 1:d.zbin
                # Loop over mileage states (dynamic characteristic)
                for x in 1:d.xbin

                     # Calculate row index in transition matrix
                     # This indexes the joint state (z,x)
                     row = x + (z-1)*d.xbin

                     # The expectation is: xtran[row,:]' · FV[z's rows, b+1, t+1]
                     # where "z's rows" means all x values for the given z
                     v1 = θ[1] + θ[2]*d.xval[x] + θ[3]*b + d.xtran[row,:]⋅FV[(z-1)*d.xbin+1:z*d.xbin, b+1, t+1]
                     
                     # Compute v₀: value of replacing engine
                     # Flow utility: normalized to 0 (replacement is the reference)
                     # Future value: E[V_{t+1} | replace, current z]
                
                     # Key difference: after replacement, mileage resets to 0
                     # So we use row = 1 + (z-1)*xbin (first mileage bin)
                     v0 =  d.xtran[1+(z-1)*d.xbin,:]⋅FV[(z-1)*d.xbin+1:z*d.xbin, b+1, t+1]

                     # Store future value using log-sum-exp formula
                     # FV(t) = β·log(exp(v₀) + exp(v₁))
                     # This is the expected value before the agent observes ε
                     FV[row, b+1, t] =  d.β * log(exp(v1) + exp(v0)) + d.β * Base.MathConstants.eulergamma  
                     
                 end
             end
         end
     end
    
    return FV
end

################################################################################
# PART 4: DYNAMIC ESTIMATION - LOG LIKELIHOOD (Question 3d)
################################################################################

@views @inbounds function log_likelihood_dynamic(θ, d)
    # First, compute future values for all states given current θ
    FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
    # Solve for future values
    compute_future_value!(FV, θ, d)
    
    # Now compute likelihood using only observed states
    loglike = 0.0

    # Loop over individuals
    for i in 1:d.N

        # Pre-compute the row index for replacement (mileage = 0)
        # This is constant across time for each individual
        row0 = (d.Zstate[i] - 1) * d.xbin + 1

         # Loop over time periods
         for t in 1:d.T
             
             # Get row index for current state (observed mileage and route usage)
             row1 = d.Xstate[i,t] + (d.Zstate[i] - 1) * d.xbin
            
             # Compute conditional value difference: v₁ - v₀
             flow_diff = θ[1] + θ[2]*d.X[i,t] + θ[3]*d.B[i]

             # Part 2: Expected future value difference
             #   E[V|continue] uses transition probs from current state (row1)
             #   E[V|replace] uses transition probs from mileage=0 (row0)
             #   Both look at same FV array, just different transition weights
             # 
             # This is the elegant part: we DIFFERENCE the transition matrices!
             ev_diff = (d.xtran[row1,:] .- d.xtran[row0,:])⋅FV[row0:row0+d.xbin-1, d.B[i]+1, t+1]

             # Total conditional value difference
             v_diff = flow_diff + d.β * ev_diff #don't need β * eulergamma here because the v's are differenced
             
             # Add to log likelihood
             # Efficient form: ℓ = Σ[Y·v_diff - log(1 + exp(v_diff))]
             # This is numerically stable and avoids computing P₀ and P₁ separately
             loglike += (d.Y[i,t] == 1) * v_diff - log(1 + exp(v_diff))
         end
     end
    
    # Return NEGATIVE log likelihood (Optim minimizes)
    return -loglike
end


################################################################################
# PART 5: OPTIMIZATION WRAPPER (Question 3e-h)
################################################################################

function estimate_dynamic_model(d; θ_start=nothing)
    println("="^70)
    println("DYNAMIC MODEL ESTIMATION")
    println("="^70)
    
    # Set starting values
    if isnothing(θ_start)
        θ_start = rand(3)  # Random start if nothing provided
        println("\nUsing random starting values: ", θ_start)
    else
        println("\nUsing provided starting values: ", θ_start)
    end
    
    # Define objective function (just passes θ to likelihood)
    # The data is "captured" by the closure
    objective = θ -> log_likelihood_dynamic(θ, d)
    
    # Time the likelihood evaluation (good practice!)
    println("\nTiming likelihood evaluation...")
    @time objective(θ_start)
    @time objective(θ_start)
    
    # Run optimization
    println("\nOptimizing (this may take several minutes)...")
    result = optimize(objective, θ_start, LBFGS(), 
                                    iterations=100_000, 
                                    show_trace=true,
                                    show_every=10)


    # Display results
    println("\n" * "="^70)
    println("RESULTS")
    println("="^70)
    println("Parameter estimates:")
    println("  θ₀ (constant):     ", round(result.minimizer[1], digits=4))
    println("  θ₁ (mileage):      ", round(result.minimizer[2], digits=4))
    println("  θ₂ (brand):        ", round(result.minimizer[3], digits=4))
    println("\nLog likelihood:      ", round(-result.minimum, digits=2))
    println("Converged:           ", Optim.converged(result))
    println("Iterations:          ", Optim.iterations(result))
    println("="^70)
    
    return nothing
end

################################################################################
# MAIN EXECUTION WRAPPER (Question 3f)
################################################################################

function main()
    println("\n" * "="^70)
    println("PROBLEM SET 5: BUS ENGINE REPLACEMENT MODEL")
    println("="^70)
    
    #---------------------------------------------------------------------------
    # Part 1: Static Estimation (Questions 1-2)
    #---------------------------------------------------------------------------
    println("\n" * "-"^70)
    println("PART 1: STATIC (MYOPIC) ESTIMATION")
    println("-"^70)
    
    # Load data
    println("\nLoading and reshaping data...")
    df_long = load_static_data()
    println("Observations: ", nrow(df_long))
    println("First few rows:")
    println(first(df_long, 5))
    
    # Estimate static model
    println("\nEstimating static logit model...")
    estimate_static_model(df_long)

    
    # Uncomment when static estimation is implemented
     if !isnothing(θ̂_static)
         println("\nStatic estimates:")
         println(θ̂_static)
         
         # Extract coefficients to use as starting values for dynamic model
         θ_start_dynamic = coef(θ̂_static)
     else
         θ_start_dynamic = nothing
     end

    θ_start_dynamic = [2.0, -0.15, 1.0]  # Placeholder starting values
    
    #---------------------------------------------------------------------------
    # Part 2: Dynamic Estimation (Question 3)
    #---------------------------------------------------------------------------
    println("\n" * "-"^70)
    println("PART 2: DYNAMIC ESTIMATION")  
    println("-"^70)
    
    # Load data
    println("\nLoading dynamic data...")
    d = load_dynamic_data()
    println("Buses (N): ", d.N)
    println("Time periods (T): ", d.T)
    println("Mileage bins: ", d.xbin)
    println("Route usage bins: ", d.zbin)
    println("Total state space size: ", d.xbin * d.zbin)
    println("Discount factor (β): ", d.β)
    return
    
    # Estimate dynamic model
    println("\nSetting up dynamic estimation...")
    estimate_dynamic_model(d, θ_start=θ_start_dynamic)
    
    println("\n" * "="^70)
    println("END OF PROBLEM SET")
    println("="^70)
    
    return nothing
end
