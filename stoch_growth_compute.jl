using Parameters, Plots #import the libraries we want
using Distributed
using SharedArrays

include("stoch_growth_model.jl") #import the functions that solve our growth model


prim, res = Initialize() #initialize primitive and results structs
@elapsed Solve_model(prim, res) #solve the model!
@unpack val_func, pol_func = res
@unpack k_grid = prim

#parallel
nprocs()
addprocs(3)
workers()
prim, res = Initialize() #initialize primitive and results structs
@elapsed Solve_model_parall(prim, res) #solve the model!
@unpack val_func, pol_func = res
@unpack k_grid = prim
############## plots


#value function
Plots.plot(k_grid, val_func[:,2], title="Value Function", label = "low")
Plots.plot!(k_grid, val_func[:,1], title="Value Function", label = "high")
Plots.savefig("02_Value_Functions.png")


#policy functions
Plots.plot(k_grid, pol_func[:,2], title="Policy Functions", label = "policy low")
Plots.plot!(k_grid, pol_func[:,1], title="Policy Functions", label = "policy high")
Plots.savefig("02_Policy_Functions.png")

#changes in policy function
pol_func_δ = copy(pol_func).-k_grid
Plots.plot(k_grid, pol_func_δ[:,2], title="Policy Functions Changes",label="low")
Plots.plot!(k_grid, pol_func_δ[:,1], title="Policy Functions Changes",label="high")

Plots.savefig("02_Policy_Functions_Changes.png")


println("All done!")
################################
