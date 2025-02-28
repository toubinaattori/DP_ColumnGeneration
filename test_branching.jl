using DecisionProgramming
using JuMP, Gurobi, DataStructures, Statistics
using Base.Iterators: product
using Distributed, Hwloc
addprocs(4, topology = :master_worker)
include("new_inequality_labeling_w_branching.jl")


problem = "FORTIFICATION"
problem = "PIGFARM"
No = 7


c_k = rand(1:50,No)
b = 0.03
fortification(k, a) = [c_k[k], 0][a]

@info("Creating the influence diagram.")
diagram = InfluenceDiagram()

add_node!(diagram, ChanceNode("L", [], ["high", "low"]))

for i in 1:No
    add_node!(diagram, ChanceNode("R$i", ["L"], ["high", "low"]))
    add_node!(diagram, DecisionNode("A$i", ["R$i"], ["yes", "no"]))
end

add_node!(diagram, ChanceNode("F", ["L", ["A$i" for i in 1:No]...], ["failure", "success"]))

add_node!(diagram, ValueNode("T", ["F", ["A$i" for i in 1:No]...]))

generate_arcs!(diagram)

X_L = [rand(), 0]
X_L[2] = 1.0 - X_L[1]
add_probabilities!(diagram, "L", X_L)

for i in 1:No
    x_R, y_R = rand(2)
    X_R = ProbabilityMatrix(diagram, "R$i")
    X_R["high", "high"] = max(x_R, 1-x_R)
    X_R["high", "low"] = 1 - max(x_R, 1-x_R)
    X_R["low", "low"] = max(y_R, 1-y_R)
    X_R["low", "high"] = 1-max(y_R, 1-y_R)
    add_probabilities!(diagram, "R$i", X_R)
end

X_F = ProbabilityMatrix(diagram, "F")
x_F, y_F = rand(2)
for s in paths([State(2) for i in 1:No])
    denominator = exp(b * sum(fortification(k, a) for (k, a) in enumerate(s)))
    X_F[1, s..., 1] = max(x_F, 1-x_F) / denominator
    X_F[1, s..., 2] = 1.0 - X_F[1, s..., 1]
    X_F[2, s..., 1] = min(y_F, 1-y_F) / denominator
    X_F[2, s..., 2] = 1.0 - X_F[2, s..., 1]
end
add_probabilities!(diagram, "F", X_F)

Y_T = UtilityMatrix(diagram, "T")
for s in paths([State(2) for i in 1:No])
    cost = sum(-fortification(k, a) for (k, a) in enumerate(s))
    Y_T[1, s...] = 1000 + cost
    Y_T[2, s...] = 1100 + cost
end
add_utilities!(diagram, "T", Y_T)


generate_diagram!(diagram,positive_path_utility=true)
dt2 = @elapsed begin
    # model2, z, μ_s = generate_model(diagram, model_type="DP")
    model2 = Model()
    z = DecisionVariables(model2, diagram, names=false)
    variables = RJTVariables(model2, diagram, z, names=false)
    EV = expected_value(model2, diagram, variables)
    @objective(model2, Max, EV)
    
    @info("Starting the optimization process.")
    optimizer = optimizer_with_attributes(Gurobi.Optimizer,"OptimalityTol" => 1e-09, "MIPGap" => 0)
    set_optimizer(model2, optimizer)
    
    optimize!(model2)
    println(objective_value(model2) - diagram.translation)
end
dt3 = @elapsed begin
    (dt4,obj3,max_branch3) = column_generation_DP_ordered_lists(diagram)
end
dt4 = @elapsed begin
    (a,b,c,d,e) = solve_model_with_paths_inequality_MIP(diagram,0.0)
end
println("Full model time: ",dt4)
println("Objective full model: ", c)
println("Objective RJT: ", objective_value(model2))
println("OrderedLists time: ", dt3)
println("RJT time: ", dt2)
for i in keys(z)
    println(i)
    println(value.(z[i].z))
end
println(objective_value(model2) - diagram.translation)
rmprocs(2:workers()[end])

