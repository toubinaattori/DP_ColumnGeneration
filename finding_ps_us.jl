using DecisionProgramming
using JuMP, Gurobi, DataStructures, Statistics
using Base.Iterators: product
include("new_inequality_labeling.jl")

No = 11
resps = zeros(2,20)
for nn in 1:20

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

X_Rs = []

for i in 1:No
    x_R, y_R = rand(2)
    X_R = ProbabilityMatrix(diagram, "R$i")
    X_R["high", "high"] = max(x_R, 1-x_R)
    X_R["high", "low"] = 1 - max(x_R, 1-x_R)
    X_R["low", "low"] = max(y_R, 1-y_R)
    X_R["low", "high"] = 1-max(y_R, 1-y_R)
    add_probabilities!(diagram, "R$i", X_R)
    push!(X_Rs, X_R)
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


 generate_diagram!(diagram)

I_j_indices_result = I_j_indices(diagram, diagram.Nodes)
z_indices = indices(diagram.D)
all_c_indices = indices(diagram.C)
decision_set = unique([I_j_indices_result[z_indices]...; z_indices...])
c_indices_in_decision_set = filter(c -> c in decision_set, all_c_indices)
index_o = Dict(zip(decision_set,1:length(decision_set)))
dims1 = States(get_values(diagram.S))[decision_set]
c_indices = filter(c -> c in decision_set, indices(diagram.C))
c_indices_ds = [index_o[c] for c in c_indices]
c_index = Dict(zip([c for (c,n) in collect(diagram.Nodes)[c_indices_in_decision_set]],1:length(c_indices_in_decision_set)))
dims = States(get_values(diagram.S))[[c_indices;]]
minimum_largest = 1
results = []
dims_decisions = States(get_values(diagram.S))[z_indices]


s_c = [1 for i in 1:No]
dt = @elapsed begin
        model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(Gurobi.Env())))
        d = Dict{Int64,Dict{Int64,VariableRef}}()
        for j in 1:No
            d[j] = Dict{Int64,VariableRef}()
            for i in 1:2
                d[j][i] = @variable(model, binary = true)
                #d[j][i] = @variable(model, upper_bound = 1, lower_bound = 0)
            end
            @constraint(model, sum(d[j][i] for i in 1:2) == 1)
        end
        @variable(model, p_last[1:2,1:2], upper_bound = 1, lower_bound = 0)
        @constraint(model,[s1 in 1:2, slast in 1:2], p_last[s1,slast] <= maximum(X_F[s1,[Colon() for i in 1:No]...,slast]))
        @constraint(model, [s1 in 1:2], sum(p_last[s1,slast] for slast in 1:2) == 1)
        @variable(model, U[1:2],lower_bound = 0)
        @constraint(model,[slast in 1:2], U[slast] <= maximum(Y_T[slast,[Colon() for i in 1:No]...]))
        @objective(model, Max, sum(X_L[s1]*prod(X_Rs[i][s1,s_c[i]] for i in 1:No)*p_last[s1,slast]*U[slast] for s1 in 1:2, slast in 1:2))
        for s in paths(dims_decisions)
            @constraint(model, [s1 in 1:2, slast in 1:2], p_last[s1,slast] <= X_F[s1,s...,slast] + No - sum(d[j][s[j]] for j in 1:No))
            @constraint(model, [slast in 1:2], U[slast] <= Y_T[slast,s...] + 1200*(No - sum(d[j][s[j]] for j in 1:No)))
        end



        optimize!(model)
        println(objective_value(model))

        println(value.(p_last))
        println(value.(U))
end

dt2 = @elapsed begin
    bloood = -1.2
    bloood_p = []
    for s in paths(dims1, FixedPath(zip(c_indices_ds,[s_c...])))
        # global bloood
        # global bloood_p
        pu = sum(diagram.P(s2)*(diagram.U(s2)) for s2 in paths(States(get_values(diagram.S)), FixedPath(zip(decision_set,[s...]))))
        if pu >= bloood
            bloood = pu
            bloood_p = [s]
        end
    end

end
println(bloood)
println(bloood_p)
println(dt)
println(dt2)
    resps[1,nn] = dt
    resps[2,nn] = dt2

end

println(mean(resps[1,:]))
println(mean(resps[2,:]))