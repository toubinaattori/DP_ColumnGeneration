using DecisionProgramming
using JuMP, Gurobi, DataStructures

mutable struct LabelWithUpperBound
    pi::Float64
    theta::Float64
    sigma::Float64
    rc::Float64
    upper_bound::Float64
    visited::Array{Int64}
end

struct DecisionVariable
    D::Name
    I_d::Vector{Name}
    z::Array{VariableRef}
end

function decision_variable_cont(model::Model, S::States, d::Node, I_d::Vector{Node}, names::Bool, base_name::String="")
    # Create decision variables.
    dims = S[[I_d; d]]
    z_d = Array{VariableRef}(undef, dims...)
    for s in paths(dims)
        if names == true
            name = join([base_name, s...], "_")
            z_d[s...] = @variable(model, base_name=name)
        else
            z_d[s...] = @variable(model)
        end
    end
    # Constraints to one decision per decision strategy.
    for s_I in paths(S[I_d])
        @constraint(model, sum(z_d[s_I..., s_d] for s_d in 1:S[d]) == 1)
    end
    return z_d
end


function DecisionVariablesCont(model::Model, diagram::InfluenceDiagram; names::Bool=true)
    decVars = OrderedDict{Name, DecisionVariable}()

    for (key, node) in diagram.D
        states = States(get_values(diagram.S))
        I_d = convert(Vector{Node}, indices_in_vector(diagram, diagram.D[key].I_j))
        base_name = names ? "$(diagram.D[key].name)" : ""

        decVars[key] = DecisionVariable(key, diagram.D[key].I_j, decision_variable_cont(model, states, node.index, I_d, names, base_name)) 
    end

    return decVars
end

function column_generation_DP(diagram::InfluenceDiagram)
    value_node_evaluations = Dict{Int16,Array{String}}()
    for v in keys(diagram.V)
        n = maximum(diagram.Nodes[v].index for v in diagram.Nodes[v].I_j)
        if haskey(value_node_evaluations, n)
            push!(value_node_evaluations[n],v)
        else
            value_node_evaluations[n] = [v]
        end
    end
    max_p = ones(length(diagram.Nodes))
    min_p = ones(length(diagram.Nodes))
    max_u = zeros(length(diagram.Nodes))
    min_u = zeros(length(diagram.Nodes))
    for c in keys(diagram.C)
        max_p[diagram.C[c].index] = maximum(diagram.X[c])
        min_p[diagram.C[c].index] = minimum(diagram.X[c])
    end
    for v in keys(diagram.V)
        max_u[collect(keys(value_node_evaluations))[findfirst(x -> v in value_node_evaluations[x], collect(keys(value_node_evaluations)))]] = maximum(diagram.Y[v])
        min_u[collect(keys(value_node_evaluations))[findfirst(x -> v in value_node_evaluations[x], collect(keys(value_node_evaluations)))]] = minimum(diagram.Y[v])
    end
    pathss = []

    # TODO: Run heuristics before entering the labeling algorithm

    # Labeling algorithm starts here
    # TODO: Save p(s)u(s) once calculated
    labeling_with_pruning(diagram, pathss,max_p, min_p, max_u, min_u, value_node_evaluations)
    #labeling_without_pruning(diagram, pathss)
end

function labeling_with_pruning(diagram::InfluenceDiagram, pathss::Vector{Any}, max_p::Array{Float64}, min_p::Array{Float64}, max_u::Array{Float64}, min_u::Array{Float64},value_node_evaluations::Dict{Int16,Array{String}})
    last_node = diagram.Nodes[collect(keys(diagram.Nodes))[findfirst(key -> isa(diagram.Nodes[key],ValueNode) , collect(keys(diagram.Nodes))) - 1]]
    maximum_reduced_cost = -1
    maximum_label = ones(length(diagram.C)+length(diagram.D))
    N = length(diagram.C) + length(diagram.D)
    iteration = 0
    while 1 == 1
        # if iteration >= 20
        #     println("optimal_found")
        #     break
        # end
        iteration = iteration + 1
        model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(Gurobi.Env())))
        z = DecisionVariablesCont(model, diagram)

        x_s = Dict{Path{N}, VariableRef}(
            s => @variable(model, base_name="$(s)", lower_bound = 0, upper_bound = 1)
            for s in pathss)

        I_j_indices_result = I_j_indices(diagram, diagram.Nodes)
        z_indices = indices(diagram.D)
        z_z = [decision_node.z for decision_node in get_values(z)]
        mu = Dict{Tuple{Int16,Vector{Int16}}, ConstraintRef}()
    
        for (d, z_d) in zip(z_indices, z_z)
            dims = States(get_values(diagram.S))[[I_j_indices_result[d]; d]]
            existing_paths = keys(x_s)
            for s_d_s_Id in paths(dims)
                mu[(d,collect(s_d_s_Id))] = @constraint(model,  sum(x_s[s] for s in filter(s -> s[[I_j_indices_result[d]; d]] == s_d_s_Id, existing_paths)) ≤ z_d[s_d_s_Id...] * 100)
            end
        end
        @objective(model, Max, sum( diagram.P(s)*diagram.U(s)*x_s[s] for s in keys(x_s)))

        optimize!(model)

        max_duals = zeros(length(diagram.Nodes))
        min_duals = zeros(length(diagram.Nodes))
        for dd in diagram.D
            d = dd[2].index
            dims = States(get_values(diagram.S))[[I_j_indices_result[d]; d]]
            max_duals[d] = maximum([dual.(mu[(d,collect(s_d_s_Id))]) for s_d_s_Id in paths(dims)])
            min_duals[d] = minimum([dual.(mu[(d,collect(s_d_s_Id))]) for s_d_s_Id in paths(dims)])
        end

        labels = Dict{Tuple{String,Int64},Array{LabelWithUpperBound}}()
        labels[("0",0)] =  [LabelWithUpperBound(1, 0, 0, 0, 0, [])]
        labelmax = LabelWithUpperBound(1, 0, 0, -1, 0, [])
        labelmax_rc = -1
        next_labelmax = LabelWithUpperBound(1, 0, 0, -1, 0, [])
        previous = "0"
        previous_states = [0]
        for node in diagram.Nodes
            if !any(haskey(labels, (previous,s_prev)) for s_prev in previous_states)
                println("optimal_found")
                break
            end
            if node[2] isa ValueNode
                # for s_prev in previous_states, lab in labels[(previous,s_prev)]
                #     label = calculate_label(diagram, node[2], 0, lab, mu,value_node_evaluations)
                #     if Tuple(label.visited) in pathss
                #         continue
                #     end
                #     if !(node[2].index == length(diagram.Nodes)) && (label.theta*prod(max_p[i] for i in node[2].index+1:length(diagram.Nodes)) + label.pi*prod(max_p[i] for i in node[2].index+1:length(diagram.Nodes))*(sum(max_u[i] for i in node[2].index+1:length(diagram.Nodes))) + sum(max_duals[i] for i in node[2].index+1:length(diagram.Nodes))) <= labelmax_rc
                #         continue
                #     end
                #     if !((node[2].name,0) in keys(labels))
                #         labels[(node[2].name,0)] = [label]
                #     else
                #         push!(labels[(node[2].name,0)], label)
                #     end
                #     if next_labelmax.rc <= label.rc
                #         next_labelmax = label
                #     end
                # end
                # delete!(labels, (previous,0))
                # previous = node[2].name
                # previous_states = [0]
                
            else
                for state in 1:length(node[2].states), s_prev in previous_states, lab in labels[(previous,s_prev)]
                    label = calculate_label(diagram, node[2], state, lab, mu,value_node_evaluations)
                    if Tuple(label.visited) in pathss
                        continue
                    end
                    if !(node[1] == last_node.name) && (label.theta*prod(max_p[i] for i in node[2].index+1:length(diagram.Nodes)) + label.pi*prod(max_p[i] for i in node[2].index+1:length(diagram.Nodes))*(sum(max_u[i] for i in node[2].index+1:length(diagram.Nodes))) + sum(max_duals[i] for i in node[2].index+1:length(diagram.Nodes))) <= labelmax_rc
                        continue
                    end
                    if !((node[1],state) in keys(labels))
                        labels[(node[1],state)] = [label]
                    else
                        push!(labels[(node[1],state)], label)
                    end
                    if next_labelmax.rc < label.rc
                        next_labelmax = label
                    end
                end
                # for s_prev in previous_states
                #     delete!(labels, (previous,s_prev))
                # end
                previous = node[2].name
                previous_states = 1:length(node[2].states)
            end
            if !(node[2].index == length(diagram.Nodes))
                labelmax = next_labelmax
                labelmax_rc = labelmax.theta*prod(min_p[i] for i in node[2].index+1:length(diagram.Nodes)) + labelmax.pi*prod(min_p[i] for i in node[2].index+1:length(diagram.Nodes))*(sum(min_u[i] for i in node[2].index+1:length(diagram.Nodes)))+ sum(min_duals[i] for i in node[2].index+1:length(diagram.Nodes))
                next_labelmax = LabelWithUpperBound(1, 0, 0, -1, 0, [])
            end
        end
        if !any(haskey(labels,(last_node.name,state)) for state in 1:length(last_node.states))
            println("moiii")
            println("optimal_found")
            println(length(pathss))
            println(objective_value(model))
            break
        end
        maximum_rc = 0
        maximum_path = []
        for state in 1:length(last_node.states)
            ma = maximum(x -> x.rc ,labels[(last_node.name,state)])
            if ma >= maximum_rc + 1e-6
                maximum_rc = ma
                push!(maximum_path, labels[(last_node.name,state)][findfirst(x -> x.rc == maximum_rc, labels[(last_node.name,state)])].visited)
            end
        end
        if !isempty(maximum_path)
            println(maximum_path[end])
            println(maximum_rc)
            println(length(pathss))
            push!(pathss, Tuple(x for x in maximum_path[end]))
        else
            println("found optimal solution")
            println(length(pathss))
            println(objective_value(model))
            break
        end
    end
end

function labeling_without_pruning(diagram::InfluenceDiagram, pathss::Vector{Any})
    N = length(diagram.C) + length(diagram.D)
    while 1 == 1
        model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(Gurobi.Env())))
        z = DecisionVariablesCont(model, diagram)

        x_s = Dict{Path{N}, VariableRef}(
            s => @variable(model, base_name="$(s)", lower_bound = 0, upper_bound = 1)
            for s in pathss)

        I_j_indices_result = I_j_indices(diagram, diagram.Nodes)
        z_indices = indices(diagram.D)
        z_z = [decision_node.z for decision_node in get_values(z)]
        mu = Dict{Tuple{Int16,Vector{Int16}}, ConstraintRef}()
    
        for (d, z_d) in zip(z_indices, z_z)
            dims = States(get_values(diagram.S))[[I_j_indices_result[d]; d]]
            existing_paths = keys(x_s)
            for s_d_s_Id in paths(dims)
                mu[(d,collect(s_d_s_Id))] = @constraint(model,  sum(x_s[s] for s in filter(s -> s[[I_j_indices_result[d]; d]] == s_d_s_Id, existing_paths)) ≤ z_d[s_d_s_Id...] * 100)
            end
        end
        @objective(model, Max, sum( diagram.P(s)*diagram.U(s)*x_s[s] for s in keys(x_s)))

        optimize!(model)


        labels = Dict{Tuple{String,Int64},Array{LabelWithUpperBound}}()
        labels[("0",0)] =  [LabelWithUpperBound(1, 0, 0, 0, 0, [])]
        previous = "0"
        previous_states = [0]
        for node in diagram.Nodes
            if node[2] isa ValueNode
                for s_prev in previous_states, lab in labels[(previous,s_prev)]
                    label = calculate_label(diagram, node[2], 0, lab, mu)
                    if Tuple(label.visited) in pathss
                        continue
                    end
                    if !((node[2].name,0) in keys(labels))
                        labels[(node[2].name,0)] = [label]
                    else
                        push!(labels[(node[2].name,0)], label)
                    end
                end
                previous = node[2].name
                previous_states = [0]
                continue
            else
                for state in 1:length(node[2].states), s_prev in previous_states, lab in labels[(previous,s_prev)]
                    label = calculate_label(diagram, node[2], state, lab, mu)
                    if Tuple(label.visited) in pathss
                        continue
                    end
                    if !((node[2].name,state) in keys(labels))
                        labels[(node[2].name,state)] = [label]
                    else
                        push!(labels[(node[2].name,state)], label)
                    end
                end
                previous = node[2].name
                previous_states = 1:length(node[2].states)
            end
        end
        maximum_rc = maximum(map(x -> x.rc, labels[(collect(keys(diagram.Nodes))[end],0)]))
        maximum_path = labels[(collect(keys(diagram.Nodes))[end],0)][findfirst(x -> x.rc == maximum_rc, labels[(collect(keys(diagram.Nodes))[end],0)])].visited
        if maximum_rc <= 0
            println("found optimal solution")
            println(length(pathss))
            println(objective_value(model))
            break
        end
        println(maximum_path)
        println(maximum_rc)
        push!(pathss, Tuple(x for x in maximum_path))
    end
end

function calculate_label(diagram::InfluenceDiagram, node::AbstractNode, state::Int64, label::LabelWithUpperBound, mu::Dict{Tuple{Int16,Vector{Int16}}, ConstraintRef},value_node_evaluations::Dict{Int16,Array{String}})
    I_j = findall(x -> x in node.I_j, collect(keys(diagram.Nodes)))
    d =  findfirst(x -> x == node.name, collect(keys(diagram.Nodes)))
    if node isa ChanceNode
        visited = [label.visited..., state]
        pi = label.pi * diagram.X[node.name][state,visited[I_j]...]
        theta = label.theta*diagram.X[node.name][state,visited[I_j]...]
        if haskey(value_node_evaluations, d)
            theta = theta + pi*sum( diagram.Y[n][visited[findall(x -> x in diagram.Nodes[n].I_j, collect(keys(diagram.Nodes)))]...] for n in value_node_evaluations[d])
        end
        sigma = label.sigma
        rc = theta + sigma
        return LabelWithUpperBound(pi, theta, sigma, rc, 0, visited)
    end
    if node isa DecisionNode
        visited = [label.visited..., state]
        pi = label.pi 
        theta = label.theta
        if haskey(value_node_evaluations, d)
            theta = theta + pi*sum( diagram.Y[n][visited[findall(x -> x in diagram.Nodes[n].I_j, collect(keys(diagram.Nodes)))]...] for n in value_node_evaluations[d])
        end
        sigma = label.sigma + dual(mu[(d,[state,visited[I_j]...])])
        rc = theta + sigma
        return LabelWithUpperBound(pi, theta, sigma, rc, 0, visited)
    end
end



No = 4

@info("Creating the influence diagram.")
diagram = InfluenceDiagram()

add_node!(diagram, ChanceNode("H1", [], ["ill", "healthy"]))
for i in 1:No-1
    # Testing result
    add_node!(diagram, ChanceNode("T$i", ["H$i"], ["positive", "negative"]))
    # Decision to treat
    add_node!(diagram, DecisionNode("D$i", ["T$i"], ["treat", "pass"]))
    # Cost of treatment
    add_node!(diagram, ValueNode("V$i", ["D$i"]))
    # Health of next period
    add_node!(diagram, ChanceNode("H$(i+1)", ["H$(i)", "D$(i)"], ["ill", "healthy"]))
end
add_node!(diagram, ValueNode("V$No", ["H$No"]))

generate_arcs!(diagram)

# Add probabilities for node H1
add_probabilities!(diagram, "H1", [0.1, 0.9])

# Declare probability matrix for health nodes H_2, ... H_N-1, which have identical information sets and states
X_H = ProbabilityMatrix(diagram, "H2")
X_H["healthy", "pass", :] = [0.2, 0.8]
X_H["healthy", "treat", :] = [0.1, 0.9]
X_H["ill", "pass", :] = [0.9, 0.1]
X_H["ill", "treat", :] = [0.5, 0.5]

# Declare probability matrix for test result nodes T_1...T_N
X_T = ProbabilityMatrix(diagram, "T1")
X_T["ill", "positive"] = 0.8
X_T["ill", "negative"] = 0.2
X_T["healthy", "negative"] = 0.9
X_T["healthy", "positive"] = 0.1

for i in 1:No-1
    add_probabilities!(diagram, "T$i", X_T)
    add_probabilities!(diagram, "H$(i+1)", X_H)
end

for i in 1:No-1
    add_utilities!(diagram, "V$i", [-100.0, 0.0])
end

add_utilities!(diagram, "V$No", [300.0, 1000.0])
generate_diagram!(diagram)

column_generation_DP(diagram)
