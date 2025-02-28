using JuMP, Gurobi, Statistics, Pkg
using Base.Iterators: product
using Distributed, Hwloc
Distributed.@everywhere using DecisionProgramming, DataStructures
Distributed.@everywhere include("parallel_stuff_branching.jl")

mutable struct LabelWithUpperBound
    pi::Float64
    theta::Float64
    sigma::Float64
    rc::Float64
    visited::Array{Int64}
end


Base.:(==)(x::Cut, y::Cut) = x.d == y.d && x.s == y.s && x.v == y.v

struct Branch
    cuts::Array{Cut}
    pathss::Vector{Path} where NN
    ordered_lists::Dict{Path, Vector{Path}}
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

function pathsa(states::AbstractVector{State})
    product(UnitRange.(one(eltype(states)), states)...)
end

function pathsa(states::AbstractVector{State}, fixed::FixedPath)
    iters = collect(UnitRange.(one(eltype(states)), states))
    for (i, v) in fixed
        iters[i] = UnitRange(v, v)
    end
    product(iters...)
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


 function column_generation_DP_ordered_lists(diagram::InfluenceDiagram)
    iterator = 0
    I_j_indices_result = I_j_indices(diagram, diagram.Nodes)
    z_indices = indices(diagram.D)
    decision_set = sort(unique([I_j_indices_result[z_indices]...; z_indices...]))
    # TODO: Run heuristics before entering the labeling algorithm
    pathss = Path[]
    times_searching_for_paths = 0
    dt = @elapsed begin
        dt8 = @elapsed begin
            (pathss,ordered_lists) = init_paths_ordered_list(diagram,pathss,false,0.0)
        end
        dt5 = @elapsed begin
        branches = Branch[]
        (mu,lambda,obj,sol,z_z) = solve_model_with_paths_inequality(diagram,pathss,0.0,false)
        end
        for (d, z_d) in zip(z_indices, z_z)
            println(value.(z_d))
            if !all(value.(z_d) .== 1 .|| value.(z_d) .== 0)
                indices = findall(value.(z_d) .!= 1 .&& value.(z_d) .!= 0)
                push!(branches, Branch([Cut(d,collect(Tuple(indices[1])),1)],pathss,ordered_lists))
                push!(branches, Branch([Cut(d,collect(Tuple(indices[1])),0)],pathss,ordered_lists))
                break
            end
        end
        global_optimum = 0
        if !isempty(branches)
            max_branch = branches[1]
        else
            println(typeof(ordered_lists))
            max_branch = Branch(Cut[],pathss,ordered_lists)
        end
        max_dec_vars = z_z
        dt3 = @elapsed begin
            while 1==1
                if isempty(branches)
                    global_optimum = maximum([global_optimum,obj])
                    break
                end
                new_branches = []
                pathsss = Path[]
                for branch in branches
                    dt10 = @elapsed begin
                        dt6 = @elapsed begin
                            (iter,pathsss) = update_paths_ordered_lists_dist(diagram,branch)
                            #(iter,pathsss) = update_paths_ordered_lists(diagram,branch)
                        end
                        times_searching_for_paths = times_searching_for_paths + dt6
                        iterator = iterator + 1
                        (mu,lambda,obj,sol,z_z) = solve_model_with_paths_inequality_with_branching(diagram,pathsss,0.0,false,branch)
                        if obj <= global_optimum
                            continue
                        end
                        z_indices = indices(diagram.D)
                        all_integer = true
                        for (d, z_d) in zip(z_indices, z_z)
                            if !all(value.(z_d) .== 1 .|| value.(z_d) .== 0)
                                indices = findall(value.(z_d) .!= 1 .&& value.(z_d) .!= 0)
                                push!(new_branches, Branch([branch.cuts..., Cut(d,collect(Tuple(indices[1])),1)], pathsss, branch.ordered_lists))
                                push!(new_branches,Branch([branch.cuts..., Cut(d,collect(Tuple(indices[1])),0)], pathsss, branch.ordered_lists))
                                all_integer = false
                                break
                            end
                        end
                        if all_integer
                            if obj >= global_optimum
                                max_branch = Branch(branch.cuts, pathsss, branch.ordered_lists)
                                max_dec_vars = z_z
                            end
                            global_optimum = maximum([global_optimum,obj])
                        end
                    end
                    println("Time taken for this branch: ",dt10)
                end
                branches = new_branches
            end
        end
    end
    for (d, z_d) in zip(z_indices, z_z)
        println(d)
        println(value.(z_d))
    end
    println("Processing first paths ordered_lists: ", dt8)
    println("Rest: ", dt5)
    println("Sol_time: ",dt)
    println("Total time updating paths: ",times_searching_for_paths)
    println("Branching_time: ",dt3)
    println("Model_solve_time: ",sol)
    println("Objective: ",obj)
    println("Global optimum: ",global_optimum)
    println("Branches covered: ",iterator)
    return (dt,obj,max_branch)
 end

 


 function init_paths_ordered_list(diagram::InfluenceDiagram,path_list::Vector{Path},prints::Bool,scaling_factor::Float64) where NN
    I_j_indices_result = I_j_indices(diagram, diagram.Nodes)
    z_indices = indices(diagram.D)
    decision_set = sort(unique([I_j_indices_result[z_indices]...; z_indices...]))
    index_o = Dict(zip(decision_set,1:length(decision_set)))
    dims1 = States(get_values(diagram.S))[decision_set]
    c_indices = filter(c -> c in decision_set, indices(diagram.C))
    c_indices_ds = [index_o[c] for c in c_indices]
    dims = States(get_values(diagram.S))[[c_indices;]]
    queues = Dict{Path,Vector{Path}}()
    dt2 = @elapsed begin
        for s_c in paths(dims)
            blah = Dict{Path,Float64}()
            for s in paths(dims1, FixedPath(zip(c_indices_ds,[s_c...])))
                blah[s] = sum(diagram.P(s2)*(diagram.U(s2) + scaling_factor) for s2 in paths(States(get_values(diagram.S)), FixedPath(zip(decision_set,[s...]))))
            end
            queues[s_c] = map(y -> y[1],sort(collect(blah), by=x->-x[2]))
            push!(path_list,queues[s_c][1])
        end
    end
    return(path_list, queues)
 end


function solve_model_with_paths_inequality(diagram::InfluenceDiagram, pathss::Vector{Path},scaling_factor::Float64,use_stricter_tolerance::Bool)
    z_indices = indices(diagram.D)
    I_j_indices_result = I_j_indices(diagram, diagram.Nodes)
    decision_set = sort(unique([I_j_indices_result[z_indices]...; z_indices...]))
    index_o = Dict(zip(decision_set,1:length(decision_set)))
    dims1 = States(get_values(diagram.S))[decision_set]
    N = length(decision_set)
    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(Gurobi.Env())))


    z = DecisionVariablesCont(model, diagram)
    x_s = Dict{Path{N}, VariableRef}(
        s => @variable(model, base_name="$(s)", lower_bound = 0, upper_bound = 1)
        for s in pathss)

    z_z = [decision_node.z for decision_node in get_values(z)]
    mu = Dict{Tuple{Int16,Vector{Int16}}, ConstraintRef}()
    for (d, z_d) in zip(z_indices, z_z)
        feasible_paths = prod(get_values(diagram.S)[key] for key in filter(i -> !(i in [I_j_indices_result[d]; d]), 1:length(diagram.C)+length(diagram.D)))
        dims = States(get_values(diagram.S))[[I_j_indices_result[d]; d]]
        other_decisions = filter(j -> all(j != d_set for d_set in [I_j_indices_result[d]; d]), z_indices)
        theoretical_ub = prod(get_values(diagram.S)[decision_set])/prod(dims)/prod(get_values(diagram.S)[other_decisions])
        existing_paths = keys(x_s)
        d_ds = index_o[d]
        Id_ds = [index_o[i] for i in I_j_indices_result[d]]
        for s_Id_d in paths(dims)
            mu[(d,collect(s_Id_d))] = @constraint(model,  sum(x_s[s] for s in filter(s -> s[[Id_ds; d_ds]] == s_Id_d, existing_paths)) ≤ z_d[s_Id_d...] * minimum([feasible_paths,theoretical_ub]))
        end
    end
    c_indices = filter(c -> c in decision_set, indices(diagram.C))
    dims = States(get_values(diagram.S))[[c_indices;]]
    c_indices_ds = [index_o[c] for c in c_indices]
    existing_paths = keys(x_s)
    lambda = Dict{Vector{Int16}, ConstraintRef}()
    for s_c in paths(dims)
        lambda[[s_c...]] = @constraint(model, sum(x_s[s] for s in filter(s -> s[[c_indices_ds;]] == s_c, existing_paths)) <= 1)
    end
    @objective(model, Max, sum( sum(diagram.P(s2)*(diagram.U(s2)) for s2 in paths(States(get_values(diagram.S)), FixedPath(zip(decision_set,[s...]))))*x_s[s] for s in keys(x_s)))
    optimize!(model)
    #pathsss = filter(s -> value(x_s[s]) >= 1e-8, keys(x_s))
    return (mu,lambda, objective_value(model),solve_time(model),z_z)
end

function solve_model_with_paths_inequality_MIP(diagram::InfluenceDiagram,scaling_factor::Float64)
    z_indices = indices(diagram.D)
    I_j_indices_result = I_j_indices(diagram, diagram.Nodes)
    decision_set = sort(unique([I_j_indices_result[z_indices]...; z_indices...]))
    index_o = Dict(zip(decision_set,1:length(decision_set)))
    dims1 = States(get_values(diagram.S))[decision_set]
    N = length(decision_set)
    optimizer = optimizer_with_attributes(Gurobi.Optimizer,"OptimalityTol" => 1e-09, "MIPGap" => 0)
    model = Model(optimizer)


    z = DecisionVariables(model, diagram)
    x_s = Dict{Path{N}, VariableRef}(
        s => @variable(model, base_name="$(s)", lower_bound = 0, upper_bound = 1)
        for s in paths(dims1))

    z_z = [decision_node.z for decision_node in get_values(z)]
    mu = Dict{Tuple{Int16,Vector{Int16}}, ConstraintRef}()
    for (d, z_d) in zip(z_indices, z_z)
        feasible_paths = prod(get_values(diagram.S)[key] for key in filter(i -> !(i in [I_j_indices_result[d]; d]), 1:length(diagram.C)+length(diagram.D)))
        dims = States(get_values(diagram.S))[[I_j_indices_result[d]; d]]
        other_decisions = filter(j -> all(j != d_set for d_set in [I_j_indices_result[d]; d]), z_indices)
        theoretical_ub = prod(get_values(diagram.S)[decision_set])/prod(dims)/prod(get_values(diagram.S)[other_decisions])
        existing_paths = keys(x_s)
        d_ds = index_o[d]
        Id_ds = [index_o[i] for i in I_j_indices_result[d]]
        for s_Id_d in paths(dims)
            mu[(d,collect(s_Id_d))] = @constraint(model,  sum(x_s[s] for s in filter(s -> s[[Id_ds; d_ds]] == s_Id_d, existing_paths)) ≤ z_d[s_Id_d...] * minimum([feasible_paths,theoretical_ub]))
        end
    end
    c_indices = filter(c -> c in decision_set, indices(diagram.C))
    dims = States(get_values(diagram.S))[[c_indices;]]
    c_indices_ds = [index_o[c] for c in c_indices]
    existing_paths = keys(x_s)
    lambda = Dict{Vector{Int16}, ConstraintRef}()
    for s_c in paths(dims)
        lambda[[s_c...]] = @constraint(model, sum(x_s[s] for s in filter(s -> s[[c_indices_ds;]] == s_c, existing_paths)) <= 1)
    end
    @objective(model, Max, sum( sum(diagram.P(s2)*(diagram.U(s2)) for s2 in paths(States(get_values(diagram.S)), FixedPath(zip(decision_set,[s...]))))*x_s[s] for s in keys(x_s)))
    optimize!(model)
    return (mu,lambda, objective_value(model),solve_time(model),z_z)
end

function solve_model_with_paths_inequality_with_branching(diagram::InfluenceDiagram, pathss::Vector{NTuple{NN, Int16}},scaling_factor::Float64,use_stricter_tolerance::Bool,branch::Branch) where NN
    z_indices = indices(diagram.D)
    I_j_indices_result = I_j_indices(diagram, diagram.Nodes)
    decision_set = sort(unique([I_j_indices_result[z_indices]...; z_indices...]))
    index_o = Dict(zip(decision_set,1:length(decision_set)))
    dims1 = States(get_values(diagram.S))[decision_set]
    N = length(decision_set)
    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(Gurobi.Env())))


    z = DecisionVariablesCont(model, diagram)
    x_s = Dict{Path{N}, VariableRef}(
        s => @variable(model, base_name="$(s)", lower_bound = 0, upper_bound = 1)
        for s in pathss)

    z_z = [decision_node.z for decision_node in get_values(z)]
    for b in branch.cuts
        @constraint(model, z_z[findfirst(x -> x == b.d, z_indices)][b.s...] == b.v)
    end
    mu = Dict{Tuple{Int16,Vector{Int16}}, ConstraintRef}()
    for (d, z_d) in zip(z_indices, z_z)
        feasible_paths = prod(get_values(diagram.S)[key] for key in filter(i -> !(i in [I_j_indices_result[d]; d]), 1:length(diagram.C)+length(diagram.D)))
        dims = States(get_values(diagram.S))[[I_j_indices_result[d]; d]]
        other_decisions = filter(j -> all(j != d_set for d_set in [I_j_indices_result[d]; d]), z_indices)
        theoretical_ub = prod(get_values(diagram.S)[decision_set])/prod(dims)/prod(get_values(diagram.S)[other_decisions])
        existing_paths = keys(x_s)
        d_ds = index_o[d]
        Id_ds = [index_o[i] for i in I_j_indices_result[d]]
        for s_Id_d in paths(dims)
            mu[(d,collect(s_Id_d))] = @constraint(model,  sum(x_s[s] for s in filter(s -> s[[Id_ds; d_ds]] == s_Id_d, existing_paths)) ≤ z_d[s_Id_d...] * minimum([feasible_paths,theoretical_ub]))
        end
    end
    c_indices = filter(c -> c in decision_set, indices(diagram.C))
    dims = States(get_values(diagram.S))[[c_indices;]]
    c_indices_ds = [index_o[c] for c in c_indices]
    existing_paths = keys(x_s)
    lambda = Dict{Vector{Int16}, ConstraintRef}()
    for s_c in paths(dims)
        lambda[[s_c...]] = @constraint(model, sum(x_s[s] for s in filter(s -> s[[c_indices_ds;]] == s_c, existing_paths)) <= 1)
    end
    @objective(model, Max, sum( sum(diagram.P(s2)*(diagram.U(s2)) for s2 in paths(States(get_values(diagram.S)), FixedPath(zip(decision_set,[s...]))))*x_s[s] for s in keys(x_s)))
    optimize!(model)
    return (mu,lambda, objective_value(model),solve_time(model),z_z)
end


function update_paths_ordered_lists(diagram::InfluenceDiagram, branch::Branch )
    iter = 0
    I_j_indices_result = I_j_indices(diagram, diagram.Nodes)
    z_indices = indices(diagram.D)
    all_c_indices = indices(diagram.C)
    decision_set = sort(unique([I_j_indices_result[z_indices]...; z_indices...]))
    c_indices_in_decision_set = filter(c -> c in decision_set, all_c_indices)
    index_o = Dict(zip(decision_set,1:length(decision_set)))
    dims1 = States(get_values(diagram.S))[decision_set]
    c_indices = filter(c -> c in decision_set, indices(diagram.C))
    c_indices_ds = [index_o[c] for c in c_indices]
    c_index = Dict(zip([c for (c,n) in collect(diagram.Nodes)[c_indices_in_decision_set]],1:length(c_indices_in_decision_set)))
    dims = States(get_values(diagram.S))[[c_indices;]]
    new_pathss = Path[]
    for s in branch.pathss
        if fulfills_cuts(diagram,s,branch.cuts,index_o)
            push!(new_pathss,s)
        else
            iter = iter + 1
            s_c = s[c_indices_ds]
            for new_path in branch.ordered_lists[s_c]
                if fulfills_cuts(diagram,new_path,branch.cuts,index_o)
                    push!(new_pathss,new_path)
                    break
                end
            end
        end
    end
    return (iter,new_pathss)
end

function update_paths_ordered_lists_dist(diagram::InfluenceDiagram, branch::Branch )
    iter = 0
    I_j_indices_result = I_j_indices(diagram, diagram.Nodes)
    z_indices = indices(diagram.D)
    all_c_indices = indices(diagram.C)
    decision_set = sort(unique([I_j_indices_result[z_indices]...; z_indices...]))
    c_indices_in_decision_set = filter(c -> c in decision_set, all_c_indices)
    index_o = Dict(zip(decision_set,1:length(decision_set)))
    c_indices = filter(c -> c in decision_set, indices(diagram.C))
    c_indices_ds = [index_o[c] for c in c_indices]
    c_index = Dict(zip([c for (c,n) in collect(diagram.Nodes)[c_indices_in_decision_set]],1:length(c_indices_in_decision_set)))
    dims = States(get_values(diagram.S))[[c_indices;]]
    new_paths = Distributed.pmap(update_paths_dist_improved, branch.ordered_lists, [branch.cuts for s in 1:length(branch.ordered_lists)], [diagram for s in 1:length(branch.ordered_lists)],[index_o for s in 1:length(branch.ordered_lists)])
    return (0,new_paths)
end

function fulfills_cuts(diagram::InfluenceDiagram,path::Path,cuts::Array{Cut},index_o::Dict{Int16,Int64})
    I_j_indices_result = I_j_indices(diagram, diagram.Nodes)
    for cut in cuts
        if cut.v == 0
            if all(path[index_o[I_j_indices_result[cut.d]...]] .== cut.s[1:end-1]) && cut.s[end] == path[index_o[cut.d]]
                return false
            end
        elseif cut.v == 1
            if all(path[index_o[I_j_indices_result[cut.d]...]] .== cut.s[1:end-1]) && cut.s[end] != path[index_o[cut.d]]
                return false
            end
        end
    end
    return true
end

function calculate_maximum_paths(diagram::InfluenceDiagram,cuts::Array{Cut})
    I_j_indices_result = I_j_indices(diagram, diagram.Nodes)
    z_indices = indices(diagram.D)
    decision_set = sort(unique([I_j_indices_result[z_indices]...; z_indices...]))
    index_o = Dict(zip(decision_set,1:length(decision_set)))
    dims1 = States(get_values(diagram.S))[decision_set]
    c_indices = filter(c -> c in decision_set, indices(diagram.C))
    c_indices_ds = [index_o[c] for c in c_indices]
    dims = States(get_values(diagram.S))[[c_indices;]]
    path_list = Path[]
    dt2 = @elapsed begin
        for s_c in paths(dims)
            blah = Dict{Path,Float64}()
            for s in paths(dims1, FixedPath(zip(c_indices_ds,[s_c...])))
                if fulfills_cuts(diagram,s,cuts,index_o)
                    blah[s] = sum(diagram.P(s2)*(diagram.U(s2)) for s2 in paths(States(get_values(diagram.S)), FixedPath(zip(decision_set,[s...]))))
                end
            end
            map(y -> y[1],sort(collect(blah), by=x->-x[2]))
            push!(path_list,map(y -> y[1],sort(collect(blah), by=x->-x[2]))[1])
        end
    end
    return path_list
end