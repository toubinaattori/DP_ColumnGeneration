struct Cut
    d::Int16
    s::Array{Int64}
    v::Int64
end

function init_paths_dist(s_c::Path,diagram::InfluenceDiagram,scaling_factor::Float64,c_indices_ds::Vector{Int64},decision_set::Vector{Int16})
    dt = @elapsed begin
        dims1 = States(get_values(diagram.S))[decision_set]
        NN = length(decision_set)
        blah = Dict{Path,Float64}()
        for s in paths(dims1, FixedPath(zip(c_indices_ds,[s_c...])))
            blah[s] = sum(diagram.P(s2)*(diagram.U(s2) + scaling_factor) for s2 in paths(States(get_values(diagram.S)), FixedPath(zip(decision_set,[s...]))))
        end
        queue = PriorityQueue(Base.Order.Reverse, blah)
        path = dequeue_pair!(queue)[1]
    end
    println("Time taken in worker: ", dt)
    return(path,s_c,queue)
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

function update_paths_dist(s::Path,ordered_list::Vector{Path},cuts::Array{Cut},diagram::InfluenceDiagram,index_o::Dict{Int16,Int64} )
    if fulfills_cuts(diagram,s,cuts,index_o)
        return s
    else
        for new_path in ordered_list
            if fulfills_cuts(diagram,new_path,cuts,index_o)
                return new_path
            end
        end
    end
end

function update_paths_dist_improved(ordered_list::Pair{Path,Vector{Path}},cuts::Array{Cut},diagram::InfluenceDiagram,index_o::Dict{Int16,Int64} )
    for new_path in ordered_list[2]
        if fulfills_cuts(diagram,new_path,cuts,index_o)
            return new_path
        end
    end
end