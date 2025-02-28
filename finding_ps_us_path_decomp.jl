using DecisionProgramming
using JuMP, Gurobi, DataStructures, Statistics
using Base.Iterators: product
include("new_inequality_labeling.jl")

No = 4
resps = zeros(2,20)

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

dt1 = @elapsed begin

s = (1,1,1,1,1,1)

# max_paths1 = Dict{Tuple{Int64,Int64},Int64}()
# max_ps1 = Dict{Tuple{Int64,Int64},Float64}()
# for s1 in 1:2, s6 in 1:2
#     max_p = 0
#     max_path = 0
#     for s2 in 1:2
#         p = X_L[s1]*X_Rs[1][s1,s2]
#         if p >= max_p
#             max_p = p
#             max_path = s2
#         end
#         max_paths1[(s1,s6)] = max_path
#         max_ps1[(s1,s6)] = max_p
#     end
# end

# max_paths2 = Dict{Tuple{Int64,Int64,Int64},Int64}()
# max_ps2 = Dict{Tuple{Int64,Int64,Int64},Float64}()
# for s1 in 1:2, s7 in 1:2, s6 in 1:2
#     max_p = 0
#     max_path = 0
#     for s3 in 1:2
#         p = max_ps1[(s1,s6)]*X_Rs[2][s1,s3]
#         if p >= max_p
#             max_p = p
#             max_path = s3
#         end
#         max_paths2[(s1,s6,s7)] = max_path
#         max_ps2[(s1,s6,s7)] = max_p
#     end
# end

# max_paths3 = Dict{Tuple{Int64,Int64,Int64,Int64},Int64}()
# max_ps3 = Dict{Tuple{Int64,Int64,Int64,Int64},Float64}()
# for s1 in 1:2, s8 in 1:2, s7 in 1:2, s6 in 1:2
#     max_p = 0
#     max_path = 0
#     for s4 in 1:2
#         p = max_ps2[(s1,s6,s7)]*X_Rs[3][s1,s4]
#         if p >= max_p
#             max_p = p
#             max_path = s4
#         end
#         max_paths3[(s1,s6,s7,s8)] = max_path
#         max_ps3[(s1,s6,s7,s8)] = max_p
#     end
# end

# max_paths4 = Dict{Tuple{Int64,Int64,Int64,Int64,Int64},Int64}()
# max_ps4 = Dict{Tuple{Int64,Int64,Int64,Int64,Int64},Float64}()
# for s1 in 1:2, s9 in 1:2, s8 in 1:2, s7 in 1:2, s6 in 1:2
#     max_p = 0
#     max_path = 0
#     for s5 in 1:2
#         p = max_ps3[(s1,s6,s7,s8)]*X_Rs[4][s1,s5]
#         if p >= max_p
#             max_p = p
#             max_path = s5
#         end
#         max_paths4[(s1,s6,s7,s8,s9)] = max_path
#         max_ps4[(s1,s6,s7,s8,s9)] = max_p
#     end
# end

# max_paths5 = Dict{Tuple{Int64,Int64,Int64,Int64,Int64,Int64},Int64}()
# max_ps5 = Dict{Tuple{Int64,Int64,Int64,Int64,Int64,Int64},Float64}()
# max_p = [0.0]
# max_pa = [(0,0,0,0,0,0)]
# for s1 in 1:2, s10 in 1:2, s9 in 1:2, s8 in 1:2, s7 in 1:2, s6 in 1:2
#     pu = max_ps4[(s1,s6,s7,s8,s9)]*X_F[s1,s6,s7,s8,s9,s10]*Y_T[s10,s6,s7,s8,s9]
#     if pu >= max_p[1]
#         max_p[1] = pu
#         max_pa[1] = (s1,s6,s7,s8,s9,s10)
#     end
# end
# max_path = max_pa[1]

# path = [max_path[1],max_paths1[(max_path[1],max_path[2])],max_paths2[(max_path[1],max_path[2],max_path[3])],max_paths3[(max_path[1],max_path[2],max_path[3],max_path[4])],max_paths4[(max_path[1],max_path[2],max_path[3],max_path[4],max_path[5])],max_path[2:end]...]

max_paths1 = Dict{Tuple{Int64,Int64},Int64}()
max_ps1 = Dict{Tuple{Int64,Int64},Float64}()
for s6 in 1:2
    max_p = 0
    max_path = 0
    p = X_L[s[1]]*X_Rs[1][s[1],s[2]]
    if p >= max_p
        max_p = p
        max_path = s[2]
    end
    max_paths1[(s[1],s6)] = max_path
    max_ps1[(s[1],s6)] = max_p
end

max_paths2 = Dict{Tuple{Int64,Int64,Int64},Int64}()
max_ps2 = Dict{Tuple{Int64,Int64,Int64},Float64}()
for  s7 in 1:2, s6 in 1:2
    max_p = 0
    max_path = 0
    p = max_ps1[(s[1],s6)]*X_Rs[2][s[1],s[3]]
    if p >= max_p
        max_p = p
        max_path = s[3]
    end
    max_paths2[(s[1],s6,s7)] = max_path
    max_ps2[(s[1],s6,s7)] = max_p
end

max_paths3 = Dict{Tuple{Int64,Int64,Int64,Int64},Int64}()
max_ps3 = Dict{Tuple{Int64,Int64,Int64,Int64},Float64}()
for  s8 in 1:2, s7 in 1:2, s6 in 1:2
    max_p = 0
    max_path = 0
    p = max_ps2[(s[1],s6,s7)]*X_Rs[3][s[1],s[4]]
    if p >= max_p
        max_p = p
        max_path = s[4]
    end
    max_paths3[(s[1],s6,s7,s8)] = max_path
    max_ps3[(s[1],s6,s7,s8)] = max_p
end

max_paths4 = Dict{Tuple{Int64,Int64,Int64,Int64,Int64},Int64}()
max_ps4 = Dict{Tuple{Int64,Int64,Int64,Int64,Int64},Float64}()
for  s9 in 1:2, s8 in 1:2, s7 in 1:2, s6 in 1:2
    max_p = 0
    max_path = 0
    p = max_ps3[(s[1],s6,s7,s8)]*X_Rs[4][s[1],s[5]]
    if p >= max_p
        max_p = p
        max_path = s[5]
    end
    max_paths4[(s[1],s6,s7,s8,s9)] = max_path
    max_ps4[(s[1],s6,s7,s8,s9)] = max_p
end

max_paths5 = Dict{Tuple{Int64,Int64,Int64,Int64,Int64,Int64},Int64}()
max_ps5 = Dict{Tuple{Int64,Int64,Int64,Int64,Int64,Int64},Float64}()
max_p = [0.0]
max_pa = [(0,0,0,0,0,0)]
for  s9 in 1:2, s8 in 1:2, s7 in 1:2, s6 in 1:2
    pu = max_ps4[(s[1],s6,s7,s8,s9)]*X_F[s[1],s6,s7,s8,s9,s[6]]*Y_T[s[6],s6,s7,s8,s9]
    if pu >= max_p[1]
        max_p[1] = pu
        max_pa[1] = (s[1],s6,s7,s8,s9,s[6])
    end
end
max_path = max_pa[1]

path = [max_path[1],max_paths1[(max_path[1],max_path[2])],max_paths2[(max_path[1],max_path[2],max_path[3])],max_paths3[(max_path[1],max_path[2],max_path[3],max_path[4])],max_paths4[(max_path[1],max_path[2],max_path[3],max_path[4],max_path[5])],max_path[2:end]...]


end
dims1 = States(get_values(diagram.S))
dt2 = @elapsed begin
    bloood = -1.2
    bloood_p = []
    for s in paths(dims1,FixedPath((1 => s[1], 2 => s[2], 4 => s[3], 6 => s[4], 8 => s[5], 10 => s[6])))
        global bloood
        global bloood_p
        pu = diagram.P(s)*diagram.U(s)
        if pu >= bloood
            bloood = pu
            bloood_p = [s]
        end
    end

end

println("Final path: ", path)
println("Cost: ", max_p[1])
println("Final path: ", bloood_p[1])
println("Cost: ", bloood)

println(dt1)
println(dt2)