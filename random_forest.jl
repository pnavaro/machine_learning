# [ref](https://opensourc.es/blog/random-forest)

mutable struct Node

    feature    :: Int64 
    comp_value :: Float64
    data_idxs  :: Vector{Int64}
    mean       :: Float64
    left       :: Union{Nothing, Node}
    right      :: Union{Nothing, Node}

    Node() = new()

end

mutable struct Tree
    root   :: Node
    Tree() = new()
end

function create_random_forest(feature_matrix, train_ys, ntrees)

    forest             = Vector{Tree}()
    tc                 =  1
    total_nfeatures    = size(feature_matrix)[1]
    nfeatures_per_tree = 5
    ncols_per_tree     = length(train_ys)÷2

    while tc < ntrees

        feature_idx = sample(1:total_nfeatures, 
                             nfeatures_per_tree; 
                             replace=false)

        cols = sample(1:length(train_ys), ncols_per_tree; replace=false)

        tree = create_random_tree(feature_matrix, 
                                  feature_idx, 
                                  cols, 
                                  train_ys, 
                                  total_nfeatures)

        if tree.root.left_child != nothing
            push!(forest,tree)
            tc += 1
        else
            println("Couldn't create tree!!")
        end
        flush(stdout)
    end

    forest

end

function create_random_tree(glob_feature_matrix, ...)

    tree = Tree() 
    @views feature_matrix = glob_feature_matrix[feature_idx, cols]
    root, queue = create_root_node!(...)
    
    while length(queue) > 0
        node = popfirst!(queue)
        left_node, right_node = compute_node!(...)
        if left_node != nothing
            queue_compute_nodes!(...)
        end
    end
    tree.root = root
    tree

end
