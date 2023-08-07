import inspect


class GraphKeys:
    """Class that holds the name of the data key."""

    Lattice = "lattice"  # (B, 3, 3) shape
    PBC = "pbc"  # (B, 3) shape
    Neighbors = "neighbors"  # number of neighbor index per each image of (B) shape

    Batch_idx = "batch"  # (N) shape
    Z = "z"  # (N) shape
    Pos = "pos"  # (N, 3) shape

    # Attributes marked with "index" are automatically incremented in batch processing
    Edge_idx = "edge_index"  # (2, E) shape, order is "source_to_target"
    Edge_shift = "edge_shift"  # (E, 3) shape
    Edge_dist = "edge_dist"  # (E) shape
    Edge_vec_st = "edge_vec"  # (E, 3) shape

    # 3body properties
    Idx_s_3b = "idx_s_3b"  # (n_triplets) shape
    Idx_t_3b = "idx_t_3b"  # (n_triplets) shape
    Idx_k_3b = "idx_k_3b"  # (n_triplets) shape
    Edge_idx_ks_3b = "edge_idx_ks_3b"  # (n_triplets) shape
    Edge_idx_st_3b = "edge_idx_st_3b"  # (n_triplets) shape
    Angles_3b = "angles_3b"  # (n_triplets) shape


KEYS = [
    a[1]
    for a in inspect.getmembers(GraphKeys, lambda a: not (inspect.isroutine(a)))
    if not (a[0].startswith("__") and a[0].endswith("__"))
]
