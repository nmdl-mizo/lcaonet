class GraphKeys:
    """Class that holds the name of the data key."""

    Lattice = "lattice"  # (B, 3, 3) shape
    PBC = "pbc"  # (B, 3) shape

    Batch_idx = "batch"  # (N) shape
    Z = "z"  # (N) shape
    Pos = "pos"  # (N, 3) shape

    # Attributes marked with "index" are automatically incremented in batch processing
    Edge_idx = "edge_index"  # (2, E) shape
    Edge_shift = "edge_shift"  # (E, 3) shape
