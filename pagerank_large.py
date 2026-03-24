import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

# def load_google_web_graph(filepath):
#     """
#     Parses the web-Google 10k.txt edge list into a sparse column-stochastic matrix.
#     Assumes format: FromNodeId \t ToNodeId
#     """
#     edges = []
#     nodes = set()
    
#     print(f"Loading dataset from {filepath}...")
#     with open(filepath, 'r') as f:
#         for line in f:
#             if line.startswith('#'):
#                 continue
#             src, dst = map(int, line.strip().split())
#             edges.append((src, dst))
#             nodes.add(src)
#             nodes.add(dst)
            
#     # Map original node IDs to contiguous indices 0 ... n-1
#     node_list = sorted(list(nodes))
#     node_to_idx = {node: i for i, node in enumerate(node_list)}
#     n = len(node_list)
#     print(f"Found {n} unique nodes and {len(edges)} edges.")
    
#     # Build sparse matrix P (column-stochastic for non-dangling nodes)
#     row_idx = [node_to_idx[dst] for src, dst in edges]
#     col_idx = [node_to_idx[src] for src, dst in edges]
#     data = np.ones(len(edges))
    
#     # Adjacency matrix A (A[i, j] = 1 if j points to i)
#     A = sp.coo_matrix((data, (row_idx, col_idx)), shape=(n, n)).tocsc()
    
#     # Calculate out-degrees
#     out_degrees = np.array(A.sum(axis=0)).flatten()
    
#     # Avoid division by zero for dangling nodes (outdegree = 0)
#     out_degrees_safe = out_degrees.copy()
#     out_degrees_safe[out_degrees_safe == 0] = 1.0
    
#     # Normalize columns to create transition matrix P
#     data_normalized = A.data / out_degrees_safe[A.indices]
#     P = sp.csc_matrix((data_normalized, A.indices, A.indptr), shape=(n, n))
    
#     # Create a boolean mask for dangling nodes
#     dangling_mask = (out_degrees == 0)
    
#     return P, dangling_mask, node_list, n

def load_google_web_graph(filepath):
    """
    Parses the web-Google 10k.txt edge list into a sparse column-stochastic matrix.
    Assumes format: FromNodeId \t ToNodeId
    """
    edges = []
    nodes = set()
    
    print(f"Loading dataset from {filepath}...")
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            src, dst = map(int, line.strip().split())
            edges.append((src, dst))
            nodes.add(src)
            nodes.add(dst)
            
    # Map original node IDs to contiguous indices 0 ... n-1
    node_list = sorted(list(nodes))
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    n = len(node_list)
    
    # Build sparse matrix P (column-stochastic for non-dangling nodes)
    row_idx = [node_to_idx[dst] for src, dst in edges]
    col_idx = [node_to_idx[src] for src, dst in edges]
    data = np.ones(len(edges))
    
    # Use COO format for safe column-based normalization
    A_coo = sp.coo_matrix((data, (row_idx, col_idx)), shape=(n, n))
    
    # Calculate out-degrees (sum of columns)
    out_degrees = np.array(A_coo.sum(axis=0)).flatten()
    
    # Avoid division by zero for dangling nodes (outdegree = 0)
    out_degrees_safe = out_degrees.copy()
    out_degrees_safe[out_degrees_safe == 0] = 1.0
    
    # CORRECTED: Divide by the out-degree of the source node (col)
    data_normalized = A_coo.data / out_degrees_safe[A_coo.col]
    
    # Convert to CSC format for fast matrix math later
    P = sp.coo_matrix((data_normalized, (A_coo.row, A_coo.col)), shape=(n, n)).tocsc()
    
    # Create a boolean mask for dangling nodes
    dangling_mask = (out_degrees == 0)
    
    return P, dangling_mask, node_list, n

def sparse_pagerank_power(P, dangling_mask, n, p=0.15, tol=1e-10, max_iter=1000):
    """Power iteration optimized for sparse matrices."""
    r = np.ones(n) / n
    for iteration in range(max_iter):
        # 1. Standard link contribution
        link_contrib = P.dot(r)
        
        # 2. Re-distribute probability mass from dangling nodes + baseline teleportation
        dangling_sum = np.sum(r[dangling_mask])
        teleport_contrib = (p + (1 - p) * dangling_sum) / n
        
        r_new = (1 - p) * link_contrib + teleport_contrib
        
        if np.linalg.norm(r_new - r, 1) < tol:
            print(f"  Power Iteration converged in {iteration + 1} iterations")
            break
        r = r_new
    return r

def sparse_pagerank_closed_form(P, n, p=0.15):
    """
    Closed-form solution using scipy.sparse.linalg.
    Trick: Solve (I - (1-p)P) x = 1, then normalize x to sum to 1.
    """
    I = sp.eye(n, format='csc')
    A = I - (1 - p) * P
    b = np.ones(n)
    
    # Solve the sparse linear system
    x = splinalg.spsolve(A, b)
    
    # Normalize to get the proper probability distribution
    r = x / np.sum(x)
    return r

# ─────────────────────────────────────────────
# 9. Large Dataset Evaluation
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Note: Ensure you have the 'web-Google 10k.txt' file in your directory
    try:
        filepath = "web-Google_10k.txt"
        P, dangling_mask, node_list, n = load_google_web_graph(filepath)
        
        print("\nRunning Sparse Power Iteration...")
        t0 = time.time()
        pr_power = sparse_pagerank_power(P, dangling_mask, n)
        power_time = time.time() - t0
        print(f"  Time taken: {power_time:.4f} seconds")
        
        print("\nRunning Sparse Closed-Form Solver...")
        t1 = time.time()
        pr_closed = sparse_pagerank_closed_form(P, n)
        closed_time = time.time() - t1
        print(f"  Time taken: {closed_time:.4f} seconds")
        
        l1_error = np.linalg.norm(pr_power - pr_closed, 1)
        print(f"\nL1 Error between methods on 10k dataset: {l1_error:.4e}")
        
    except FileNotFoundError:
        print("\nSkipping large dataset evaluation: 'web-Google_10k.txt' not found.")
        print("Download it and run to complete the final assignment requirement.")