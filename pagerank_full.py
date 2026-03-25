import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt

def load_google_web_graph(filepath):
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
            
    node_list = sorted(list(nodes))
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    n = len(node_list)
    
    row_idx = [node_to_idx[dst] for src, dst in edges]
    col_idx = [node_to_idx[src] for src, dst in edges]
    data = np.ones(len(edges))
    
    A_coo = sp.coo_matrix((data, (row_idx, col_idx)), shape=(n, n))
    out_degrees = np.array(A_coo.sum(axis=0)).flatten()
    
    out_degrees_safe = out_degrees.copy()
    out_degrees_safe[out_degrees_safe == 0] = 1.0
    
    data_normalized = A_coo.data / out_degrees_safe[A_coo.col]
    P = sp.coo_matrix((data_normalized, (A_coo.row, A_coo.col)), shape=(n, n)).tocsc()
    dangling_mask = (out_degrees == 0)
    
    return P, dangling_mask, node_list, n

def sparse_pagerank_closed_form(P, n, p=0.15):
    """Run closed-form first so we have the absolute truth to compare against."""
    I = sp.eye(n, format='csc')
    A = I - (1 - p) * P
    b = np.ones(n)
    x = splinalg.spsolve(A, b)
    r = x / np.sum(x)
    return r

def sparse_pagerank_power_tracked(P, dangling_mask, n, true_pr, p=0.15, tol=1e-10, max_iter=1000):
    """Power iteration optimized for sparse matrices, now tracking L1 error."""
    r = np.ones(n) / n
    errors = []
    
    for iteration in range(max_iter):
        link_contrib = P.dot(r)
        dangling_sum = np.sum(r[dangling_mask])
        teleport_contrib = (p + (1 - p) * dangling_sum) / n
        r_new = (1 - p) * link_contrib + teleport_contrib
        
        # Track the error against the true closed-form solution
        current_error = np.linalg.norm(r_new - true_pr, 1)
        errors.append(current_error)
        
        if np.linalg.norm(r_new - r, 1) < tol:
            print(f"  Power Iteration converged in {iteration + 1} iterations")
            break
        r = r_new
    return r, errors

def generate_large_graph_visuals(pr_scores, errors, node_list):
    """Generates the three graphs for the full dataset."""
    print("\nGenerating visual graphs...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Convergence Error Plot (Log Scale)
    ax1 = axes[0]
    ax1.semilogy(errors, color='#E63946', linewidth=2)
    ax1.set_title('L1 Error Convergence (All Nodes)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('L1 Error vs. Closed-Form', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. Top 10 Nodes Bar Chart
    ax2 = axes[1]
    # Sort nodes by score
    sorted_indices = np.argsort(pr_scores)[::-1]
    top_10_idx = sorted_indices[:10]
    top_10_scores = pr_scores[top_10_idx]
    top_10_labels = [str(node_list[i]) for i in top_10_idx]
    
    bars = ax2.bar(top_10_labels, top_10_scores, color='#2196F3', alpha=0.85)
    ax2.set_title('Top 10 Highest PageRank Nodes', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Node ID', fontsize=12)
    ax2.set_ylabel('PageRank Score', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Power Law Distribution
    ax3 = axes[2]
    ax3.hist(pr_scores, bins=50, color='#4CAF50', alpha=0.85, log=True)
    ax3.set_title('PageRank Score Distribution (Log Scale)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('PageRank Score', fontsize=12)
    ax3.set_ylabel('Number of Nodes (Log)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pagerank_full_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved graphs as: pagerank_full_analysis.png")

# ─────────────────────────────────────────────
# 9. Large Dataset Evaluation
# ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        filepath = "web-Google.txt"
        P, dangling_mask, node_list, n = load_google_web_graph(filepath)
        
        print("\nRunning Sparse Closed-Form Solver (Baseline)...")
        t1 = time.time()
        pr_closed = sparse_pagerank_closed_form(P, n)
        closed_time = time.time() - t1
        print(f"  Time taken: {closed_time:.4f} seconds")

        print("\nRunning Sparse Power Iteration (Tracking Error)...")
        t0 = time.time()
        # We pass the closed-form truth in so we can track the error at each step
        pr_power, convergence_errors = sparse_pagerank_power_tracked(P, dangling_mask, n, pr_closed)
        power_time = time.time() - t0
        print(f"  Time taken: {power_time:.4f} seconds")
        
        l1_error = np.linalg.norm(pr_power - pr_closed, 1)
        print(f"\nFinal L1 Error between methods on full dataset: {l1_error:.4e}")
        
        # Generate the graphs
        generate_large_graph_visuals(pr_power, convergence_errors, node_list)
        
    except FileNotFoundError:
        print(f"\nDataset '{filepath}' not found. Please ensure it is in the same directory.")