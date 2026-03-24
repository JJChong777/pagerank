import numpy as np
import matplotlib.pyplot as plt

# 1. Core PageRank Engine

def build_google_matrix(adj: dict, p: float = 0.15) -> np.ndarray:
    """
    Build the Google (teleportation) matrix from an adjacency dict.
    adj: {node: [list of outgoing nodes]}
    Returns the nxn Google matrix G = (1-p)*M + p*(1/n)*ones
    """
    nodes = sorted(adj.keys())
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    
    # Build column-stochastic transition matrix M
    M = np.zeros((n, n))
    for src, dsts in adj.items():
        j = idx[src]
        if len(dsts) == 0:
            # Dangling node: distribute uniformly
            M[:, j] = 1.0 / n
        else:
            for dst in dsts:
                i = idx[dst]
                M[i, j] += 1.0 / len(dsts)
    
    # Google matrix: G = (1-p)*M + p*(1/n)*ones
    E = np.ones((n, n)) / n
    G = (1 - p) * M + p * E
    return G, M, nodes


def pagerank_power(adj: dict, p: float = 0.15, tol: float = 1e-10, max_iter: int = 1000):
    """Power iteration PageRank."""
    G, M, nodes = build_google_matrix(adj, p)
    n = len(nodes)
    r = np.ones(n) / n          # uniform initial distribution
    for iteration in range(max_iter):
        r_new = G @ r
        if np.linalg.norm(r_new - r, 1) < tol:
            print(f"  Converged after {iteration+1} iterations")
            break
        r = r_new
    return dict(zip(nodes, r_new)), G, M, nodes


def pagerank_closed_form(adj: dict, p: float = 0.15):
    """
    Closed-form PageRank via linear system:
    r = G*r  =>  (I - G)*r = 0  with sum(r)=1
    Equivalently solve: (I - (1-p)*M) * r = p/n * ones
    """
    G, M, nodes = build_google_matrix(adj, p)
    n = len(nodes)
    A = np.eye(n) - G
    # Replace last row with normalisation constraint
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    r = np.linalg.solve(A, b)
    return dict(zip(nodes, r)), G, M, nodes


# 3. Tutorial Graph (b): 4 nodes {1,2,3,4}
# From the diagram:
# 1->{2, 3, 4}, 2->{3, 4}, 3->{1}, 4->{1, 3}

graph_b = {
    1: [2, 3, 4],         
    2: [3, 4],      
    3: [1],   
    4: [1, 3]         
}

print("\n" + "=" * 60)
print("Graph (b): nodes {1, 2, 3, 4}")
print("=" * 60)
pr_b_power, G_b, M_b, nodes_b = pagerank_power(graph_b, p=0.15)
pr_b_closed, *_ = pagerank_closed_form(graph_b, p=0.15)

print("\nPower Iteration PageRank:")
for node, score in sorted(pr_b_power.items()):
    print(f"  PR({node}) = {score:.6f}")

print("\nClosed-Form PageRank:")
for node, score in sorted(pr_b_closed.items()):
    print(f"  PR({node}) = {score:.6f}")

print("\nTransition matrix M:")
print(np.round(M_b, 4))


# 4. Effect of teleportation probability p
print("\n" + "=" * 60)
print("Effect of teleportation probability p on PageRank (Graph b)")
print("=" * 60)

p_values = np.linspace(0.01, 0.99, 50)
pr_history = {node: [] for node in [1, 2, 3, 4]}

for p_val in p_values:
    pr, *_ = pagerank_closed_form(graph_b, p=p_val)
    for node in [1, 2, 3, 4]:
        pr_history[node].append(pr[node])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: p effect on graph b
ax = axes[0]
colors = ['#E63946', '#2196F3', '#4CAF50', '#FF9800']
for i, node in enumerate([1, 2, 3, 4]):
    ax.plot(p_values, pr_history[node], label=f'Node {node}', 
            color=colors[i], linewidth=2)
ax.axvline(x=0.15, color='gray', linestyle='--', alpha=0.7, label='p=0.15 (default)')
ax.set_xlabel('Teleportation Probability p', fontsize=12)
ax.set_ylabel('PageRank Score', fontsize=12)
ax.set_title('Effect of p on PageRank (Graph b)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)

# Right: bar chart at p=0.15
ax2 = axes[1]
graphs = ['Graph (b)']
all_scores = [
    [pr_b_closed[1], pr_b_closed[2], pr_b_closed[3], pr_b_closed[4]]
]
labels_b = ['1', '2', '3', '4']

x = np.arange(len(labels_b))
bars = ax2.bar(x - 0.2, all_scores[0], 0.35, label='Graph (b)', color='#2196F3', alpha=0.85)
ax2.set_xticks(x - 0.2)
ax2.set_xticklabels(labels_b)
ax2.set_xlabel('Node', fontsize=12)
ax2.set_ylabel('PageRank Score', fontsize=12)
ax2.set_title('PageRank Scores at p=0.15\n(Graph b)', fontsize=13, fontweight='bold')
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('pagerank_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: pagerank_analysis.png")



# 7. Convergence visualization
def pagerank_convergence(adj, p=0.15, max_iter=50):
    G, M, nodes = build_google_matrix(adj, p)
    n = len(nodes)
    r = np.ones(n) / n
    history = [r.copy()]
    errors = []
    pr_final, *_ = pagerank_closed_form(adj, p)
    r_true = np.array([pr_final[node] for node in nodes])
    
    for _ in range(max_iter):
        r_new = G @ r
        errors.append(np.linalg.norm(r_new - r_true, 1))
        history.append(r_new.copy())
        r = r_new
    return history, errors, nodes

history_b, errors_b, nodes_b_list = pagerank_convergence(graph_b)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for i, node in enumerate(nodes_b_list):
    scores = [h[i] for h in history_b]
    ax.plot(scores, label=f'Node {node}', color=colors[i], linewidth=2)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('PageRank Score', fontsize=12)
ax.set_title('Power Iteration Convergence (Graph b)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.semilogy(errors_b, color='#E63946', linewidth=2)
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('L1 Error (log scale)', fontsize=12)
ax2.set_title('Convergence Error vs. Closed-Form Solution', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pagerank_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: pagerank_convergence.png")


# 8. Print final summary
print("\n" + "=" * 60)
print("SUMMARY OF RESULTS")
print("=" * 60)

print("\nGraph (b) PageRank (p=0.15):")
for node in sorted(pr_b_closed):
    print(f"  PR({node}) = {pr_b_closed[node]:.6f}")

print("\nClosed-form formula:")
print("  r = (I - (1-p)*M)^{-1} * (p/n) * 1")
print("  where M is the column-stochastic transition matrix")
print("  and 1 is the all-ones vector of length n")
print("\nDone.")
