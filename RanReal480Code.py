import random
import math
import time
import statistics
import matplotlib.pyplot as plt

# Function to parse the instance file
def parse_instance(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline().strip().split()
        M = int(first_line[0])  # Number of nodes
        C = int(first_line[1])  # Number of clusters

        # Cluster capacity limits
        limits = []
        index = 2
        for i in range(C):
            lower_limit = int(first_line[index])
            upper_limit = int(first_line[index + 1])
            limits.append((lower_limit, upper_limit))
            index += 2

        # Node weights
        node_weights = list(map(int, first_line[index + 1:]))

        # Edges and edge weights
        edges = []
        for line in file:
            elementA, elementB, edge_weight = line.strip().split()
            elementA = int(elementA)
            elementB = int(elementB)
            edge_weight = float(edge_weight)
            edges.append((elementA, elementB, edge_weight))

    return M, C, limits, node_weights, edges

# Greedy heuristic for initial clustering
def greedy_heuristic_instance(M, C, limits, node_weights, edges):
    clusters = {i: [] for i in range(C)}
    cluster_weights = {i: 0 for i in range(C)}

    sorted_nodes = sorted(enumerate(node_weights), key=lambda x: x[1], reverse=True)

    for node, weight in sorted_nodes:
        best_cluster = None
        best_improvement = -float('inf')

        for cluster in range(C):
            lower_limit, upper_limit = limits[cluster]
            if cluster_weights[cluster] + weight <= upper_limit:
                # Calculate improvement in score by adding the node to this cluster
                improvement = 0
                for (elementA, elementB, edge_weight) in edges:
                    if (elementA == node and elementB in clusters[cluster]) or (elementB == node and elementA in clusters[cluster]):
                        improvement += edge_weight
                
                # Choose the best cluster based on improvement
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_cluster = cluster

        if best_cluster is not None:
            clusters[best_cluster].append(node)
            cluster_weights[best_cluster] += weight

    # Ensure all clusters have weights within the limits
    for cluster in range(C):
        while cluster_weights[cluster] < limits[cluster][0]:
            for donor_cluster in range(C):
                if donor_cluster != cluster and cluster_weights[donor_cluster] > limits[donor_cluster][0]:
                    node_to_move = clusters[donor_cluster].pop()
                    clusters[cluster].append(node_to_move)
                    cluster_weights[cluster] += node_weights[node_to_move]
                    cluster_weights[donor_cluster] -= node_weights[node_to_move]
                    if cluster_weights[cluster] >= limits[cluster][0]:
                        break

    return clusters, cluster_weights

# Function to calculate the score of a partition
def calculate_score_instance(edges, clusters):
    score = 0
    for cluster in clusters.values():
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                nodeA, nodeB = cluster[i], cluster[j]
                for (elementA, elementB, edge_weight) in edges:
                    if (elementA == nodeA and elementB == nodeB) or (elementA == nodeB and elementB == nodeA):
                        score += edge_weight
    return score

# Function to check if a move is valid in terms of capacity limits
def is_valid_move(cluster, node, node_weights, limits):
    total_weight = sum(node_weights[n] for n in cluster) + node_weights[node]
    return limits[0] <= total_weight <= limits[1]

# Simulated Annealing function
def simulated_annealing(M, C, limits, node_weights, edges, initial_clusters, initial_score, max_iterations=10000, initial_temp=500, cooling_rate=0.995):
    current_clusters = initial_clusters.copy()
    current_score = initial_score
    best_clusters = current_clusters.copy()
    best_score = current_score

    temperature = initial_temp
    scores = []
    times = []
    stale_iterations = 0
    stale_threshold = 1000  # Stop if no improvement for 1000 iterations

    for iteration in range(max_iterations):
        if temperature < 1e-3 or stale_iterations >= stale_threshold:  # Stop if temperature is close to 0 or if no improvement
            break
        
        start_time = time.time()

        # Randomly choose a node and attempt to move it to a different cluster
        node = random.randint(0, M - 1)
        current_cluster = None
        for cluster_id, cluster in current_clusters.items():
            if node in cluster:
                current_cluster = cluster_id
                break

        new_cluster = random.choice([c for c in range(C) if c != current_cluster])

        # Ensure current cluster has enough nodes to remove and new cluster doesn't exceed limits
        if len(current_clusters[current_cluster]) > 1 and is_valid_move(current_clusters[new_cluster], node, node_weights, limits[new_cluster]):
            # Create a new configuration by moving the node to the new cluster
            new_clusters = {k: v[:] for k, v in current_clusters.items()}
            new_clusters[current_cluster].remove(node)
            new_clusters[new_cluster].append(node)

            # Calculate only the delta in the score
            new_score = current_score
            for (elementA, elementB, edge_weight) in edges:
                if (elementA == node and elementB in current_clusters[current_cluster]) or (elementB == node and elementA in current_clusters[current_cluster]):
                    new_score -= edge_weight
                if (elementA == node and elementB in new_clusters[new_cluster]) or (elementB == node and elementA in new_clusters[new_cluster]):
                    new_score += edge_weight

            # Ensure new cluster weights are within the limits
            new_cluster_weight = sum(node_weights[n] for n in new_clusters[new_cluster])
            current_cluster_weight = sum(node_weights[n] for n in new_clusters[current_cluster])
            if limits[new_cluster][0] <= new_cluster_weight <= limits[new_cluster][1] and limits[current_cluster][0] <= current_cluster_weight <= limits[current_cluster][1]:
                # Accept new solution if it's better or with a probability based on temperature
                if new_score > current_score or random.uniform(0, 1) < math.exp((new_score - current_score) / temperature):
                    current_clusters = new_clusters
                    current_score = new_score
                    stale_iterations = 0

                    # Update the best solution found
                    if current_score > best_score:
                        best_clusters = current_clusters.copy()
                        best_score = current_score
                else:
                    stale_iterations += 1
            else:
                stale_iterations += 1

        # Reduce temperature according to the cooling schedule
        temperature *= cooling_rate  
        end_time = time.time()
        elapsed_time = end_time - start_time
        scores.append(current_score)
        times.append(elapsed_time)

    return best_clusters, best_score, scores, times


# Main execution
file_path = r'C:\Users\omar2\OneDrive\سطح المكتب\CS361Project_AbdulrhmanAlandas_OmarAlsuraia_OmarAlshuail_KhaledAlTomihi\RanReal480\RanReal480.txt'
M, C, limits, node_weights, edges = parse_instance(file_path)

# Greedy heuristic for initial solution
start_time = time.time()
initial_clusters, initial_weights = greedy_heuristic_instance(M, C, limits, node_weights, edges)
initial_score = calculate_score_instance(edges, initial_clusters)
end_time = time.time()
greedy_time = end_time - start_time

print(f"Initial Clusters (Greedy Heuristic):")
for i, cluster in initial_clusters.items():
    print(f"  Cluster {i + 1}: {cluster}, Weight: {initial_weights[i]}")
print(f"Initial Score: {initial_score}")
print(f"Greedy Heuristic Time: {greedy_time:.4f} seconds")

# Plot weights for each cluster from Greedy Heuristic
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
cluster_ids = range(1, C + 1)
plt.bar(cluster_ids, initial_weights.values(), color='skyblue')
plt.title('Weights of Initial Clusters (Greedy Heuristic)')
plt.xlabel('Cluster')
plt.ylabel('Weight')
plt.grid(True)

# Plot the initial score as a bar plot
plt.subplot(1, 2, 2)
plt.bar(['Initial Score'], [initial_score], color='orange')
plt.title('Initial Score (Greedy Heuristic)')
plt.ylabel('Score')
plt.grid(True)

plt.tight_layout()
plt.show()

# Run multiple times for Simulated Annealing
num_runs = 10
all_scores = []
all_times = []

for run in range(num_runs):
    print(f"\nRun {run + 1}:")

    # Run Simulated Annealing with initial clusters from greedy heuristic
    start_time = time.time()
    best_clusters, best_score, sa_scores, sa_times = simulated_annealing(M, C, limits, node_weights, edges, initial_clusters, initial_score)
    end_time = time.time()
    sa_time = end_time - start_time

    print(f"Run Simulated Annealing Results:")
    for i, cluster in best_clusters.items():
        cluster_weight = sum(node_weights[node] for node in cluster)
        print(f"  Cluster {i + 1}: {cluster}, Weight: {cluster_weight}")
    print(f"Best Score: {best_score}")
    print(f"Simulated Annealing Time: {sa_time:.4f} seconds")

    all_scores.append(best_score)
    all_times.append(sum(sa_times))

# Plot scores and times for all runs
runs = range(1, num_runs + 1)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(runs, all_scores, marker='o', linestyle='-', color='b', label='Best Score')
plt.title('Best Score in Each Run')
plt.xlabel('Run')
plt.ylabel('Score')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(runs, all_times, marker='o', linestyle='-', color='r', label='Average Computation Time (seconds)')
plt.title('Average Computation Time in Each Run')
plt.xlabel('Run')
plt.ylabel('Time (seconds)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Final statistics
print("\nFinal Results:")
print(f"Best Score across all runs: {max(all_scores)}")
print(f"Average Score across all runs: {statistics.mean(all_scores)}")
print(f"Standard Deviation of Scores across all runs: {statistics.stdev(all_scores)}")
print(f"Average Computation Time across all runs: {statistics.mean(all_times):.4f} seconds")
