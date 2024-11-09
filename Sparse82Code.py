import random
import time
import statistics
import matplotlib.pyplot as plt
import math

def read_sparse_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        header = lines[0].split()
        num_nodes = int(header[0])
        num_clusters = int(header[1])
        min_weight = int(header[2])
        max_weight = int(header[3])
        node_weights = list(map(int, header[header.index('W') + 1:]))
    return num_nodes, num_clusters, min_weight, max_weight, node_weights

def read_edges_file(file_path):
    edges = {}
    with open(file_path, 'r') as f:
        for line in f:
            node1, node2, weight = map(float, line.split())
            edges[(int(node1), int(node2))] = weight
    return edges

def calculate_score(clusters, edges):
    score = 0
    for cluster in clusters:
        for i, node1 in enumerate(cluster):
            for node2 in cluster[i + 1:]:
                if (int(node1.split()[1])-1, int(node2.split()[1])-1) in edges:
                    score += edges[(int(node1.split()[1])-1, int(node2.split()[1])-1)]
                elif (int(node2.split()[1])-1, int(node1.split()[1])-1) in edges:
                    score += edges[(int(node2.split()[1])-1, int(node1.split()[1])-1)]
    return max(0, score)

def calculate_weights(clusters, nodes):
    weights = [sum(nodes[node] for node in cluster) for cluster in clusters]
    return weights

def greedy_heuristic_random(nodes, L, U, num_clusters):
    nodes_list = list(nodes.items())
    random.shuffle(nodes_list)
    clusters = [[] for _ in range(num_clusters)]
    cluster_weights = [0] * num_clusters
# 70 - 9
    for node, weight in nodes_list:
        available_clusters = [i for i in range(num_clusters) if cluster_weights[i] + weight <= U]
        if available_clusters:
            selected_cluster = random.choice(available_clusters)
            clusters[selected_cluster].append(node)
            cluster_weights[selected_cluster] += weight

    for i in range(num_clusters):
        if cluster_weights[i] < L:
            for j in range(num_clusters):
                if i != j and cluster_weights[j] > L:
                    node_to_move = clusters[j].pop()
                    clusters[i].append(node_to_move)
                    cluster_weights[i] += nodes[node_to_move]
                    cluster_weights[j] -= nodes[node_to_move]
                    if cluster_weights[i] >= L:
                        break

    return clusters

def plot_greedy_heuristic_results(initial_clusters, initial_weights, initial_score):
    num_clusters = len(initial_clusters)
    
    # Plotting weights for each cluster
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    cluster_ids = range(1, num_clusters + 1)
    plt.bar(cluster_ids, initial_weights, color='skyblue')
    plt.title('Weights of Initial Clusters (Greedy Heuristic)')
    plt.xlabel('Cluster')
    plt.ylabel('Total Weight')
    plt.grid(True)

    # Showing initial score
    plt.subplot(1, 2, 2)
    plt.bar(['Initial Score'], [initial_score], color='orange')
    plt.title('Initial Score (Greedy Heuristic)')
    plt.ylabel('Score')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def simulated_annealing(clusters, nodes, edges, L, U, initial_temperature, cooling_rate, max_iterations):
    current_clusters = clusters
    current_score = calculate_score(current_clusters, edges)
    best_score = current_score
    best_clusters = current_clusters

    temperature = initial_temperature

    for iteration in range(max_iterations):
        new_clusters = [cluster.copy() for cluster in current_clusters]

        # Randomly select two clusters to swap a node between them
        cluster_index1 = random.randint(0, len(new_clusters) - 1)
        cluster_index2 = random.randint(0, len(new_clusters) - 1)

        if cluster_index1 != cluster_index2 and new_clusters[cluster_index1] and new_clusters[cluster_index2]:
            node1 = random.choice(new_clusters[cluster_index1])
            node2 = random.choice(new_clusters[cluster_index2])

            # Swap nodes
            new_clusters[cluster_index1].remove(node1)
            new_clusters[cluster_index2].append(node1)
            new_clusters[cluster_index2].remove(node2)
            new_clusters[cluster_index1].append(node2)

            new_score = calculate_score(new_clusters, edges)
            delta_score = new_score - current_score

            # Accept new solution based on temperature
            if delta_score > 0 or random.uniform(0, 1) < math.exp(delta_score / temperature):
                current_clusters = new_clusters
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_clusters = current_clusters

        # Cool down the temperature
        temperature *= cooling_rate

    return best_clusters, best_score

def hill_climbing(initial_clusters, nodes, edges, L, U, max_swaps=5000):  
    current_clusters = initial_clusters
    current_score = calculate_score(current_clusters, edges)
    best_score = current_score
    best_clusters = current_clusters

    while True:
        improved = False
        best_attempt_score = current_score

        for swap_attempt in range(max_swaps):
            cluster_index1 = random.randint(0, len(current_clusters) - 1)
            cluster_index2 = random.randint(0, len(current_clusters) - 1)

            if cluster_index1 != cluster_index2:
                if current_clusters[cluster_index1] and current_clusters[cluster_index2]:
                    max_nodes_to_swap = min(len(current_clusters[cluster_index1]), len(current_clusters[cluster_index2]))
                    num_nodes_to_swap = random.randint(1, max_nodes_to_swap)  # Ensure it's not larger than available
                    nodes_to_swap1 = random.sample(current_clusters[cluster_index1], num_nodes_to_swap)
                    nodes_to_swap2 = random.sample(current_clusters[cluster_index2], num_nodes_to_swap)

                    temp_clusters = [cluster.copy() for cluster in current_clusters]
                    for node in nodes_to_swap1:
                        temp_clusters[cluster_index1].remove(node)
                        temp_clusters[cluster_index2].append(node)
                    for node in nodes_to_swap2:
                        temp_clusters[cluster_index2].remove(node)
                        temp_clusters[cluster_index1].append(node)

                    temp_score = calculate_score(temp_clusters, edges)

                    if temp_score > best_attempt_score:
                        best_attempt_score = temp_score
                        best_clusters = temp_clusters
                        improved = True

        if improved:
            current_clusters = best_clusters
            current_score = best_attempt_score
            if best_attempt_score > best_score:
                best_score = best_attempt_score
        else:
            break  # لا توجد تحسينات جديدة

    return current_clusters, best_score

def run_multiple_hill_climbing(nodes, edges, L, U, runs=10):  # Set runs to 10
    initial_clusters = greedy_heuristic_random(nodes, L, U, num_clusters)
    initial_weights = calculate_weights(initial_clusters, nodes)
    initial_score = calculate_score(initial_clusters, edges)
    print(f"Initial clusters (Greedy Heuristic): {initial_clusters}, Weights: {initial_weights}, Score: {initial_score}")

    # Plot initial Greedy Heuristic results
    plot_greedy_heuristic_results(initial_clusters, initial_weights, initial_score)

    best_score = initial_score
    scores = []
    times = []

    for run in range(runs):
        start_time = time.time()
        
        # First use hill climbing
        improved_clusters, improved_score = hill_climbing(initial_clusters, nodes, edges, L, U)
        
        # Then refine using simulated annealing
        refined_clusters, refined_score = simulated_annealing(improved_clusters, nodes, edges, L, U, initial_temperature=4000, cooling_rate=0.9, max_iterations=10000)  
        
        end_time = time.time()

        elapsed_time = end_time - start_time
        scores.append(refined_score)
        times.append(elapsed_time)

        improved_weights = calculate_weights(refined_clusters, nodes)
        
        print(f"\nRun {run + 1} Resulting Clusters:")
        print(f"  Clusters: {refined_clusters}")
        print(f"  Weights: {improved_weights}")
        print(f"  Score: {refined_score}")
        print(f"  Time: {elapsed_time:.4f} seconds")

        # التحقق من التحسين
        if refined_score > best_score:
            best_score = refined_score
            initial_clusters = refined_clusters  # Update to the best clusters
            print("Improved solution found!")
        else:
            print("No improvement found, reverting to previous clusters.")
            # إعادة تشغيل الخوارزمية مع أفضل الحلول السابقة
            improved_clusters, improved_score = hill_climbing(initial_clusters, nodes, edges, L, U)

    print("\nFinal Results:")
    print(f"Best Score: {best_score}")
    print(f"Average Score: {statistics.mean(scores)}")
    print(f"Standard Deviation of Scores: {statistics.stdev(scores)}")
    print(f"Average Computation Time: {statistics.mean(times):.4f} seconds")

    return scores, times

# Read data from files
sparse_file_path =r'C:\Users\omar2\OneDrive\سطح المكتب\CS361Project_AbdulrhmanAlandas_OmarAlsuraia_OmarAlshuail_KhaledAlTomihi\Sparse82\Sparse82.txt'
edges_file_path = r'C:\Users\omar2\OneDrive\سطح المكتب\CS361Project_AbdulrhmanAlandas_OmarAlsuraia_OmarAlshuail_KhaledAlTomihi\Sparse82\edges_only.txt'

num_nodes, num_clusters, L, U, node_weights = read_sparse_file(sparse_file_path)
edges = read_edges_file(edges_file_path)
nodes = {f'Node {i+1}': node_weights[i] for i in range(num_nodes)}

# Run multiple hill climbing searches
scores, times = run_multiple_hill_climbing(nodes, edges, L, U, runs=10)

# Plot results
runs = range(1, len(scores) + 1)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(runs, scores, marker='o', linestyle='-', color='b', label='Best Score')
plt.title('Best Score in Each Run')
plt.xlabel('Run')
plt.ylabel('Score')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(runs, times, marker='o', linestyle='-', color='r', label='Computation Time (seconds)')
plt.title('Computation Time in Each Run')
plt.xlabel('Run')
plt.ylabel('Time (seconds)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()