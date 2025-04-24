import numpy as np
import matplotlib.pyplot as plt
import random

# Simulation settings
NUM_NODES = 100
FIELD_SIZE = 100
BS_POSITION = (50, 50)  # Base station in the center

# Generate random node positions
nodes = np.random.rand(NUM_NODES, 2) * FIELD_SIZE

# Flat routing simulation (each node sends to BS)
def flat_routing(nodes, bs_pos):
    total_distance = 0
    for node in nodes:
        dist = np.linalg.norm(node - bs_pos)
        total_distance += dist
    return total_distance

# LEACH-style clustering routing simulation
def hierarchical_routing(nodes, bs_pos, p=0.05):
    num_clusters = int(NUM_NODES * p)
    cluster_heads = nodes[np.random.choice(range(NUM_NODES), num_clusters, replace=False)]
    total_distance = 0

    for node in nodes:
        if node.tolist() in cluster_heads.tolist():
            continue  # Skip cluster heads themselves
        # Send to nearest cluster head
        distances = np.linalg.norm(cluster_heads - node, axis=1)
        total_distance += min(distances)

    # CHs send to BS
    for ch in cluster_heads:
        total_distance += np.linalg.norm(ch - bs_pos)

    return total_distance

# Location-based routing: only nodes in a certain region transmit
def location_based_routing(nodes, bs_pos, region=(30, 70)):
    total_distance = 0
    for node in nodes:
        if region[0] < node[0] < region[1] and region[0] < node[1] < region[1]:
            total_distance += np.linalg.norm(node - bs_pos)
    return total_distance

# Plot WSN field
def plot_wsn(nodes, bs_pos, title):
    plt.figure(figsize=(6,6))
    plt.scatter(nodes[:,0], nodes[:,1], c='blue', label='Sensor Nodes')
    plt.scatter(bs_pos[0], bs_pos[1], c='red', marker='X', label='Base Station')
    plt.title(title)
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Simulate and display results
plot_wsn(nodes, BS_POSITION, "WSN Deployment")
print("Total communication cost (distance units):")
print("Flat routing:", flat_routing(nodes, BS_POSITION))
print("Hierarchical (LEACH-style) routing:", hierarchical_routing(nodes, BS_POSITION))
print("Location-based routing:", location_based_routing(nodes, BS_POSITION))
