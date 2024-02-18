import osmnx as ox
import random
import networkx as nx
import heapq
import random
import time
import matplotlib.pyplot as plt

# Define the heuristic function (straight-line distance + random value as risk function)
def hstd(node1, node2):
    
    hstd_value = ox.distance.euclidean_dist_vec(graph.nodes[node1]['y'], graph.nodes[node1]['x'],
                                                 graph.nodes[node2]['y'], graph.nodes[node2]['x'])
    
    risk_value = random.random() * 100  # Generate a random value between 0 and 100
    
   
    return hstd_value + risk_value

# Implement Dijkstra's algorithm to calculate actual cost
def dijkstra(graph, start_node):
    
    dist = {node: float('inf') for node in graph.nodes}
    dist[start_node] = 0
    predecessors = {}
    # Priority queue to store nodes to visit
    pq = [(0, start_node)]
   
    while pq:
        # Pop the node with the smallest distance from the priority queue
        current_dist, current_node = heapq.heappop(pq)
        # Check if this is a shorter path to current_node
        if current_dist > dist[current_node]:
            continue
        # Iterate over neighbors of current_node
        for neighbor, edge in graph[current_node].items():
            weight = edge.get('weight', 1)  
            
            new_dist = dist[current_node] + weight
            # Update distance and predecessor if shorter path is found
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                predecessors[neighbor] = current_node
                # Add neighbor to priority queue
                heapq.heappush(pq, (new_dist, neighbor))
    return dist, predecessors

# Define the evaluation function (actual cost + heuristic)
def evaluation_function(node, target_node, actual_cost, heuristic):
    return actual_cost[node] + heuristic(node, target_node)

# Define the evaluation function (actual cost + heuristic)
def evaluation_function_weight(node, target_node, actual_cost, heuristic):
    weigh=4
    return actual_cost[node] + weigh * heuristic(node, target_node)

# Implement Greedy Best First Search algorithm
def greedy_best_first_search(graph, start_node, target_node, heuristic):
    visited = set()
    pq = [(heuristic(start_node, target_node), start_node)]
    while pq:
        _, current_node = heapq.heappop(pq)
        if current_node == target_node:
            return visited
        visited.add(current_node)
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                heapq.heappush(pq, (heuristic(neighbor, target_node), neighbor))
    return visited

# Implement A* algorithm
def a_star(graph, start_node, target_node, actual_cost, heuristic):
    visited = set()
    pq = [(evaluation_function(start_node, target_node, actual_cost, heuristic), start_node)]
    while pq:
        _, current_node = heapq.heappop(pq)
        if current_node == target_node:
            return visited
        visited.add(current_node)
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                heapq.heappush(pq, (evaluation_function(neighbor, target_node, actual_cost, heuristic), neighbor))
    return visited

# Implement Weighted A* algorithm
def weighted_a_star(graph, start_node, target_node, actual_cost, heuristic):
    visited = set()
    pq = [(evaluation_function_weight(start_node, target_node, actual_cost, heuristic), start_node)]
    while pq:
        _, current_node = heapq.heappop(pq)
        if current_node == target_node:
            return visited
        visited.add(current_node)
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                heapq.heappush(pq, (evaluation_function_weight(neighbor, target_node, actual_cost, heuristic) , neighbor))
    return visited

# Download OSM data for Feni, Bangladesh
city_name = "Feni, Bangladesh"
graph = ox.graph_from_place(city_name, network_type='all')

# Access nodes and edges
nodes = graph.nodes
edges = graph.edges

# Open a file in write mode
with open("node.txt", "w") as file:
    # Iterate over nodes and write them to the file
    for node, data in nodes(data=True):
        file.write(f"{node}: {data}\n")



# Get list of all nodes in the graph
all_nodes = list(graph.nodes)

# Choose random start and target nodes
start_node = random.choice(all_nodes)
target_node = random.choice(all_nodes)

# Run Dijkstra's algorithm
start_time = time.time()
actual_cost, _ = dijkstra(graph, start_node)
dijkstra_time = time.time() - start_time

# Run Greedy Best First Search
start_time = time.time()
gbfs_visited = greedy_best_first_search(graph, start_node, target_node, hstd)
gbfs_time = time.time() - start_time
gbfs_path = list(gbfs_visited)

# Write GBFS path to file
with open("gbfs.txt", "w") as file:
    file.write("GBFS Path:\n")
    file.write(",".join(str(node) for node in gbfs_path))

# Run A*
start_time = time.time()
a_star_visited = a_star(graph, start_node, target_node, actual_cost, hstd)
a_star_time = time.time() - start_time
a_star_path = list(a_star_visited)

# Write A* path to file
with open("astar.txt", "w") as file:
    file.write("A* Path:\n")
    file.write(",".join(str(node) for node in a_star_path))

# Run Weighted A* (with weight = 2)
start_time = time.time()
weighted_a_star_visited = weighted_a_star(graph, start_node, target_node, actual_cost, hstd)
weighted_a_star_time = time.time() - start_time
weighted_a_star_path = list(weighted_a_star_visited)

# Write Weighted A* path to file
with open("weightedastar.txt", "w") as file:
    file.write("Weighted A* Path:\n")
    file.write(",".join(str(node) for node in weighted_a_star_path))


# Print chosen start and target nodes
print("Start Node:", start_node)
print("Target Node:", target_node)

# Convert time to milliseconds
gbfs_time_ms = gbfs_time * 1000
a_star_time_ms = a_star_time * 1000
weighted_a_star_time_ms = weighted_a_star_time * 1000

# Print results (memory, time, search space)
print("\nGreedy Best First Search:")
print("Memory:", len(gbfs_visited)*8,"Bytes")
print("Time:", gbfs_time_ms, "ms")
print("Search Space:", len(gbfs_visited))

print("\nA*:")
print("Memory:", len(a_star_visited)*8,"Bytes")
print("Time:", a_star_time_ms, "ms")
print("Search Space:", len(a_star_visited))

print("\nWeighted A*:")
print("Memory:", len(weighted_a_star_visited)*8,"Bytes")
print("Time:", weighted_a_star_time_ms, "ms")
print("Search Space:", len(weighted_a_star_visited))

# Visualize the city map with a custom title
ox.plot_graph(ox.project_graph(graph))