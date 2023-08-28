def bfs(graph, start):
    visited = set()  # To keep track of visited nodes
    queue = [start]  # Initialize the queue with the starting node

    while queue:
        node = queue.pop(0)  # Remove the front node from the queue
        if node not in visited:
            print(node, end=' ')  # Process the node (you can replace this with your desired action)
            visited.add(node)  # Mark the node as visited
            neighbors = graph[node]  # Get the neighbors of the current node
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)  # Add unvisited neighbors to the queue

# Example graph as an adjacency list
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

start_node = 'A'
print("BFS traversal starting from", start_node, ":")
bfs(graph, start_node)