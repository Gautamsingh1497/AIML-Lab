#Using dictionary to create adjacency list
graph = {
    '4' : ['5','8'],
    '5' : ['2','4'],
    '7' : ['3'],
    '2' : [],
    '3' : ['8'],
    '8' : []
}
path_visited = set() # Set to keep track of visited nodes of the graph.
def dfs(path_visited, graph, n): #function for dfs
 if n not in path_visited:
    print (n)
    path_visited.add(n)
    for neighbor in graph[n]:
        dfs(path_visited, graph, neighbor) # Driver Code
print("Following is the Depth-First Search")
dfs(path_visited, graph, '5')

