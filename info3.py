
import heapq

graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('D', 2), ('E', 4)],
    'C': [('F', 2)],
    'D': [],
    'E': [('G', 2)],
    'F': [('G', 1)],
    'G': []
}

h = {'A': 5, 'B': 4, 'C': 3, 'D': 4, 'E': 2, 'F': 1, 'G': 0}

def astar(start, goal):
    pq = [(h[start], 0, start, [start])]
    visited = set()

    while pq:
        f, cost, node, path = heapq.heappop(pq)

        if node == goal:
            print("Path:", path)
            print("Cost:", cost)
            return

        if node in visited:
            continue
        visited.add(node)

        for next_node, c in graph[node]:
            if next_node not in visited:
                new_cost = cost + c
                heapq.heappush(pq, (new_cost + h[next_node],
                                    new_cost, next_node,
                                    path + [next_node]))

astar('A', 'G')
