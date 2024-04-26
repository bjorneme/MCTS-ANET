import numpy as np

# Disjoint Set: Used searching winning paths
class DisjointSet:
    def __init__(self, size):
        # Each element points to itself initially
        self.parent = np.arange(size, dtype=int)

        # Rank to keep the union opperation efficient
        self.rank = np.zeros(size, dtype=int)

    def find(self, x):
        # Find the root of x
        if self.parent[x] != x:
            
            # Apply path compression
            self.parent[x] = self.find(self.parent[x])

        # Return the parent of the node
        return self.parent[x]
    
    def union(self, x, y):
        # Find roots of x and y
        root_x = self.find(x)
        root_y = self.find(y)

        # If parents are different, do union by rank
        if root_x != root_y:
            
            # Root x has higher rank, make it the parent
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x

            # Root y has higher rank, make it the parent
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y

            # If ranks are the same, make one root of the other and increase rank
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

