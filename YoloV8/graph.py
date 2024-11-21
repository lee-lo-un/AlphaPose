import numpy as np

class Graph():
    def __init__(self, layout='openpose', strategy='spatial'):
        self.get_edge(layout)
        self.get_adjacency(strategy)

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                           (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                           (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        else:
            raise ValueError(f"미지원 레이아웃: {layout}")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.num_node)
        adjacency = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            adjacency[j, i] = 1
            adjacency[i, j] = 1
        
        if strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = adjacency.copy()
                a_further = np.zeros((self.num_node, self.num_node))
                for h in range(hop):
                    a_further += np.linalg.matrix_power(adjacency, h + 1)
                a_further = (a_further > 0) * 1
                
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_close + a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError(f"미지원 전략: {strategy}") 