from .tools import get_spatial_graph

num_node = 23
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(2, 5), (5, 8), (0, 3), (3, 6), (6, 9), (1, 4), (4, 7), (7, 12), 
            (12, 15), (15, 17), (17, 19), (19, 21), (7, 11), (11, 14), (14, 16), 
            (16, 18), (18, 20), (7, 10), (10, 22), (22, 13)]
skeleton_ntu_rgb_d = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:

    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == "__main__":
    sklist = []
    for sk in inward_ori_index:
        a = sk[0]
        b = sk[1]
        if a - 1 < 0 or b - 1 < 0:
            continue
        if a - 1 > 22 or b - 1 > 22:
            continue
        sklist.append((sk[0] - 1, sk[1] - 1))
    print(sklist)
