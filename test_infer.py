import unittest
from factor import Factor
from infer import Model


class TestBuildGraph(unittest.TestCase):
    def test_get_largest_sepset_small(self):
        a = Factor([(0, 2), (1, 2), (2, 2)])
        b = Factor([(2, 2), (3, 2), (3, 2)])

        model = Model([a, b])
        model.build_graph()

        print model.edges
        print [(a, b)]
        self.assertItemsEqual(model.edges[0], (a, b))

    def test_get_largest_sepset_larger(self):
        a = Factor([(0, 2), (1, 2), (2, 2)])
        b = Factor([(2, 2), (3, 2), (4, 2)])
        c = Factor([(1, 2), (2, 2), (5, 3), (6, 3)])

        model = Model([a, b, c])
        model.build_graph()

        print model.edges[0]
        print [(a, c)]
        print
        print model.edges[1]
        print [(a, b)]
        # Expect:
        #  a{0 1 2} --[2 3]-- c{1 2 5 6}
        #      \
        #       [0]
        #          \
        #           b{0 3 4}
        self.assertItemsEqual(model.edges[0], (a, c))
        self.assertItemsEqual(model.edges[1], (a, b))

    def test_get_largest_sepset_large(self):
        a = Factor([0, 1, 2, 3, 4, 5])
        b = Factor([1, 2, 3, 4, 5, 6])
        c = Factor([3, 4, 5, 6, 8])
        d = Factor([0, 1, 2, 7])
        e = Factor([1, 7, 8])

        # a{0 1 2 3 4 5} --[1 2 3 4 5]-- b{1 2 3 4 5 6} --[3 4 5 6]-- c{3 4 5 6 8}
        #      \                                                    /
        #    [0 1 2]                                            [8]
        #         \                                           /
        #       d{0 1 2 7} ---------[1 7]---------------   e{1 7 8}

        model = Model([a, b, c, d, e])
        model.build_graph()

        expected_edges = [(a, b), (b, c), (a, d), (d, e), (e, c)]
        for expected_edge, edge in zip(expected_edges, model.edges):
            print edge[0], edge[1]
            print expected_edge[0], expected_edge[1]
            print
            #self.assertItemsEqual(edge, expected_edge)

if __name__ == '__main__':
    unittest.main()
