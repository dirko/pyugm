"""
Helpers for tests.
"""

import unittest


class GraphTestCase(unittest.TestCase):
    # pylint: disable=too-many-public-methods
    """
    Test case with additional assertions to do with graphs.
    """
    @staticmethod
    def _assert_edge_equal(edge1, edge2, msg=''):
        """
        Assert that a pair of nodes is equal to another pair of nodes or its inverse.
        :param edge1: Tuple of two nodes.
        :param edge2: Tuple of two nodes.
        """
        if edge1 == edge2 or (edge1[1], edge1[0]) == edge2:
            pass
        else:
            raise AssertionError('Edges not equal, {0}) != {1}: \n{2}'.format(edge1, edge2, msg))

    def _assert_edge_sets_equal(self, set1, set2):
        """
        Assert that two sets contain the same undirected node pairs.
        :param set1: Set of tuples containing two nodes.
        :param set2: Set of tuples containing two nodes.
        """
        set1_copy = set(list(set1))
        for edge in list(set1):
            set1_copy.add((edge[1], edge[0]))
        set2_copy = set(list(set2))
        for edge in list(set2):
            set2_copy.add((edge[1], edge[0]))
        try:
            self.assertSetEqual(set1_copy, set2_copy)
        except AssertionError:
            for edge in list(set1_copy):
                if edge not in set2_copy:
                    print 'In 1 not in 2: ({0}, {1})'.format(edge[0], edge[1])
            for edge in list(set2_copy):
                if edge not in set1_copy:
                    print 'In 2 not in 1: ({0}, {1})'.format(edge[0], edge[1])
            raise


def print_edge_set(edges):
    """
    Helper to display edges.
    :param edges: Iterable of tuples.
    """
    for edge in list(edges):
        print '({0}, {1})'.format(edge[0], edge[1])
