"""
This module contains several tests for the functions
written in project.py.
"""
from cse163_utils import assert_equals

import pandas as pd


import project


data1 = pd.read_csv('/home/economicdata2017-2017.csv')
data2 = pd.read_csv('/home/economic_test.csv')


def test_highest_freedom_index():
    """
    Tests the function highest_freedom_index in project.py
    """
    print('Testing highest_freedom_index')

    assert_equals(('Hong Kong', 8.91), project.highest_freedom_index(data1))
    assert_equals(('Australia', 8.07), project.highest_freedom_index(data2))


def test_lowest_freedom_index():
    """
    Tests the function lowest_freedom_index in project.py
    """
    print('Testing lowest_freedom_index')

    assert_equals(('Venezuela', 2.58), project.lowest_freedom_index(data1))
    assert_equals(('Angola', 4.83), project.lowest_freedom_index(data2))


def main():
    test_highest_freedom_index()
    test_lowest_freedom_index()


if __name__ == '__main__':
    main()
