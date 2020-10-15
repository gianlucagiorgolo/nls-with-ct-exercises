# coding=utf-8

# The MIT License (MIT)
# Copyright (c) 2017 Gianluca Giorgolo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

import collections 
from copy import copy
from mb.monads import Maybe
from IPython.display import SVG, display
import mb.svg as svg

def typecheckedequality(eq_f):
    def typed_eq(a, b, *rest):
        return type(a) == type(b) and eq_f(a, b, *rest)

    return typed_eq

def find(lst, predicate):
    for l in lst:
        if predicate(l):
            return Maybe(l)
    return Maybe.nothing()


class EqualitySet(object):
    """A class implementing a set interface based on equality"""
    def __init__(self):
        self.elements = list()

    @classmethod
    def empty_set(cls):
        return EqualitySet()

    @classmethod
    def singleton(cls, e):
        s = EqualitySet()
        s.add(e)
        return s

    def to_list(self):
        return self.elements

    def add(self, element):
        if not element in self.elements:
            self.elements.append(element)

    def __contains__(self, item):
        return item in self.elements

    def __iter__(self):
        return iter(self.elements)

    def union(self, other):
        s = EqualitySet.empty_set()
        for e in self.elements:
            if not e in s:
                s.add(e)
        for e in other:
            if not e in s:
                s.add(e)
        return s

    def __str__(self):
        return str(self.elements)
    
    @classmethod
    def filter(cls, eqset, predicate):
        """Filters the equality set and returns a new equality set that contains only the elements
           that stisfy the passed predicate
        """
        res = EqualitySet()
        for element in eqset.elements:
            if predicate(element):
                res.add(element)
        return res

class Map(collections.MutableMapping):
    """Very simple map object based on equality rather than hashing"""
    def __init__(self):
        self.mapping = list()

    def __getitem__(self, key):
        for k,v in self.mapping:
            if k == key:
                return v
        raise KeyError

    def __len__(self):
        return len(self.mapping)

    def __iter__(self):
        for e in self.mapping:
            yield e

    def __setitem__(self, key, value):
        for i in range(len(self.mapping)):
            if self.mapping[i][0] == key:
                self.mapping[i] = (key,value)
                return None
        self.mapping.append((key,value))

    def __delitem__(self, key):
        index = None
        for i in range(len(self.mapping)):
            k,_ = self.mapping[i]
            if k == key:
                index = i
                break
        if not index is None:
            del self.mapping[i]

    def remove(self, key):
        """A non destructive version of del, returns a new copy of the map with the key removed"""
        new = copy(self)
        del new[key]
        return new


    def keys(self):
        return map(lambda x: x[0],self.mapping)

def mean(x,y):
    return (x + y) / 2.0

def unzip(list_of_pairs):
    acc1 = list()
    acc2 = list()
    for x,y in list_of_pairs:
        acc1.append(x)
        acc2.append(y)
    return (acc1,acc2)

def rev(lst):
    return lst[::-1]

class Tree(object):
    def __init__(self, item, children=None):
        self.item = item
        if children is None:
            children = list()
        self.children = children
        self.x_position = 0

    def pprint(self, prefix, is_tail, show_fun=str):
        res = prefix
        if is_tail:
            res += '└── '
            child_pref = '    '
        else:
            res += '├── '
            child_pref = '|   '
        res += show_fun(self.item) + '\n'
        for i in range(len(self.children) - 1):
            res += self.children[i].pprint(prefix + child_pref, False)
        if len(self.children) > 0:
            res += self.children[-1].pprint(prefix + child_pref, True)
        return res

    def get_depth(self):
        mx = 0
        for c in self.children:
            mx = max(mx,c.get_depth())
        return mx + 1

    def is_leaf(self):
        return len(self.children) == 0

    # SVG stuff

    minimum_node_separation = 1.0


    def pprint_pos(self, prefix, is_tail, show_fun=str):
        res = prefix
        if is_tail:
            res += '└── '
            child_pref = '    '
        else:
            res += '├── '
            child_pref = '|   '
        res += show_fun(self.item) + ' : ' + str(self.x_position) + '\n'
        for i in range(len(self.children) - 1):
            res += self.children[i].pprint_pos(prefix + child_pref, False)
        if len(self.children) > 0:
            res += self.children[-1].pprint_pos(prefix + child_pref, True)
        return res


    def display(self):
        display(SVG(str(self.to_svg())))

    def to_svg(self, x=0, y=0):
        raise NotImplementedError()

    def _repr_svg_(self):
        return str(self.to_svg())


# Reingold-Tilford

def reingold_tilford(tree):
    def walk1(tree,x):
        if tree.is_leaf():
            tree.x_position = x
        else:
            for i in range(len(tree.children)):
                walk1(tree.children[i],i-1)
