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


class Monad(object):
    def __rshift__(self, other):
        return self * (lambda x: other)


class Maybe(Monad):
    def __init__(self, something=None, nothing=False):
        self.something = something
        self.nothing = nothing

    def is_something(self):
        return not self.nothing

    def is_nothing(self):
        return self.nothing

    @classmethod
    def unit(cls, something):
        return Maybe(something)

    def __mul__(self, fun):
        if self.is_nothing():
            return self
        else:
            return fun(self.something)

    @classmethod
    def nothing(cls):
        return Maybe(None,True)


class Reader(Monad):
    def __init__(self, reader):
        self.reader = reader

    @classmethod
    def unit(cls, value):
        return Reader(lambda x: value)

    def run(self, r):
        return self.reader(r)

    def __call__(self, r):
        return self.run(r)

    @classmethod
    def read(cls):
        return Reader(lambda r: r)

    def __mul__(self, fun):
        return Reader(lambda r: fun(self(r))(r))


class Writer(Monad):
    def __init__(self, value, monoid_empty=list, monoid_concat=lambda a, b: a + b):
        self.value = value
        self.monoid_empty = monoid_empty
        self.monoid_concat = monoid_concat
        self.monoid = self.monoid_empty()

    @classmethod
    def unit(cls, value, monoid_empty=list, monoid_concat=lambda a, b: a + b):
        return Writer(value, monoid_empty, monoid_concat)

    @classmethod
    def write(cls, monoid):
        return Writer(None, lambda: monoid)

    def __mul__(self, fun):
        other = fun(self.value)
        return Writer(other.value, lambda: self.monoid_concat(self.monoid, other.monoid), self.monoid_concat)



class State(Monad):
    def __init__(self, state_fun):
        self.state_fun = state_fun

    @classmethod
    def unit(cls, value):
        return State(lambda s: (value, s))

    def run(self, s):
        return self.state_fun(s)

    def __call__(self, r):
        return self.run(r)

    def __mod__(self, fun):
        def aux(s):
            (x, z) = self(s)
            return fun(x)(z)

        return State(aux)

    def eval(self, s):
        return self(s)[0]

    @classmethod
    def get(cls):
        return State(lambda s: (s, s))

    @classmethod
    def set(cls, s):
        return State(lambda _: (None, s))

    @classmethod
    def modify(cls, f):
        return State.get() * (lambda s: State.set(f(s)))


class List(Monad):
    def __init__(self, lst):
        self.lst = lst

    @classmethod
    def unit(cls, value):
        return List([value])

    def __mul__(self, fun):
        return List([x for l in [fun(y) for y in self.lst] for x in l.lst])

    def map(self, f):
        return List(map(f, self.lst))

    def __getitem__(self, item):
        if type(item) == int:
            return self.lst[item]
        else:
            return List(self.lst.__getitem__(item))

    def __str__(self):
        return self.lst.__str__()

