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
from mb.util import typecheckedequality, Map, EqualitySet
import mb.html as html
from IPython.display import HTML, display


class Term(object):
    def __call__(self, arg):
        return Application(self, arg)

    def __xor__(self, other):
        return Abstraction(self, other)

    def __mul__(self, other):
        return Pair(self, other)

    def __pow__(self, other):
        return Bind(self, other)

    counter = 0

    @classmethod
    def fresh_variable(cls):
        Term.counter += 1
        v = Var('V' + str(Term.counter))
        v._internal = True
        return v

    @classmethod
    def reset_counter(cls):
        Term.counter = 0

    def rename_vars(self, rename_map):
        pass

    def to_python_executable(self, constants_map):
        return compile(self.to_python_source_code(constants_map), '<string>', 'eval')

    def to_python_source_code(self, constants_map):
        raise NotImplementedError

    def show(self):
        display(HTML(str(self.to_html())))



class Var(Term):
    def __init__(self, identifier):
        self.identifier = identifier
        self._internal = False

    def reduce(self, bindings=None):
        if bindings is None or not self in bindings:
            return self
        else:
            return bindings[self]

    def __hash__(self):
        return hash((self.identifier, 'var'))

    @typecheckedequality
    def __eq__(self, other):
        return self.identifier == other.identifier

    def __str__(self):
        return str(self.identifier)

    def free_vars(self):
        return [self]

    @typecheckedequality
    def alpha_equivalent(self, other, bindings=None):
        if bindings is None or not self in bindings:
            return self == other
        else:
            return other == bindings[self]

    def substitute(self, new, old):
        if self == old:
            return new
        else:
            return self

    def to_html(self):
        return html.text(self.identifier)

    def to_latex(self):
        return self.identifier

    def collect_vars(self):
        return EqualitySet.singleton(self)

    def rename_vars(self, rename_map):
        if self.identifier in rename_map:
            self.identifier = rename_map[self.identifier]

    def to_python_source_code(self, constants_map):
        return self.identifier

class Const(Term):
    def __init__(self, name):
        self.name = name

    def reduce(self, bindings=None):
        return self

    @typecheckedequality
    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return str(self.name)

    def free_vars(self):
        return []

    @typecheckedequality
    def alpha_equivalent(self, other, bindings=None):
        return self == other

    def substitute(self, new, old):
        if self == old:
            return new
        else:
            return self

    def to_html(self):
        return html.b(html.text(self.name))

    def to_latex(self):
        return '\\mathbf{' + self.name + '}'

    def collect_vars(self):
        return EqualitySet.empty_set()

    def to_python_source_code(self, constants_map):
        if self.name in constants_map:
            return constants_map[self.name]
        else:
            return self.name


class Abstraction(Term):
    def __init__(self, variable, body):
        if not type(variable) == Var:
            raise NameError("Hey hey you are trying to abstract something that's not a variable")
        self.variable = variable
        self.body = body

    def reduce(self, bindings=None):
        if not bindings is None:
            bindings = bindings.remove(self.variable)
            capturable = list()
            for k in bindings.keys():
                if self.variable in bindings[k].free_vars():
                    capturable.append(k)
            for c in capturable:
                bindings = bindings.remove(c)
        body_prime = self.body.reduce(bindings)
        # eta reduction
        if type(body_prime) == Application and self.variable == body_prime.argument:
            return body_prime.function
        else:
            return self.variable ^ body_prime
#        return self.variable ^ self.body.reduce(bindings)

    def substitute(self, new, old):
        if self.variable == old or self.variable in new.free_vars():
            return self
        else:
            return self.variable ^ self.body.substitute(new, old)

    @typecheckedequality
    def __eq__(self, other):
        return self.variable == other.variable and self.body == other.body

    def __str__(self):
        return '(λ ' + str(self.variable) + ' . ' + str(self.body) + ')'

    def to_html(self):
        return html.text('(') + html.raw_html('&lambda; ') + self.variable.to_html() + html.text(
            ' . ') + self.body.to_html() + html.text(')')

    def to_latex(self):
        return '(\\lambda ' + self.variable.to_latex() + ' . ' + self.body.to_latex() + ')' 

    def free_vars(self):
        frees = self.body.free_vars()
        if self.variable in frees:
            frees.remove(self.variable)
        return frees

    @typecheckedequality
    def alpha_equivalent(self, other, bindings=None):
        if bindings is None:
            bindings = Map()
        bindings[self.variable] = other.variable
        return self.body.alpha_equivalent(other.body, bindings)

    def collect_vars(self):
        return self.variable.collect_vars().union(self.body.collect_vars())

    def rename_vars(self, rename_map):
        self.variable.rename_vars(rename_map)
        self.body.rename_vars(rename_map)

    def to_python_source_code(self, constants_map):
        return '(lambda ' + self.variable.to_python_source_code(constants_map) + ' : ' + self.body.to_python_source_code(constants_map) + ')'


class Application(Term):
    def __init__(self, function, argument):
        self.function = function
        self.argument = argument

    def reduce(self, bindings=None):
        function_prime = self.function.reduce(bindings)
        argument_prime = self.argument.reduce(bindings)
        if type(function_prime) == Abstraction:
            if bindings is None:
                bindings = Map()
            bindings[function_prime.variable] = argument_prime
            return function_prime.body.reduce(bindings)
        else:
            return function_prime(argument_prime)

    def substitute(self, new, old):
        return self.function.substitute(new, old)(self.argument.substitute(new, old))

    @typecheckedequality
    def __eq__(self, other):
        return self.function == other.function and self.argument == other.argument

    def __str__(self):
        return str(self.function) + '(' + str(self.argument) + ')'

    def to_html(self):
        return self.function.to_html() + html.text('(') + self.argument.to_html() + html.text(')')

    def to_latex(self):
        return self.function.to_latex() + '(' + self.argument.to_latex() + ')'

    def free_vars(self):
        return self.function.free_vars() + self.argument.free_vars()

    @typecheckedequality
    def alpha_equivalent(self, other, bindings=None):
        return self.function.alpha_equivalent(other.function, bindings) and self.argument.alpha_equivalent(
            other.argument, bindings)

    def collect_vars(self):
        return self.argument.collect_vars().union(self.function.collect_vars())

    def rename_vars(self, rename_map):
        self.function.rename_vars(rename_map)

    def to_python_source_code(self, constants_map):
        return self.function.to_python_source_code(constants_map) + '(' + self.argument.to_python_source_code(constants_map) + ')'

class Pair(Term):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def reduce(self, bindings=None):
        return self.left.reduce(bindings) * self.right.reduce(bindings)

    def substitute(self, new, old):
        return self.left.substitute(new, old) * self.right.substitute(new, old)

    @typecheckedequality
    def __eq__(self, other):
        return self.left == other.left and self.right == other.right

    def __str__(self):
        return '<' + str(self.left) + ', ' + str(self.right) + '>'

    def to_html(self):
        return html.raw_html('&lang;') + self.left.to_html() + html.text(', ') + self.right.to_html() + html.raw_html(
            '&rang;')

    def to_latex(self):
        return '\\langle' + self.left.to_latex() + ' , ' + self.right.to_latex() + '\\rangle'

    def free_vars(self):
        return self.left.free_vars() + self.right.free_vars()

    @typecheckedequality
    def alpha_equivalent(self, other, bindings=None):
        return self.left.alpha_equivalent(other.left, bindings) and self.right.alpha_equivalent(other.right, bindings)

    def collect_vars(self):
        return self.left.collect_vars().union(self.right.collect_vars())

    def rename_vars(self, rename_map):
        self.left.rename_vars(rename_map)
        self.right.rename_vars(rename_map)

    def to_python_source_code(self, constants_map):
        return '(' + self.left.to_python_source_code(constants_map) + ',' + self.right.to_python_source_code(constants_map) + ')'


class FirstProjection(Term):
    def __init__(self, body):
        self.body = body

    def reduce(self, bindings=None):
        body_prime = self.body.reduce(bindings)
        if type(body_prime) == Pair:
            return body_prime.left
        else:
            return FirstProjection(body_prime)

    def substitute(self, new, old):
        return FirstProjection(self.body.substitute(new, old))

    @typecheckedequality
    def __eq__(self, other):
        return self.body == other.body

    def __str__(self):
        return 'p1(' + str(self.body) + ')'

    def to_html(self):
        return html.raw_html('&pi;1(') + self.body.to_html() + html.text(')')

    def to_latex(self):
        return '\\pi_1(' + self.body.to_latex() + ')'

    def free_vars(self):
        return self.body.free_vars()

    @typecheckedequality
    def alpha_equivalent(self, other, bindings=None):
        return self.body.alpha_equivalent(other.body, bindings)

    def collect_vars(self):
        return self.body.collect_vars()

    def rename_vars(self, rename_map):
        self.body.rename_vars(rename_map)

    def to_python_source_code(self, constants_map):
        return self.body.to_python_source_code(constants_map) + '[0]'


class SecondProjection(Term):
    def __init__(self, body):
        self.body = body

    def reduce(self, bindings=None):
        body_prime = self.body.reduce(bindings)
        if type(body_prime) == Pair:
            return body_prime.right
        else:
            return SecondProjection(body_prime)

    def substitute(self, new, old):
        return SecondProjection(self.body.substitute(new, old))

    @typecheckedequality
    def __eq__(self, other):
        return self.body == other.body

    def __str__(self):
        return 'p2(' + str(self.body) + ')'

    def to_html(self):
        return html.raw_html('&pi;2(') + self.body.to_html() + html.text(')')

    def to_latex(self):
        return '\\pi_2(' + self.body.to_latex() + ')'

    def free_vars(self):
        return self.body.free_vars()

    @typecheckedequality
    def alpha_equivalent(self, other, bindings=None):
        return self.body.alpha_equivalent(other.body, bindings)

    def collect_vars(self):
        return self.body.collect_vars()

    def rename_vars(self, rename_map):
        self.body.rename_vars(rename_map)

    def to_python_source_code(self, constants_map):
        return self.body.to_python_source_code(constants_map) + '[1]'



class Unit(Term):
    def __init__(self, body):
        self.body = body

    def reduce(self, bindings=None):
        return Unit(self.body.reduce(bindings))

    @typecheckedequality
    def __eq__(self, other):
        return self.body == other.body

    def __str__(self):
        return 'eta(' + str(self.body) + ')'

    def to_html(self):
        return html.raw_html('&eta;(') + self.body.to_html() + html.text(')')

    def to_latex(self):
        return '\\eta(' + self.body.to_latex() + ')'

    def free_vars(self):
        return self.body.free_vars()

    @typecheckedequality
    def alpha_equivalent(self, other, bindings=None):
        return self.body.alpha_equivalent(other.body, bindings)

    def substitute(self, new, old):
        return Unit(self.body.substitute(new, old))

    def collect_vars(self):
        return self.body.collect_vars()

    def rename_vars(self, rename_map):
        self.body.rename_vars(rename_map)


class Bind(Term):
    def __init__(self, monad, function):
        self.monad = monad
        self.function = function

    def reduce(self, bindings=None):
        monad_prime = self.monad.reduce(bindings)
        function_prime = self.function.reduce(bindings)
        if type(monad_prime) == Unit:
            return (function_prime(monad_prime.body)).reduce(bindings)
        else:
            return monad_prime ** function_prime

    @typecheckedequality
    def __eq__(self, other):
        return self.monad == other.monad and self.function == other.function

    def __str__(self):
        return str(self.monad) + ' ** (' + str(self.function) + ')'

    def to_html(self):
        return self.monad.to_html() + html.raw_html(' &starf; (') + self.function.to_html() + html.text(')')

    def to_latex(self):
        return self.monad.to_latex() + '\\star (' + self.function.to_latex() + ')'

    def free_vars(self):
        return self.monad.free_vars() + self.function.free_vars()

    @typecheckedequality
    def alpha_equivalent(self, other, bindings=None):
        return self.monad.alpha_equivalent(other.monad, bindings) and self.function.alpha_equivalent(other.function,
                                                                                                     bindings)

    def substitute(self, new, old):
        return self.monad.substitute(new, old) ** self.function.substitute(new, old)

    def collect_vars(self):
        return self.monad.collect_vars().union(self.function.collect_vars())

    def rename_vars(self, rename_map):
        self.monad.rename_vars(rename_map)
        self.function.rename_vars(rename_map)


# Testing

import unittest


class LambdaTest(unittest.TestCase):
    def test_reduce_canonical(self):
        x = Var('x')
        y = Var('y')
        z = Var('z')
        a = Var('a')
        b = Var('b')
        c = Var('c')
        d = Var('d')
        t1 = (x ^ x)(y ^ (z ^ z))
        o1 = y ^ (z ^ z)
        t2 = (x ^ ((y ^ y)(x)))
        o2 = x ^ x
        t3 = (x ^ (y ^ x))(a ^ a)
        o3 = y ^ (a ^ a)
        t4 = ((x ^ (y ^ x))(a ^ a))(b ^ b)
        o4 = a ^ a
        t5 = (x ^ (y ^ y))(a ^ a)
        o5 = y ^ y
        t6 = ((x ^ (y ^ y))(a ^ a))(b ^ b)
        o6 = b ^ b
        t7 = (a ^ (b ^ (a(a(a(b))))))(c ^ (d ^ (c(c(d)))))
        o7 = b ^ (d ^ (b(b(b(b(b(b(b(b(d))))))))))
        self.assertTrue(t1.reduce() == o1)
        self.assertTrue(t2.reduce() == o2)
        self.assertTrue(t3.reduce() == o3)
        self.assertTrue(t4.reduce() == o4)
        self.assertTrue(t5.reduce() == o5)
        self.assertTrue(t6.reduce() == o6)
        self.assertTrue(t7.reduce() == o7)

    def test_alpha_equivalence(self):
        x = Var('x')
        y = Var('y')
        z = Var('z')
        a = Var('a')
        b = Var('b')
        c = Var('c')
        n = Var('n')
        q = Var('q')
        t1 = x ^ (y ^ (x(y(z))))
        e1 = a ^ (b ^ (a(b(z))))
        self.assertTrue(t1.alpha_equivalent(e1))
        t2 = t1
        e2 = a ^ (b ^ (a(b(c))))
        self.assertFalse(t2.alpha_equivalent(e2))
        t3 = n ^ (a ^ (q ^ x(q(z(a)))))
        e3 = a ^ (b ^ (c ^ x(c(z(b)))))
        self.assertTrue(t3.alpha_equivalent(e3))

    def test_eta_equivalence(self):
        x = Var('x')
        y = Var('y')
        t1 = x ^ (y(x))
        self.assertTrue(t1.reduce() == y)

def runTests():
    suite = unittest.TestLoader().loadTestsFromTestCase(LambdaTest)
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    runTests()
