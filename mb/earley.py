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


from mb.util import typecheckedequality, EqualitySet, Tree
from copy import deepcopy
from IPython.display import SVG, display


class Symbol(object):
    def __init__(self, symbol):
        self.symbol = symbol

    @typecheckedequality
    def __eq__(self, other):
        return self.symbol == other.symbol

    def __add__(self, other):
        if isinstance(other, Symbol):
            return SymbolList([self, other])
        else:
            other.add(self)
            return other

    def __rshift__(self, other):
        if isinstance(other, Symbol):
            return Rule(self, [other])
        else:
            return Rule(self, other.symbols)

    def __str__(self):
        return self.symbol


class SymbolList(object):
    def __init__(self, symbols):
        self.symbols = symbols

    def add(self, symbol):
        self.symbols.append(symbol)

    def __add__(self, other):
        if isinstance(other, Symbol):
            self.add(other)
            return self
        else:
            return SymbolList(self.symbols + other.symbols)


class T(Symbol):
    def is_terminal(self):
        return True


class NT(Symbol):
    def is_terminal(self):
        return False


class StartSymbol(Symbol):
    pass


start_symbol = StartSymbol('Gamma')


class Rule(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def get_rhs(self):
        return self.rhs

    def get_lhs(self):
        return self.lhs

    def is_unary_production(self):
        return len(self.rhs) == 1

    def __str__(self):
        return str(self.lhs) + ' -> ' + ' '.join(str(r) for r in self.rhs)


class Grammar(object):
    def __init__(self, terminals, non_terminals, initial_symbols, rules):
        self.terminals = terminals
        self.non_terminals = non_terminals
        self.initial_symbols = initial_symbols
        self.rules = rules
        self.initial_rules = filter(lambda r: r.lhs in self.initial_symbols, self.rules)

    def get_initial_symbols(self):
        return self.initial_symbols

    def get_rules_for(self, cat):
        return filter(lambda r: r.lhs == cat, self.rules)


class State(object):
    def __init__(self, rule, current_position, origin_position, tree):
        self.rule = rule
        self.current_position = current_position
        self.origin_position = origin_position
        self.tree = tree

    def complete(self):
        return self.current_position >= len(self.rule.get_rhs())

    def next_cat(self):
        if self.current_position < len(self.rule.get_rhs()):
            return self.rule.get_rhs()[self.current_position]
        else:
            return None

    def advance_position(self):
        self.current_position += 1

    def __str__(self):
        return str(self.rule) + ', ' + str(self.current_position) + ', ' + str(self.origin_position)

    @typecheckedequality
    def __eq__(self, other):
        return self.rule == other.rule and self.current_position == other.current_position and self.origin_position == other.origin_position

    def add_child(self, child):
        self.tree.children.append(child)


class EarlyParser(object):
    def __init__(self, grammar):
        self.grammar = grammar
        self.chart = None
        self.reset_chart()

    def reset_chart(self, n_tokens=0):
        self.chart = [EqualitySet() for i in range(n_tokens + 1)]

    def add_to_set(self, state, i):
        self.chart[i].add(state)

    def parse(self, sentence):
        return self._parse(sentence.split())

    def _parse(self, words):
        self.reset_chart(len(words))
        words = [None] + words + [None]
        for s in self.grammar.get_initial_symbols():
            self.add_to_set(State(Rule(start_symbol, [s]), 0, 0, Tree(start_symbol)), 0)
        for i in range(len(self.chart)):
            for s in self.chart[i]:
                if not s.complete():
                    if s.next_cat().is_terminal():
                        self.scanner(s, i, words[i + 1])
                    else:
                        self.predictor(s, i)
                else:
                    self.completer(s, i)
        res = list()
        for s in self.chart[-1]:
            if s.rule.get_lhs() == start_symbol:
                res.append(s.tree.children[0])
        return ParseResults(res)

    def predictor(self, state, i):
        b = state.next_cat()
        for r in self.grammar.get_rules_for(b):
            self.add_to_set(State(r, 0, i, Tree(b)), i)

    def scanner(self, state, i, w):
        if state.next_cat().symbol == w:
            s_prime = deepcopy(state)  # need deepcopy to avoid sharing of subtrees, not very space efficient
            s_prime.advance_position()
            s_prime.add_child(Tree(state.next_cat()))
            self.add_to_set(s_prime, i + 1)

    def completer(self, state, i):
        b = state.rule.get_lhs()
        j = state.origin_position
        for s in [s for s in self.chart[j] if s.next_cat() == b]:
            s_prime = deepcopy(s)  # need deepcopy to avoid sharing of subtrees, not very space efficient
            s_prime.advance_position()
            s_prime.add_child(state.tree)
            self.add_to_set(s_prime, i)


class ParseResults(object):
    def __init__(self, trees):
        self.trees = trees

    def __str__(self):
        return '\n'.join(t.pprint('', True) for t in self.trees)

    def show(self):
        for t in self.trees:
            t.display()
