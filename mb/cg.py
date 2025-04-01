# coding=utf-8

# The MIT License (MIT)
# Copyright (c) 2018 Gianluca Giorgolo
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

import string
from copy import copy
import mb.html as html 
from mb.util import typecheckedequality, find, EqualitySet
from mb.lambda_calc import Term, Unit, FirstProjection, SecondProjection
from IPython.display import HTML, display
from functools import reduce

# data types

class Type:
    pass



class AtomicType(Type):
    def __init__(self, typ):
        self.typ = typ

    def __eq__(self, other):
        return isinstance(other, UniversalType) or (isinstance(other, AtomicType) and self.typ == other.typ)

    def __str__(self):
        return self.typ

    def __hash__(self):
        return hash(self.typ)

class ArrowType(Type):
    """ A -> B """
    def __init__(self, argument, result):
        self.argument = argument
        self.result = result

    def __eq__(self, other):
        return isinstance(other, UniversalType) or (
            isinstance(other, ArrowType) and self.argument == other.argument and self.result == other.result)

    def __str__(self):
        return '(' + str(self.argument) + ' -> ' + str(self.result) + ')'

    def __hash__(self):
        return hash((self.argument, self.result, 'arrow_type'))

class TensorType(Type):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __eq__(self, other):
        return isinstance(other, UniversalType) or (
            isinstance(other, TensorType) and self.left == other.left and self.right == other.right)

    def __str__(self):
        return '(' + str(self.left) + ' * ' + str(self.right) + ')'

    def __hash__(self):
        return hash((self.left, self.right, 'tensor-type'))

class MonadType(Type):
    def __init__(self, inner, modality):
        self.inner = inner
        self.modality = modality

    def __eq__(self, other):
        return isinstance(other, UniversalType) or (
            isinstance(other, MonadType) and self.modality == other.modality and self.inner == other.inner)

    def __str__(self):
        return '<' + self.modality + '>' + str(self.inner)

    def __hash__(self):
        return hash((self.inner, self.modality, 'monad-type'))

class UniversalType(Type):
    def __init__(self):
        pass

    def __eq__(self, other):
        return isinstance(other, Type)

    def __str__(self):
        return 'ANYTYPE'

    def __hash__(self):
        return hash('universal-type')

# just for convenience
universal_type = UniversalType()


class Formula(object):
    def set_linear(self, flag):
        pass

    def to_html(self):
        pass

    def __mul__(self, other):
        return Tensor(self, other)

    def __lshift__(self, other):
        return RightImplication(other, self)

    def __rshift__(self, other):
        return LeftImplication(self, other)

    def show(self):
        display(HTML(str(self.to_html())))

    formula_counter = 0

    @classmethod
    def get_unique_id(cls):
        Formula.formula_counter += 1
        return Formula.formula_counter


class Atom(Formula):
    def __init__(self, symbol, typ, linear=True):
        self.symbol = symbol
        self.typ = typ
        self.linear = linear

    @typecheckedequality
    def __eq__(self, other):
        return self.symbol == other.symbol and self.typ == other.typ

    def __str__(self):
        return self.symbol

    def set_linear(self, flag):
        self.linear = flag

    def to_html(self):
        return html.text(self.symbol)

    def assign_unique_id(self):
        new = copy(self)
        new.unique_id = Formula.get_unique_id()
        return new


class Variable(Formula):
    def __init__(self, identifier, typ, linear=True):
        self.identifier = identifier
        self.typ = typ
        self.linear = linear

    @typecheckedequality
    def __eq__(self, other):
        return self.identifier == self.identifier and self.typ == other.typ

    def __str__(self):
        return 'var_' + str(self.identifier)

    def set_linear(self, flag):
        self.linear = flag

    def to_html(self):
        return html.text(self.identifier)

    def assign_unique_id(self):
        new = copy(self)
        new.unique_id = Formula.get_unique_id()
        return new


class LeftImplication(Formula):
    def __init__(self, antecedent, consequence, linear=True):
        self.antecedent = antecedent
        self.consequence = consequence
        self.typ = ArrowType(self.antecedent.typ, self.consequence.typ)
        self.set_linear(linear)

    @typecheckedequality
    def __eq__(self, other):
        return self.antecedent == other.antecedent and self.consequence == other.consequence and self.typ == other.typ

    def __str__(self):
        return '(' + str(self.antecedent) + ' \\ ' + str(self.consequence) + ')'

    def set_linear(self, flag):
        self.linear = flag
        self.antecedent.set_linear(flag)
        self.consequence.set_linear(flag)

    def to_html(self):
        ty_ant = type(self.antecedent)
        ty_cons = type(self.consequence)
        if ty_ant == Atom or ty_ant == Variable or ty_ant == Monad:
            ant_wrapper = lambda x : x
        else:
            ant_wrapper = lambda x : html.text('(') & x & html.text(')')
        if ty_cons == Atom or ty_cons == Variable or ty_cons == Monad:
            cons_wrapper = lambda x : x
        else:
            cons_wrapper = lambda x : html.text('(') & x & html.text(')')
        return ant_wrapper(self.antecedent.to_html()) & html.raw_html(' &bsol; ') & cons_wrapper(self.consequence.to_html())    

    def assign_unique_id(self):
        new = LeftImplication(self.antecedent.assign_unique_id(), self.consequence.assign_unique_id())
        new.unique_id = Formula.get_unique_id()
        return new

class RightImplication(Formula):
    def __init__(self, antecedent, consequence, linear=True):
        self.antecedent = antecedent
        self.consequence = consequence
        self.typ = ArrowType(self.antecedent.typ, self.consequence.typ)
        self.set_linear(linear)

    @typecheckedequality
    def __eq__(self, other):
        return self.antecedent == other.antecedent and self.consequence == other.consequence and self.typ == other.typ

    def __str__(self):
        return '(' + str(self.consequence) + ' / ' + str(self.antecedent) + ')'

    def set_linear(self, flag):
        self.linear = flag
        self.antecedent.set_linear(flag)
        self.consequence.set_linear(flag)

    def to_html(self):
        ty_ant = type(self.antecedent)
        ty_cons = type(self.consequence)
        if ty_ant == Atom or ty_ant == Variable or ty_ant == Monad:
            ant_wrapper = lambda x : x
        else:
            ant_wrapper = lambda x : html.text('(') & x & html.text(')')
        if ty_cons == Atom or ty_cons == Variable or ty_cons == Monad:
            cons_wrapper = lambda x : x
        else:
            cons_wrapper = lambda x : html.text('(') & x & html.text(')')
        return cons_wrapper(self.consequence.to_html()) & html.raw_html(' &sol; ') & ant_wrapper(self.antecedent.to_html())    

    def assign_unique_id(self):
        new = RightImplication(self.antecedent.assign_unique_id(), self.consequence.assign_unique_id())
        new.unique_id = Formula.get_unique_id()
        return new



class Monad(Formula):
    def __init__(self, body, modality, linear=True):
        self.body = body
        self.typ = MonadType(self.body.typ, modality)
        self.modality = modality
        self.set_linear(linear)

    @typecheckedequality
    def __eq__(self, other):
        return self.body == other.body and self.typ == other.typ

    def __str__(self):
        return '<' + str(self.modality) + '>' + str(self.body)

    def set_linear(self, flag):
        self.linear = flag
        self.body.set_linear(flag)

    def to_html(self):
        ty_body = type(self.body)
        if ty_body == Atom or ty_body == Variable:
            return html.raw_html('&loz;') & self.body.to_html()
        else:
            return html.raw_html('&loz;') & html.text('(') & self.body.to_html() & html.text(')')

    def assign_unique_id(self):
        new = Monad(self.body.assign_unique_id(), self.modality)
        new.unique_id = Formula.get_unique_id()
        return new


class Tensor(Formula):
    def __init__(self, left, right, linear=True):
        self.left = left
        self.right = right
        self.typ = TensorType(self.left.typ, self.right.typ)
        self.set_linear(linear)

    @typecheckedequality
    def __eq__(self, other):
        return self.typ == other.typ and self.left == other.left and self.right == other.right

    def __str__(self):
        return '(' + str(self.left) + ' ⊗ ' + str(self.right) + ')'

    def set_linear(self, flag):
        self.linear = flag
        self.left.set_linear(flag)
        self.right.set_linear(flag)

    def to_html(self):
        ty_left = type(self.left)
        if ty_left == Atom or ty_left == Variable or ty_left == Monad:
            return self.left.to_html() & html.raw_html(' &otimes; ') & self.right.to_html()
        else:
            return html.text('(') & self.left.to_html() & html.text(')') & html.raw_html(
                ' &otimes; ') & self.right.to_html()

    def assign_unique_id(self):
        new = self.left.assign_unique_id() * self.right.assign_unique_id()
        new.unique_id = Formula.get_unique_id()
        return new


def move_unlimited_resources(sequent):
    """Moves the unlimited resources in the hypotheses space and de-linearizes them"""
    for f in sequent.unlimitedResources:
        f.set_linear(False)
    return Sequent(sequent.hypotheses + sequent.unlimitedResources, sequent.theorem, [])


class Sequent(object):
    def __init__(self, hypotheses, theorem, unlimitedResources=None):
        self.hypotheses = hypotheses
        self.theorem = theorem
        if unlimitedResources is None:
            self.unlimitedResources = list()
        else:
            self.unlimitedResources = unlimitedResources
        self.left_focus = -1
        self.new_subformulae = list()

    def __str__(self):
        return ', '.join(map(str, self.hypotheses)) + ' ⊢ ' + str(self.theorem)

    def to_html(self):
        hyps = [h.to_html() for h in self.hypotheses]
        if self.left_focus >= 0 and self.left_focus < len(self.hypotheses):
            hyps[self.left_focus] = (html.span() % {'style': 'color: red'}) << hyps[self.left_focus]
            t = self.theorem.to_html()
        else:
            t = (html.span() % {'style': 'color: red'}) << self.theorem.to_html()
        left = html.text(', ').intersperse(hyps)
        return left & html.raw_html(' &#8866; ') & t

    def show(self):
        display(HTML(str(self.to_html())))

class Proof(object):
    def __init__(self, sequent, label, children=None):
        self.sequent = sequent
        self.label = label
        if children is None:
            children = list()
        self.children = children

    def to_html(self):
        colspan = str(max(1, len(self.children)))
        if len(self.children) > 0:
            row1 = self.tree_tr() << reduce(lambda x, y: x & y, [self.tree_td() << c.to_html() for c in self.children])
        else:
            row1 = self.tree_tr()
        row2 = self.tree_tr() << (
            ((self.tree_td() << self.tree_hr()) % {'colspan': colspan}) & (self.tree_td() << label_to_html(self.label)))
        row3 = self.tree_tr() << ((self.tree_td() << self.sequent.to_html()) % {'colspan': colspan})
        t = self.tree_table()
        return t << (row1 & row2 & row3)

    def tree_td(self):
        return html.td() % {'class': 'proof',
                            'style': 'white-space: nowrap; text-align : center; vertical-align : bottom; padding-left : 5px; padding-right : 5px; font-family : serif; font-style : italic; border-style : hidden; border-collapse : collapse; border-width : 0px;'}

    def tree_table(self):
        return html.table() % {'class': 'proof',
                               'style': 'border-style : hidden; border-collapse : collapse; border-width : 0px;'}

    def tree_tr(self):
        return html.tr() % {'class': 'proof',
                            'style': 'border-style : hidden; border-collapse : collapse; border-width : 0px; padding-top: 1px; padding-bottom: 1px'}

    def tree_hr(self):
        return html.hr() % {'style': 'border-top: 1px solid #000000; margin-top: 0px; margin-bottom: 0px'}

    def __str__(self):
        return self.pprint('', True)

    def show(self):
        display(HTML(str(self.to_html())))

    def pprint(self, prefix, is_tail):
        res = prefix
        if is_tail:
            res += '└── '
            child_pref = '    '
        else:
            res += '├── '
            child_pref = '|   '
        res += str(self.sequent) + '    ' + self.label + '\n'
        for i in range(len(self.children) - 1):
            res += self.children[i].pprint(prefix + child_pref, False)
        if len(self.children) > 0:
            res += self.children[-1].pprint(prefix + child_pref, True)
        return res


def label_to_html(label):
    if label == axiomLabel:
        return html.raw_html('Id')
    elif label == tensorLeftLabel:
        return html.raw_html('&otimes; L')
    elif label == leftImplicationLeftLabel:
        return html.raw_html('&bsol; L')
    elif label == rightImplicationLeftLabel:
        return html.raw_html('&sol; L')
    elif label == monadLeftLabel:
        return html.raw_html('&loz; L')
    elif label == tensorRightLabel:
        return html.raw_html('&otimes; R')
    elif label == leftImplicationRightLabel:
        return html.raw_html('&bsol; R')
    elif label == rightImplicationRightLabel:
        return html.raw_html('&sol; R')
    elif label == monadRightLabel:
        return html.raw_html('&loz; R')


##### Rules

# Left rules

def remove_index(lst, i):
    del lst[i]
    return lst


def set_left_focus(sequent, i):
    sequent.left_focus = i
    return sequent


axiomLabel = "axiom"


def axiom(sequent, bindings, i):
    a = sequent.hypotheses[i]
    if all(sequent.hypotheses[j].linear == False for j in remove_index(list(range(len(sequent.hypotheses))), i)) \
            and len(sequent.unlimitedResources) == 0 and unify(a, sequent.theorem, bindings):
        sequent.left_focus = i
        return [Proof(sequent, axiomLabel)]
    else:
        return []

tensorLeftLabel = "tensor_left"

# gamma, a, b, delta |- c
# ------------------------
# gamma, a * b, delta |- c
def tensor_left_rule(sequent, bindings, i):
    a = sequent.hypotheses[i]
    if type(a) == Tensor:
        child_seq = Sequent(sequent.hypotheses[:i] + [a.left, a.right] + sequent.hypotheses[i + 1:], sequent.theorem,
                            sequent.unlimitedResources)
        child_seq.new_subformulae = [i, i + 1]
        return [Proof(set_left_focus(sequent, i), tensorLeftLabel, [c]) for c in
                raw_prove(child_seq, bindings)]
    else:
        return []


monadLeftLabel = "monad_left"

#   gamma, A, delta |- b
# -------------------------
# gamma, <>A, delta |- <> b
def monad_left_rule(sequent, bindings, i):
    a = sequent.hypotheses[i]
    if type(a) == Monad and type(sequent.theorem) == Monad and a.modality == sequent.theorem.modality:
        child_seq = Sequent(sequent.hypotheses[:i] + [a.body] + sequent.hypotheses[i + 1:], sequent.theorem,
                            sequent.unlimitedResources)
        child_seq.new_subformulae = [i]
        return [Proof(set_left_focus(sequent, i), monadLeftLabel, [c]) for c in
                raw_prove(child_seq, bindings)]
    else:
        return []


leftImplicationLeftLabel = "left_implication_left"

#  delta |- a     gamma, b, theta |- c
#  ----------------------------------- \ L
#   gamma, delta, a \ b, theta |- c
def left_implication_left_rule(sequent, bindings, i):
    a = sequent.hypotheses[i]
    if type(a) == LeftImplication:
        children = list()
        theta = sequent.hypotheses[i + 1:] 
        for (gamma, delta) in split(sequent.hypotheses[:i]):
            left_child_seq = Sequent(delta, a.antecedent, sequent.unlimitedResources)
            right_child_seq = Sequent(gamma + [a.consequence] + theta, sequent.theorem, sequent.unlimitedResources)
            right_child_seq.new_subformulae = [len(gamma)]
            children.extend((l, r) for l in raw_prove(left_child_seq, bindings)
                            for r in
                            raw_prove(right_child_seq, bindings))
        return [Proof(set_left_focus(sequent, i), leftImplicationLeftLabel, [l, r]) for (l, r) in children]

    else:
        return []

rightImplicationLeftLabel = "right_implication_left"

#  delta |- a     gamma, b, theta |- c
#  ----------------------------------- / L
#   gamma, b / a, delta, theta |- c
def right_implication_left_rule(sequent, bindings, i):
    a = sequent.hypotheses[i]
    if type(a) == RightImplication:
        children = list()
        gamma = sequent.hypotheses[:i] 
        for (delta, theta) in split(sequent.hypotheses[i + 1:]):
            left_child_seq = Sequent(delta, a.antecedent, sequent.unlimitedResources)
            right_child_seq = Sequent(gamma + [a.consequence] + theta, sequent.theorem, sequent.unlimitedResources)
            right_child_seq.new_subformulae = [len(gamma)]
            children.extend((l, r) for l in raw_prove(left_child_seq, bindings)
                            for r in
                            raw_prove(right_child_seq, bindings))
        return [Proof(set_left_focus(sequent, i), rightImplicationLeftLabel, [l, r]) for (l, r) in children]
    else:
        return []


# Right rules
monadRightLabel = "monad_right"

# gamma |- a
# ------------
# gamma |- <>a
def monad_right_rule(sequent, bindings):
    if type(sequent.theorem) == Monad:
        return [Proof(sequent, monadRightLabel, [c]) for c in
                raw_prove(Sequent(sequent.hypotheses, sequent.theorem.body, sequent.unlimitedResources), bindings)]
    else:
        return []


leftImplicationRightLabel = "left_implication_right"

# a, gamma |- b
# --------------
# gamma |- a \ b
def left_implication_right_rule(sequent, bindings):
    if type(sequent.theorem) == LeftImplication:
        s = Sequent([sequent.theorem.antecedent] + sequent.hypotheses, sequent.theorem.consequence,
                    sequent.unlimitedResources)
        s.new_subformulae = [0]
        return [Proof(sequent, leftImplicationRightLabel, [c]) for c in raw_prove(s, bindings)]
    else:
        return []

rightImplicationRightLabel = "right_implication_right"

# gamma, a |- b
# --------------
# gamma |- b / a
def right_implication_right_rule(sequent, bindings):
    if type(sequent.theorem) == RightImplication:
        s = Sequent(sequent.hypotheses + [sequent.theorem.antecedent],  sequent.theorem.consequence,
                    sequent.unlimitedResources)
        s.new_subformulae = [len(sequent.hypotheses)]
        return [Proof(sequent, leftImplicationRightLabel, [c]) for c in raw_prove(s, bindings)]
    else:
        return []


tensorRightLabel = "tensor_right"

# gamma |- a   delta |- b
# -----------------------
#  gamma, delta |- a * b
def tensor_right_rule(sequent, bindings):
    if type(sequent.theorem) == Tensor:
        children = list()
        for (gamma, delta) in split(sequent.hypotheses):
            children.extend((l, r) for l in raw_prove(Sequent(gamma, sequent.theorem.left, sequent.unlimitedResources))
                            for r in raw_prove(Sequent(delta, sequent.theorem.right, sequent.unlimitedResources)))
        return [Proof(sequent, tensorRightLabel, [l, r]) for (l, r) in children]
    else:
        return []


# Structural rules

def weakening(sequent, bindings):
    l = len(sequent.unlimitedResources)
    if l == 0:
        return []
    else:
        children = list()
        for i in range(l):
            left, _, right = split_at(sequent.unlimitedResources, i)
            children.extend(raw_prove(Sequent(sequent.hypotheses, sequent.theorem, left + right)))
        return [Proof(sequent, flipLabel, [c]) for c in children]


# Contraction is actually not implemented because we limit the number of allowed contractions...
def contraction(sequent, bindings):
    return []


flipLabel = "flip"


# instead we use flip to move resources from the "unlimited" stash to the normal linear space
def flip(sequent, bindings):
    l = len(sequent.unlimitedResources)
    if l == 0:
        return []
    else:
        children = list()
        for i in range(l):
            left, f, right = split_at(sequent.unlimitedResources, i)
            children.extend(raw_prove(Sequent(sequent.hypotheses + [f], sequent.theorem, left + right)))
        return [Proof(sequent, flipLabel, [c]) for c in children]


def split_at(lst, i):
    return (lst[:i], lst[i], lst[i + 1:])


leftRules = [axiom, tensor_left_rule, monad_left_rule, left_implication_left_rule, right_implication_left_rule]
rightRules = [monad_right_rule, left_implication_right_rule, right_implication_right_rule, tensor_right_rule]
structuralRules = []


# Main engine

def prove(sequent, associations=None):
    return ProofResults(
        raw_prove(Sequent([h.assign_unique_id() for h in sequent.hypotheses], sequent.theorem.assign_unique_id()), None), associations)

def prove_sentence(sentence, sentence_category, lexicon):
    """Tries to prove that a sentence (a string of space separated words)
       is of a given category given a lexicon that defines the words in the sentence.
       Raises an exception if not all words are defined in the lexicon
    """
    words = sentence.split()
    if not all(lexicon.word_in_lexicon(w) for w in words):
        raise Exception('Not all words are defined in the lexicon')
    hypotheses = [lexicon.get_syntactic_category(w) for w in words]
    associations = [lexicon.get_meaning(w) for w in words]
    sequent = Sequent(hypotheses, sentence_category)
    return prove(sequent, associations)
    
def raw_prove(sequent, bindings=None):
    if bindings is None:
        bindings = dict()
    left = join(
        [r(copy(sequent), copy(bindings), i)
         for i in range(len(sequent.hypotheses))
         for r in leftRules])
    right = join([r(sequent, copy(bindings)) for r in rightRules])
    struct = join([r(sequent, copy(bindings)) for r in structuralRules])
    return left + right + struct


# util

def unify(f, g, bindings):
    """Attempts to unify f and g"""
    if type(f) == Variable:
        if type(g) == Variable:  # both f and g are variables
            if f.typ == g.typ:
                binding_f = bindings.get(f)
                binding_g = bindings.get(g)
                if binding_f is None:  # f unbound
                    bindings[f] = g
                    return True
                else:
                    if binding_g is None:  # g unbound
                        bindings[g] = f
                        return True
                    else:
                        return binding_f == binding_g
            else:  # they have different types they can't be unified
                return False
        else:  # g is not a variable
            if f.typ == g.typ:
                binding_f = bindings.get(f)
                if binding_f is None:  # f unbound
                    bindings[f] = g
                    return True
                else:  # f is bound
                    return binding_f == g
            else:  # they have different types
                return False
    elif type(g) == Variable:  # g is a variable but f not
        return unify(g, f, bindings)
    else:  # both are not variables
        return f == g


def join(list_of_lists):
    """Joins a list of lists"""
    tmp = list()
    for l in list_of_lists:
        tmp.extend(l)
    return tmp


def delete(a, ls):
    """Deletes an element from a list, returning a copy of the list"""
    tmp = copy(ls)
    tmp.remove(a)
    return tmp


def split(lst):
    """Splits a list in the list of all prefixes and suffixes"""
    ln = len(lst)
    res = list()
    for i in range(ln+1):
        res.append((lst[:i], lst[i:]))
    return res


class ProofResults(object):
    def __init__(self, proofs, associations):
        self.proofs = proofs
        self.grouped = False
        self.witness_proofs = proofs
        self.curry_howarded = False
        self.associations = associations
        self.variable_sanitized = False

    def create_proof_terms(self):
        def aux(p, subs):
            s = p.sequent
            for i in range(len(s.hypotheses)):
                for k in subs.keys():
                    s.hypotheses[i][1] = s.hypotheses[i][1].substitute(subs[k], k)
            for k in subs.keys():
                s.theorem[1] = s.theorem[1].substitute(subs[k], k)
            for i in range(len(p.children)):
                p.children[i] = aux(p.children[i], subs)
            return p

        if not self.curry_howarded:
            for i in range(len(self.proofs)):
                Term.reset_counter()
                self.proofs[i] = curry_howard(self.proofs[i])
                if not self.associations is None:
                    subs = dict()
                    for j in range(len(self.proofs[i].sequent.hypotheses)):
                        h = self.proofs[i].sequent.hypotheses[j]
                        # subs[h[1]] = self.associations[j][0]
                        subs[h[1]] = self.associations[j]
                    self.proofs[i] = aux(self.proofs[i], subs)
                self.proofs[i].sequent.theorem[1] = self.proofs[i].sequent.theorem[1].reduce()

            self.curry_howarded = True

    def __iter__(self):
        if self.grouped:
            return iter(self.witness_proofs)
        else:
            return iter(self.proofs)

    def group_proofs(self):
        self.sanitize_vars()
        if not self.curry_howarded:
            self.create_proof_terms()
        res = list()
        for p in self.proofs:
            f = find(res, lambda q: p.sequent.theorem[1].alpha_equivalent(q.sequent.theorem[1]))
            if f.is_nothing():
                res.append(p)
        self.witness_proofs = res
        self.grouped = True

    def __len__(self):
        if self.grouped:
            return len(self.witness_proofs)
        else:
            return len(self.proofs)

    def __getitem__(self, idx):
        if self.grouped:
            return self.witness_proofs[idx]
        else:
            return self.proofs[idx]

    def sanitize_vars(self):
        def aux_coll_vars(t):
            vs = EqualitySet.empty_set()
            vs = vs.union(t.sequent.theorem[1].collect_vars())
            for c in t.children:
                vs = vs.union(aux_coll_vars(c))
            vs = EqualitySet.filter(vs, lambda v : v._internal)
            return vs

        def aux_ren_vars(t, rename_map):
            for h in t.sequent.hypotheses:
                h[1].rename_vars(rename_map)
            for c in t.children:
                aux_ren_vars(c, rename_map)

        if not self.variable_sanitized:
            if not self.curry_howarded:
                self.create_proof_terms()
            for p in self:
                vs = aux_coll_vars(p).to_list()
                rename_map = dict()
                l = min(len(vs), len(string.ascii_lowercase))
                for i in range(l):
                    rename_map[vs[i].identifier] = string.ascii_lowercase[i]
                aux_ren_vars(p, rename_map)
            self.variable_sanitized = True

    def show(self, show_only_witness=False):
        if show_only_witness:
            if not self.grouped:
                self.group_proofs()
            for p in self.witness_proofs:
                p.show()
        else:
            for p in self.proofs:
                p.show()


# Curry-Howard

class CurryHowardSequent(Sequent):
    def __init__(self, hypotheses, theorem):
        super(CurryHowardSequent, self).__init__(hypotheses, theorem)

    def __str__(self):
        def aux(p):
            return str(p[1]) + ' : ' + str(p[0])

        return ', '.join(map(aux, self.hypotheses)) + ' ⊢ ' + aux(self.theorem)

    def to_html(self):
        hyps = [h[1].to_html() + html.text(' : ') + h[0].to_html() for h in self.hypotheses]
        if self.left_focus >= 0 and self.left_focus < len(self.hypotheses):
            hyps[self.left_focus] = (html.span() % {'style': 'color: red'}) << hyps[self.left_focus]
            t = self.theorem[1].to_html() + html.text(' : ') + self.theorem[0].to_html()
        else:
            t = (html.span() % {'style': 'color: red'}) << (self.theorem[1].to_html() + html.text(' : ') + self.theorem[0].to_html())
        left = html.text(', ').intersperse(hyps)
        return left & html.raw_html(' &#8866; ') & t


def curry_howard(proof):
    def aux(formulae, decorated_formulae):
        """Given that formulae are sometimes shuffled around in the children
           we need a way to get the lambda terms associated with them in an order-independent way.
           This function does that"""
        formulae = copy(formulae)
        for i in range(len(formulae)):
            f = formulae[i]
            for df in decorated_formulae:
                g = df[0]
                t = df[1]
                if f.unique_id == g.unique_id:
                    formulae[i] = [f, t]
        return formulae

    if proof.label == axiomLabel:
        x = Term.fresh_variable()
        hyps = copy(proof.sequent.hypotheses)
        for i in range(len(hyps)):
            hyps[i] = [hyps[i], Term.fresh_variable()]
        hyps[proof.sequent.left_focus] = [hyps[proof.sequent.left_focus][0], x]
        s = CurryHowardSequent(hyps, [proof.sequent.theorem, x])
        s.left_focus = proof.sequent.left_focus
        s.new_subformulae = proof.sequent.new_subformulae
        return Proof(s, axiomLabel)  # DONE
    elif proof.label == tensorLeftLabel:
        c_prime = curry_howard(proof.children[0])
        x = c_prime.sequent.hypotheses[c_prime.sequent.new_subformulae[0]][1]
        y = c_prime.sequent.hypotheses[c_prime.sequent.new_subformulae[1]][1]
        t = c_prime.sequent.theorem[1]
        u = Term.fresh_variable()
        hyps = aux(proof.sequent.hypotheses, c_prime.sequent.hypotheses)
        hyps[proof.sequent.left_focus] = [proof.sequent.hypotheses[proof.sequent.left_focus], u]
        c = proof.sequent.theorem
        new_t = t.substitute(FirstProjection(u), x).substitute(SecondProjection(u), y)
        s = CurryHowardSequent(hyps, [c, new_t])
        s.left_focus = proof.sequent.left_focus
        s.new_subformulae = proof.sequent.new_subformulae
        return Proof(s, tensorLeftLabel, [c_prime])  # DONE
    elif proof.label == leftImplicationLeftLabel:
        l_prime = curry_howard(proof.children[0])
        r_prime = curry_howard(proof.children[1])
        t = l_prime.sequent.theorem[1]
        x = r_prime.sequent.hypotheses[r_prime.sequent.new_subformulae[0]][1]
        u = r_prime.sequent.theorem[1]
        y = Term.fresh_variable()
        hyps = aux(proof.sequent.hypotheses, l_prime.sequent.hypotheses + r_prime.sequent.hypotheses)
        hyps[proof.sequent.left_focus] = [proof.sequent.hypotheses[proof.sequent.left_focus], y]
        s = CurryHowardSequent(hyps, [proof.sequent.theorem, u.substitute(y(t), x)])
        s.left_focus = proof.sequent.left_focus
        s.new_subformulae = proof.sequent.new_subformulae
        return Proof(s, leftImplicationLeftLabel, [l_prime, r_prime])  # DONE
    elif proof.label == rightImplicationLeftLabel:
        l_prime = curry_howard(proof.children[0])
        r_prime = curry_howard(proof.children[1])
        t = l_prime.sequent.theorem[1]
        x = r_prime.sequent.hypotheses[r_prime.sequent.new_subformulae[0]][1]
        u = r_prime.sequent.theorem[1]
        y = Term.fresh_variable()
        hyps = aux(proof.sequent.hypotheses, l_prime.sequent.hypotheses + r_prime.sequent.hypotheses)
        hyps[proof.sequent.left_focus] = [proof.sequent.hypotheses[proof.sequent.left_focus], y]
        s = CurryHowardSequent(hyps, [proof.sequent.theorem, u.substitute(y(t), x)])
        s.left_focus = proof.sequent.left_focus
        s.new_subformulae = proof.sequent.new_subformulae
        return Proof(s, rightImplicationLeftLabel, [l_prime, r_prime])  # DONE
    elif proof.label == monadLeftLabel:
        c_prime = curry_howard(proof.children[0])
        x = c_prime.sequent.hypotheses[c_prime.sequent.new_subformulae[0]][1]
        t = c_prime.sequent.theorem[1]
        y = Term.fresh_variable()
        hyps = aux(proof.sequent.hypotheses, c_prime.sequent.hypotheses)
        hyps[proof.sequent.left_focus] = [proof.sequent.hypotheses[proof.sequent.left_focus], y]
        s = CurryHowardSequent(hyps, [proof.sequent.theorem, y ** (x ^ t)])
        s.left_focus = proof.sequent.left_focus
        s.new_subformulae = proof.sequent.new_subformulae
        return Proof(s, monadLeftLabel, [c_prime])  # DONE
    elif proof.label == tensorRightLabel:
        l_prime = curry_howard(proof.children[0])
        r_prime = curry_howard(proof.children[1])
        t = l_prime.sequent.theorem[1]
        u = r_prime.sequent.theorem[1]
        hyps = aux(proof.sequent.hypotheses, l_prime.sequent.hypotheses + r_prime.sequent.hypotheses)
        s = CurryHowardSequent(hyps, [proof.sequent.theorem, t * u])
        s.left_focus = proof.sequent.left_focus
        s.new_subformulae = proof.sequent.new_subformulae
        return Proof(s, tensorRightLabel, [l_prime, r_prime])  # DONE
    elif proof.label == leftImplicationRightLabel:
        c_prime = curry_howard(proof.children[0])
        x = c_prime.sequent.hypotheses[c_prime.sequent.new_subformulae[0]][1]
        t = c_prime.sequent.theorem[1]
        hyps = aux(proof.sequent.hypotheses, c_prime.sequent.hypotheses)
        s = CurryHowardSequent(hyps, [proof.sequent.theorem, x ^ t])
        s.left_focus = proof.sequent.left_focus
        s.new_subformulae = proof.sequent.new_subformulae
        return Proof(s, leftImplicationRightLabel, [c_prime])  # DONE
    elif proof.label == rightImplicationRightLabel:
        c_prime = curry_howard(proof.children[0])
        x = c_prime.sequent.hypotheses[c_prime.sequent.new_subformulae[0]][1]
        t = c_prime.sequent.theorem[1]
        hyps = aux(proof.sequent.hypotheses, c_prime.sequent.hypotheses)
        s = CurryHowardSequent(hyps, [proof.sequent.theorem, x ^ t])
        s.left_focus = proof.sequent.left_focus
        s.new_subformulae = proof.sequent.new_subformulae
        return Proof(s, rightImplicationRightLabel, [c_prime])  # DONE
    elif proof.label == monadRightLabel:
        c_prime = curry_howard(proof.children[0])
        hyps = aux(proof.sequent.hypotheses, c_prime.sequent.hypotheses)
        s = CurryHowardSequent(hyps, [c_prime.sequent.theorem[0], Unit(c_prime.sequent.theorem[1])])
        s.left_focus = proof.sequent.left_focus
        s.new_subformulae = proof.sequent.new_subformulae
        return Proof(s, monadRightLabel, [c_prime])  # DONE

class Lexicon(object):
    """ A simple lexicon object """
    def __init__(self):
        self.lexicon = dict()

    def add_entry(self, word, syntactic_category, meaning):
        """Add a word to the lexicon, if the word was already defined it overwrites the previous definition"""
        self.lexicon[word] = (syntactic_category, meaning)

    def get_syntactic_category(self, word):
        """Returns the syntactic category or None if the word is not in the lexicon"""
        if word in self.lexicon:
            return self.lexicon[word][0]
        else:
            return None

    def get_meaning(self, word):
        """Returns the meaning of a word or None if the word is not in the lexicon"""
        if word in self.lexicon:
            return self.lexicon[word][1]
        else:
            return None

    def word_in_lexicon(self, word):
        """Checks if a word is in the lexicon"""
        return word in self.lexicon

if __name__ == '__main__':
    from mb.lambda_calc import Const, Var, Unit
    n = Atom('n', ArrowType(AtomicType('e'), AtomicType('t')))
    np = Atom('np', AtomicType('e'))
    s = Atom('s', AtomicType('t'))
    lexicon = Lexicon()
    lexicon.add_entry('JLH', np, Const('jlh'))
    lexicon.add_entry('Tennessee', np, (Const('tn')))
    lexicon.add_entry('TBB', np, Const('tbb'))
    lexicon.add_entry('bluesman', n, Const('bluesman'))
    lexicon.add_entry('the', (np < s) > n, Var('x') ^ Var('x'))
    lexicon.add_entry('from', (n < n) > np, Var('x') ^ (Var('P') ^ (Var('y') ^ (Const('AND')(Var('P')(Var('y'))))((Const('from')(Var('y')))(Var('x'))))))
    lexicon.add_entry('appeared_in', (np < s) > np, Var('x') ^ (Var('y') ^ ((Const('appeared_in')(Var('y')))(Var('x')))))
    lexicon.add_entry('COMMA', (np < Monad(np, 'ci')) > (np < s), Var('P') ^ (Var('x') ^ ((Const("write")(Var('P')(Var('x')))) ** (Var('y') ^ (Unit(Var('x')))))))
    proofs = prove_sentence('JLH COMMA the bluesman from Tennessee appeared_in TBB', Monad(s,'ci'), lexicon)
    proofs.group_proofs()
    print(len(proofs))
