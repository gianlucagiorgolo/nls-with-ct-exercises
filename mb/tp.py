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
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __eq__(self, other):
        return isinstance(other, UniversalType) or (
            isinstance(other, ArrowType) and self.left == other.left and self.right == other.right)

    def __str__(self):
        return '(' + str(self.left) + ' -> ' + str(self.right) + ')'

    def __hash__(self):
        return hash((self.left, self.right, 'arrow_type'))


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
        return hash((self.left, self.right, 'tensor_type'))

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
        return hash((self.inner, self.modality, 'monad_type'))

class UniversalType(Type):
    def __init__(self):
        pass

    def __eq__(self, other):
        return isinstance(other, Type)

    def __str__(self):
        return 'ANYTYPE'

    def __hash__(self):
        return hash('universal_type')

# just for convenience
universal_type = UniversalType()


class Formula(object):
    def set_linear(self, flag):
        pass

    def to_html(self):
        pass

    def to_latex(self):
        return ''

    def __mul__(self, other):
        return Tensor(self, other)

    def __pow__(self, other):
        return Implication(self, other)

    def show(self):
        display(HTML(str(self.to_html())))

    def show_latex(self):
        """Returns a latex representation of the formula"""
        return self.to_latex()

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

    def to_latex(self):
        return self.symbol

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

    def to_latex(self):
        return self.identifier

    def assign_unique_id(self):
        new = copy(self)
        new.unique_id = Formula.get_unique_id()
        return new

    def __hash__(self):
        return hash((self.identifier, self.typ, self.linear, 'variable'))

class Implication(Formula):
    def __init__(self, antecedent, consequence, linear=True):
        self.antecedent = antecedent
        self.consequence = consequence
        self.typ = ArrowType(self.antecedent.typ, self.consequence.typ)
        self.set_linear(linear)

    @typecheckedequality
    def __eq__(self, other):
        return self.antecedent == other.antecedent and self.consequence == other.consequence and self.typ == other.typ

    def __str__(self):
        return '(' + str(self.antecedent) + ' ⊸ ' + str(self.consequence) + ')'

    def set_linear(self, flag):
        self.linear = flag
        self.antecedent.set_linear(flag)
        self.consequence.set_linear(flag)

    def to_html(self):
        ty_ant = type(self.antecedent)
        if ty_ant == Atom or ty_ant == Variable or ty_ant == Monad:
            return self.antecedent.to_html() & html.raw_html(' &rarr; ') & self.consequence.to_html()
        else:
            return html.text('(') & self.antecedent.to_html() & html.text(')') & html.raw_html(
                ' &rarr; ') & self.consequence.to_html()

    def to_latex(self):
        ty_ant = type(self.antecedent)
        if ty_ant == Atom or ty_ant == Variable or ty_ant == Monad:
            return self.antecedent.to_latex() + '\\rightarrow ' + self.consequence.to_latex()
        else:
            return '(' + self.antecedent.to_latex() + ')\\rightarrow ' + self.consequence.to_latex()

    def assign_unique_id(self):
        new = Implication(self.antecedent.assign_unique_id(), self.consequence.assign_unique_id())
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

    def to_latex(self):
        ty_body = type(self.body)
        if ty_body == Atom or ty_body == Variable:
            return '\\lozenge^{' + str(self.modality) + '}' + self.body.to_latex()
        else:
            return '\\lozenge^{' + str(self.modality) + '}(' + self.body.to_latex() + ')'

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

    def to_latex(self):
        ty_left = type(self.left)
        if ty_left == Atom or ty_left == Variable or ty_left == Monad:
            return self.left.to_latex() + '\\otimes ' + self.right.to_latex()
        else:
            return '(' + self.left.to_latex() + ')' + '\\otimes ' + self.right.to_latex()

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

    def to_latex(self):
        hyps = [h.to_latex() for h in self.hypotheses]
        return ','.join(hyps) + '\\vdash ' + self.theorem.to_latex()
        
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

    def to_latex(self):
        n_children = len(self.children)
        res_child = ''
        if n_children == 0:
            res_child = '\\AxiomC{}'
        else:
            res_child = ''.join(c.to_latex() for c in self.children)
        rule = '\\UnaryInfC'
        if n_children == 2:
            rule = '\\BinaryInfC'
        return res_child + '\n\\RightLabel{$' + label_to_latex(self.label) + '$}\n' + rule + '{$' + self.sequent.to_latex() + '$}'

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
    elif label == implicationLeftLabel:
        return html.raw_html('&rarr; L')
    elif label == monadLeftLabel:
        return html.raw_html('&loz; L')
    elif label == tensorRightLabel:
        return html.raw_html('&otimes; R')
    elif label == implicationRightLabel:
        return html.raw_html('&rarr; R')
    elif label == monadRightLabel:
        return html.raw_html('&loz; R')

def label_to_latex(label):
    if label == axiomLabel:
        return 'Id'
    elif label == tensorLeftLabel:
        return '\\otimes L'
    elif label == implicationLeftLabel:
        return '\\rightarrow L'
    elif label == monadLeftLabel:
        return '\\lozenge L'
    elif label == tensorRightLabel:
        return '\\otimes R'
    elif label == implicationRightLabel:
        return '\\rightarrow R'
    elif label == monadRightLabel:
        return '\\lozenge R'

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


def reinsert_hypothesis(sequent, hypothesis):
    return Sequent([hypothesis] + sequent.hypotheses, sequent.theorem)


tensorLeftLabel = "tensor_left"


def tensor_left_rule(sequent, bindings, i):
    a = sequent.hypotheses[i]
    if type(a) == Tensor:
        child_seq = Sequent([a.left, a.right] + sequent.hypotheses[:i] + sequent.hypotheses[i + 1:], sequent.theorem,
                            sequent.unlimitedResources)
        child_seq.new_subformulae = [0, 1]
        return [Proof(set_left_focus(sequent, i), tensorLeftLabel, [c]) for c in
                raw_prove(child_seq, bindings)]
    else:
        return []


monadLeftLabel = "monad_left"


def monad_left_rule(sequent, bindings, i):
    a = sequent.hypotheses[i]
    if type(a) == Monad and type(sequent.theorem) == Monad and a.modality == sequent.theorem.modality:
        child_seq = Sequent([a.body] + sequent.hypotheses[:i] + sequent.hypotheses[i + 1:], sequent.theorem,
                            sequent.unlimitedResources)
        child_seq.new_subformulae = [0]
        return [Proof(set_left_focus(sequent, i), monadLeftLabel, [c]) for c in
                raw_prove(child_seq, bindings)]
    else:
        return []


implicationLeftLabel = "implication_left"


#  gamma |- a     b, delta |- c
# ------------------------------ -o L
#   gamma, a -o b, delta |- c

def implication_left_rule(sequent, bindings, i):
    a = sequent.hypotheses[i]
    if type(a) == Implication:
        children = list()
        for (gamma, delta) in split(sequent.hypotheses[:i] + sequent.hypotheses[i + 1:]):
            left_child_seq = Sequent(gamma, a.antecedent, sequent.unlimitedResources)
            right_child_seq = Sequent([a.consequence] + delta, sequent.theorem, sequent.unlimitedResources)
            right_child_seq.new_subformulae = [0]
            children.extend((l, r) for l in raw_prove(left_child_seq, bindings)
                            for r in
                            raw_prove(right_child_seq, bindings))
        return [Proof(set_left_focus(sequent, i), implicationLeftLabel, [l, r]) for (l, r) in children]

    else:
        return []


# Right rules
monadRightLabel = "monad_right"


def monad_right_rule(sequent, bindings):
    if type(sequent.theorem) == Monad:
        return [Proof(sequent, monadRightLabel, [c]) for c in
                raw_prove(Sequent(sequent.hypotheses, sequent.theorem.body, sequent.unlimitedResources), bindings)]
    else:
        return []


implicationRightLabel = "implication_right"


def implication_right_rule(sequent, bindings):
    if type(sequent.theorem) == Implication:
        s = Sequent([sequent.theorem.antecedent] + sequent.hypotheses, sequent.theorem.consequence,
                    sequent.unlimitedResources)
        s.new_subformulae = [0]
        return [Proof(sequent, implicationRightLabel, [c]) for c in raw_prove(s, bindings)]
    else:
        return []


tensorRightLabel = "tensor_right"


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


leftRules = [axiom, tensor_left_rule, monad_left_rule, implication_left_rule]
rightRules = [monad_right_rule, implication_right_rule, tensor_right_rule]
structuralRules = [weakening, contraction, flip]


# Main engine

def prove(sequent, associations=None):
    return ProofResults(
        raw_prove(Sequent([h.assign_unique_id() for h in sequent.hypotheses], sequent.theorem), None), associations)


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


## Maybe rewrite this so that its not recursive, now it's just a translation of haskell code
def split(lst):
    """Splits a list in two in all possible ways"""
    ln = len(lst)
    if ln == 0:
        return [([], [])]
    elif ln == 1:
        return [([], lst), (lst, [])]
    else:
        a = lst[0]
        rec = split(lst[1:])
        left = [([a] + l, r) for (l, r) in rec]
        right = [(l, [a] + r) for (l, r) in rec]
        return left + right


class ProofResults(object):
    def __init__(self, proofs, associations):
        self.proofs = proofs
        self.grouped = False
        self.witness_proofs = proofs
        self.curry_howarded = False
        self.associations = associations

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
                self.proofs[i].sequent.theorem[1] = self.proofs[i].sequent.theorem[1].reduce()
                if not self.associations is None:
                    subs = dict()
                    for j in range(len(self.proofs[i].sequent.hypotheses)):
                        h = self.proofs[i].sequent.hypotheses[j]
                        # subs[h[1]] = self.associations[j][0]
                        subs[h[1]] = self.associations[j]
                    self.proofs[i] = aux(self.proofs[i], subs)

            self.curry_howarded = True

    def __iter__(self):
        if self.grouped:
            return iter(self.witness_proofs)
        else:
            return iter(self.proofs)

    def group_proofs(self):
        if not self.curry_howarded:
            self.create_proof_terms()
        self.sanitize_vars()
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

        if not self.curry_howarded:
            self.create_proof_terms()
        for p in self:
            vs = aux_coll_vars(p).to_list()
            rename_map = dict()
            l = min(len(vs), len(string.ascii_lowercase))
            for i in range(l):
                rename_map[vs[i].identifier] = string.ascii_lowercase[i]
            aux_ren_vars(p, rename_map)

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

    def to_latex(self):
        hyps = [h[1].to_latex() + ' : ' + h[0].to_latex() for h in self.hypotheses]
        cons = self.theorem[1].to_latex() + ' : ' + self.theorem[0].to_latex()
        return ','.join(hyps) + '\\vdash ' + cons


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
    elif proof.label == implicationLeftLabel:
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
        return Proof(s, implicationLeftLabel, [l_prime, r_prime])  # DONE
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
    elif proof.label == implicationRightLabel:
        c_prime = curry_howard(proof.children[0])
        x = c_prime.sequent.hypotheses[c_prime.sequent.new_subformulae[0]][1]
        t = c_prime.sequent.theorem[1]
        hyps = aux(proof.sequent.hypotheses, c_prime.sequent.hypotheses)
        s = CurryHowardSequent(hyps, [proof.sequent.theorem, x ^ t])
        s.left_focus = proof.sequent.left_focus
        s.new_subformulae = proof.sequent.new_subformulae
        return Proof(s, implicationRightLabel, [c_prime])  # DONE
    elif proof.label == monadRightLabel:
        c_prime = curry_howard(proof.children[0])
        hyps = aux(proof.sequent.hypotheses, c_prime.sequent.hypotheses)
        s = CurryHowardSequent(hyps, [c_prime.sequent.theorem[0], Unit(c_prime.sequent.theorem[1])])
        s.left_focus = proof.sequent.left_focus
        s.new_subformulae = proof.sequent.new_subformulae
        return Proof(s, monadRightLabel, [c_prime])  # DONE


# test

import unittest


def compress(ls):
    if len(ls) == 0:
        return False
    else:
        return True


class TestProve(unittest.TestCase):
    def atom(self, name):
        return Atom(name, AtomicType(name))

    def setUp(self):
        self.a = self.atom('a')
        self.b = self.atom('b')
        self.c = self.atom('c')
        self.ma = Monad(self.a, "")
        self.mb = Monad(self.b, "")
        self.a_to_a = Implication(self.a, self.a)
        self.a_to_b = Implication(self.a, self.b)
        self.a_times_b = Tensor(self.a, self.b)
        self.x = Variable('x', AtomicType('a'))
        self.y = Variable('y', UniversalType())

    def test_prove(self):
        self.assertTrue(compress(raw_prove(Sequent([self.a], self.a))))
        self.assertFalse(compress(raw_prove(Sequent([], self.a))))
        self.assertFalse(compress(raw_prove(Sequent([self.b], self.a))))
        self.assertTrue(compress(raw_prove(Sequent([self.a], self.ma))))
        self.assertTrue(compress(raw_prove(Sequent([self.ma], self.ma))))
        self.assertFalse(compress(raw_prove(Sequent([self.ma], self.a))))
        self.assertTrue(compress(raw_prove(Sequent([self.ma, self.a_to_b], self.mb))))
        self.assertTrue(compress(raw_prove(Sequent([], self.a_to_a))))
        self.assertTrue(compress(raw_prove(Sequent([self.a, self.a_to_b], self.b))))
        self.assertTrue(compress(raw_prove(Sequent([self.a, self.b], self.a_times_b))))
        self.assertTrue(compress(raw_prove(Sequent([self.b, self.a], self.a_times_b))))
        self.assertTrue(compress(raw_prove(Sequent([self.c, self.a, self.a_to_b], Tensor(self.b, self.c)))))
        self.assertTrue(
            compress(raw_prove(Sequent([self.a, Implication(self.mb, Implication(self.a, self.c)), self.mb], self.c))))
        self.assertTrue(compress(raw_prove(Sequent([self.a, Implication(self.ma, self.b)], self.b))))
        self.assertTrue(compress(raw_prove(Sequent([self.x], self.a))))
        self.assertTrue(compress(raw_prove(Sequent([self.a, Implication(self.x, self.x)], self.a))))
        l_t = Atom('l', AtomicType('t'))
        love = Implication(Atom('m', AtomicType('e')), Implication(Atom('w', AtomicType('e')), l_t))
        everyman = Implication(Implication(Atom('m', AtomicType('e')), Variable('x', AtomicType('t'))),
                               Variable('x', AtomicType('t')))
        awoman = Implication(Implication(Atom('w', AtomicType('e')), Variable('y', AtomicType('t'))),
                             Variable('y', AtomicType('t')))
        quantif_seq = Sequent([everyman, love, awoman], l_t)
        self.assertTrue(compress(raw_prove(quantif_seq)))
        self.assertTrue(compress(raw_prove(Sequent([self.y], self.a))))
        # testing structural rules
        self.assertTrue(compress(raw_prove(Sequent([self.a, self.a_to_b], self.b, [self.c]))))  # weakening works
        self.assertTrue(compress(raw_prove(Sequent([self.a], self.b, [self.a_to_b]))))  # flip works

        # bypassing the cut rule problem
        s = Atom('s', AtomicType('s'))
        f = Atom('f', AtomicType('f'))
        mp_s = Monad(s, 'p')
        mci_s = Monad(s, 'ci')
        mp_f = Monad(f, 'p')
        mci_f = Monad(f, 'ci')
        mpmci_f = Monad(mci_f, 'p')
        mcimp_f = Monad(mp_f, 'ci')
        x = Variable('x', UniversalType())
        mpmci_x = Monad(Monad(x, 'ci'), 'p')
        mcimp_x = Monad(Monad(x, 'p'), 'ci')
        swap1 = Implication(mpmci_x, mcimp_x)
        swap2 = Implication(mcimp_x, mpmci_x)
        #        self.assertTrue(compress(raw_prove(Sequent([mp_s,Implication(s,mci_s),Implication(mp_s,mp_f)],mpmci_f,[swap1,swap1,swap2,swap2]))))
        self.assertTrue(compress(raw_prove(move_unlimited_resources(
            Sequent([mp_s, Implication(s, mci_s), Implication(mp_s, mp_f)], mpmci_f, [swap1, swap2])))))
        self.assertTrue(compress(raw_prove(move_unlimited_resources(Sequent([mcimp_f, swap2], mpmci_f)))))

    def test_utils(self):
        self.assertTrue(split_at([1, 2, 3, 4], 0) == ([], 1, [2, 3, 4]))
        self.assertTrue(split_at([1, 2, 3, 4], 2) == ([1, 2], 3, [4]))
        self.assertTrue(split_at([1, 2, 3, 4], 3) == ([1, 2, 3], 4, []))


def foo():
    l_t = Atom('l', AtomicType('t'))
    love = Implication(Atom('m', AtomicType('e')), Implication(Atom('w', AtomicType('e')), l_t))
    everyman = Implication(Implication(Atom('m', AtomicType('e')), Variable('x', AtomicType('t'))),
                           Variable('x', AtomicType('t')))
    awoman = Implication(Implication(Atom('w', AtomicType('e')), Variable('y', AtomicType('t'))),
                         Variable('y', AtomicType('t')))
    quantif_seq = Sequent([everyman, love, awoman], l_t)
    print(raw_prove(quantif_seq)[0].to_html())


def runTests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestProve)
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    runTests()
