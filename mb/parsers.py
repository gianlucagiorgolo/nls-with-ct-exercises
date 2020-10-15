#!/usr/bin/env python
# -*- coding: utf-8 -*-

# CAVEAT UTILITOR
#
# This file was automatically generated by Grako.
#
#    https://pypi.python.org/pypi/grako/
#
# Any changes you make to it will be overwritten the next time
# the file is generated.

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


from __future__ import print_function, division, absolute_import, unicode_literals

from grako.buffering import Buffer
from grako.parsing import graken, Parser
from grako.util import re, RE_FLAGS, generic_main  # noqa

__all__ = [
    'SequentsParser',
    'SequentsSemantics',
    'main'
]

KEYWORDS = {}


class SequentsBuffer(Buffer):
    def __init__(
        self,
        text,
        whitespace=None,
        nameguard=None,
        comments_re=None,
        eol_comments_re=None,
        ignorecase=None,
        namechars='',
        **kwargs
    ):
        super(SequentsBuffer, self).__init__(
            text,
            whitespace=whitespace,
            nameguard=nameguard,
            comments_re=comments_re,
            eol_comments_re=eol_comments_re,
            ignorecase=ignorecase,
            namechars=namechars,
            **kwargs
        )


class SequentsParser(Parser):
    def __init__(
        self,
        whitespace=None,
        nameguard=None,
        comments_re=None,
        eol_comments_re=None,
        ignorecase=None,
        left_recursion=False,
        parseinfo=True,
        keywords=None,
        namechars='',
        buffer_class=SequentsBuffer,
        **kwargs
    ):
        if keywords is None:
            keywords = KEYWORDS
        super(SequentsParser, self).__init__(
            whitespace=whitespace,
            nameguard=nameguard,
            comments_re=comments_re,
            eol_comments_re=eol_comments_re,
            ignorecase=ignorecase,
            left_recursion=left_recursion,
            parseinfo=parseinfo,
            keywords=keywords,
            namechars=namechars,
            buffer_class=buffer_class,
            **kwargs
        )

    @graken()
    def _formula_(self):
        with self._choice():
            with self._option():
                self._par_formula_()
                self.name_last_node('par_formula')
            with self._option():
                self._monad_()
                self.name_last_node('monad')
            with self._option():
                self._product_()
                self.name_last_node('product')
            with self._option():
                self._implication_()
                self.name_last_node('implication')
            with self._option():
                self._atom_()
                self.name_last_node('atom')
            with self._option():
                self._variable_()
                self.name_last_node('variable')
            self._error('no available options')
        self.ast._define(
            ['atom', 'implication', 'monad', 'par_formula', 'product', 'variable'],
            []
        )

    @graken()
    def _monad_(self):
        self._diamond_()
        self._formula_()

    @graken()
    def _par_formula_(self):
        self._openpar_()
        self._formula_()
        self._closedpar_()

    @graken()
    def _product_(self):
        self._formula_()
        self._asterisk_()
        self._formula_()

    @graken()
    def _implication_(self):
        self._formula_()
        self._multimap_()
        self._formula_()

    @graken()
    def _atom_(self):
        self._pattern(r'[a-z][A-Za-z0-9]*')
        self._token('.')
        self._type_()

    @graken()
    def _variable_(self):
        self._pattern(r'[A-Z][A-Za-z0-9]*')
        self._token('.')
        self._type_()

    @graken()
    def _openpar_(self):
        self._token('(')

    @graken()
    def _closedpar_(self):
        self._token(')')

    @graken()
    def _diamond_(self):
        self._token('<')
        self._modality_()
        self._token('>')

    @graken()
    def _modality_(self):
        self._pattern(r'[a-zA-Z0-9]*')

    @graken()
    def _asterisk_(self):
        self._token('*')

    @graken()
    def _multimap_(self):
        self._token('-o')

    @graken()
    def _type_(self):
        self._pattern(r'[a-zA-Z0-9]+')

    @graken()
    def _turnstile_(self):
        self._token('|-')

    @graken()
    def _sequent_(self):
        with self._group():

            def sep1():
                self._token(',')

            def block1():
                self._formula_()
            self._closure(block1, sep=sep1)
        self.name_last_node('hypotheses')
        self._turnstile_()
        self._formula_()
        self.name_last_node('consequence')
        self.ast._define(
            ['consequence', 'hypotheses'],
            []
        )


class SequentsSemantics(object):
    def formula(self, ast):
        return ast

    def monad(self, ast):
        return ast

    def par_formula(self, ast):
        return ast

    def product(self, ast):
        return ast

    def implication(self, ast):
        return ast

    def atom(self, ast):
        return ast

    def variable(self, ast):
        return ast

    def openpar(self, ast):
        return ast

    def closedpar(self, ast):
        return ast

    def diamond(self, ast):
        return ast

    def modality(self, ast):
        return ast

    def asterisk(self, ast):
        return ast

    def multimap(self, ast):
        return ast

    def type(self, ast):
        return ast

    def turnstile(self, ast):
        return ast

    def sequent(self, ast):
        return ast


def main(filename, startrule, **kwargs):
    with open(filename) as f:
        text = f.read()
    parser = SequentsParser(parseinfo=False)
    return parser.parse(text, startrule, filename=filename, **kwargs)


if __name__ == '__main__':
    import json
    ast = generic_main(main, SequentsParser, name='Sequents')
    print('AST:')
    print(ast)
    print()
    print('JSON:')
    print(json.dumps(ast, indent=2))
    print()

