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


from flask import Flask, render_template
from mb.tp import Sequent, Atom, AtomicType, universal_type, Variable, Implication, Tensor, Monad

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('page.html')

@app.route('/foo/<bar>')
def foo(bar):
	return 'Foo!!!' + bar

def ast2sequent(ast):
	def get_type(type_def):
		if type_def == 'ANYTYPE':
			return universal_type
		else:
			return AtomicType(type_def)
	def gen_atom(ast):
		return Atom(ast[0], get_type(ast[2]))
	def gen_variable(ast):
		return Variable(ast[0], get_type(ast[2]))
	def gen_implication(ast):
		return Implication(gen_formula(ast[0]), gen_formula(ast[2]))
	def gen_product(ast):
		return Tensor(gen_formula(ast[0]), gen_formula(ast[2]))
	def gen_monad(ast):
		return Monad(gen_formula(ast[3]), ast[1])
	def gen_formula(ast):
		if ast['atom'] is not None:
			return gen_atom(ast['atom'])
		elif ast['variable'] is not None:
			return gen_variable(ast['variable'])
		elif ast['monad'] is not None:
			return gen_monad(ast['monad'])
		elif ast['product'] is not None:
			return gen_product(ast['product'])
		elif ast['implication'] is not None:
			return gen_implication(ast['implication'])
		else:
			return gen_formula(ast['par_formula'][1])
	def gen_hypotheses(hypotheses_asts): 
		return [gen_formula(ast) for ast in hypotheses_asts]
	def gen_consequence(consequence_ast):
		return gen_formula(consequence_ast)
	return Sequent(gen_hypotheses(ast['hypotheses']), gen_consequence(ast['consequence']))
