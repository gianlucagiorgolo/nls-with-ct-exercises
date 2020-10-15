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


from mb.html import Node, Text



def text(txt, x, y):
    attrs = dict()
    attrs['x'] = str(x)
    attrs['y'] = str(y)
    attrs['style'] = 'text-anchor: middle;'
    return Node('text', attrs, Text(txt))


def line(x1, y1, x2, y2):
    attrs = dict()
    attrs['x1'] = str(x1)
    attrs['x2'] = str(x2)
    attrs['y1'] = str(y1)
    attrs['y2'] = str(y2)
    return Node('line', attrs)


def translate(x, y, child=None):
    return Node('g', {'transform': 'translate({},{})'.format(str(x), str(y))}, child)


def scale(factor, child=None):
    return Node('g', {'transform': 'scale({})'.format(str(factor))}, child)


def svg(width, height, child=None):
    return Node('svg', {'xmlns': "http://www.w3.org/2000/svg", 'width': str(width), 'height' : str(height)}, child)
