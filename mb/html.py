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


import copy

escapes = {'"': '&quot;', '<': '&lt;', '>': '&gt;', '&': '&amp;'}


def sanitize_string(string):
    def sanitize(c):
        if c in escapes:
            return escapes[c]
        else:
            return c

    return ''.join(sanitize(c) for c in string)


class Node(object):
    def __init__(self, tag, attributes=None, child=None):
        self.tag = tag
        if attributes is None:
            self.attributes = dict()
        else:
            self.attributes = attributes
        self.child = child

    def add_attribute(self, key, value):
        self.attributes[key] = value

    def add_child(self, child):
        if self.child is None:
            self.child = child
        elif type(self.child) == Node:
            self.child = NodeList([self.child]) + child
        else:
            self.child = self.child + child

    def __lshift__(self, child):
        self.add_child(child)
        return self

    def __str__(self):
        res = ['<', self.tag]
        if len(self.attributes) > 0:
            for k, v in self.attributes.items():
                res.extend([' ', str(k), '=', '"', sanitize_string(v), '"'])
        if self.child is None:
            res.append('/>')
        else:
            res.append('>')
            res.append(str(self.child))
            res.extend(['</', self.tag, '>'])
        return ''.join(res)

    def __and__(self, other):
        if type(other) is NodeList:
            other.nodes = [self] + other.nodes
            return other
        else:
            return NodeList([self, other])

    def __add__(self, other):
        return self & other

    def __mod__(self, attributes):
        for k, v in attributes.items():
            self.add_attribute(k, v)
        return self

    def intersperse(self, node_list):
        l = len(node_list)
        if l == 0:
            return NodeList(list())
        else:
            res = node_list[0]
            for i in range(1,l):
                res = res & self & node_list[i]
            return res

class NodeList(Node):
    def __init__(self, nodes):
        self.nodes = nodes

    def __str__(self):
        return ''.join(str(n) for n in self.nodes)

    def __and__(self, other):
        if type(other) is NodeList:
            self.nodes = self.nodes + other.nodes
            return self
        else:
            self.nodes.append(other)
            return self

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return self.nodes

    def __getitem__(self, item):
        return self.nodes[item]

class Text(Node):
    def __init__(self, content):
        self.content = content

    def add_attribute(self, key, value):
        pass

    def add_child(self, child):
        pass

    def __str__(self):
        return sanitize_string(self.content)


class RawHtml(Node):
    def __init__(self, html_code):
        self.html_code = html_code

    def add_attribute(self, key, value):
        pass

    def add_child(self, child):
        pass

    def __str__(self):
        return self.html_code


def raw_html(html_code):
    return RawHtml(html_code)


def tag(name, child=None, attributes=None):
    return Node(name, attributes, child)


def h1(child=None, attributes=None):
    return Node('h1', attributes, child)


def h2(child=None, attributes=None):
    return Node('h2', attributes, child)


def h3(child=None, attributes=None):
    return Node('h3', attributes, child)


def h4(child=None, attributes=None):
    return Node('h4', attributes, child)


def h5(child=None, attributes=None):
    return Node('h5', attributes, child)


def h6(child=None, attributes=None):
    return Node('h6', attributes, child)


def hr():
    return tag('hr')


def ol(child=None, attributes=None):
    return Node('ol', attributes, child)


def ul(child=None, attributes=None):
    return Node('ul', attributes, child)


def li(child=None, attributes=None):
    return Node('li', attributes, child)


def p(child=None, attributes=None):
    return Node('p', attributes, child)


def table(child=None, attributes=None):
    return Node('table', attributes, child)


def tr(child=None, attributes=None):
    return Node('tr', attributes, child)


def td(child=None, attributes=None):
    return Node('td', attributes, child)


def div(child=None, attributes=None):
    return Node('div', attributes, child)


def a(anchor=None, child=None, attributes=None):
    if anchor is None:
        return Node('a', attributes, child)
    elif attributes is None:
        return Node('a', {'href': anchor}, child)
    else:
        attrs = copy.copy(attributes)
        attrs['href'] = anchor
        return Node('a', attrs, child)


def html(child=None, attributes=None):
    return Node('html', attributes, child)


def body(child=None, attributes=None):
    return Node('body', attributes, child)


def br():
    return Node('br')


def ordered_list(list_of_elements, attributes=None):
    cs = reduce(lambda a, b: a & b, map(li, list_of_elements))
    return ol(cs, attributes)


def text(content):
    return Text(content)
    

def b(child=None, attributes=None):
    return Node('b', attributes, child)


def form(child=None, attributes=None, action=None, method='post'):
    if attributes is None:
        attributes = dict()
    attributes['method'] = method
    if action is not None:
        attributes['action'] = action
    return Node('form', attributes, child)


def button(label, name=None):
    attributes = None
    if name is not None:
        attributes = dict()
        attributes['name'] = name
    return Node('button', attributes, text(label))


def input(value, name=None):
    attributes = dict()
    attributes['value'] = value
    if name is not None:
        attributes['name'] = name
    return Node('input', attributes, None)

def span(child=None, attributes=None):
    return Node('span',attributes,child)
