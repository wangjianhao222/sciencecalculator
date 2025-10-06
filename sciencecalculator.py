# streamlit_scientific_calculator.py
"""
Streamlit Scientific Calculator (single-file)
- Uses a safe AST-based evaluator (no direct eval of raw input)
- Supports variables, functions, numeric derivative/integral/root finding
- Matrix helpers (list-of-lists)
- Minimal dependency: streamlit
Save -> run: `streamlit run streamlit_scientific_calculator.py`
"""

import streamlit as st
import ast
import math
import operator
import random
import io
from datetime import datetime
from statistics import mean, median, stdev, variance
from fractions import Fraction
from functools import reduce

st.set_page_config(page_title="Scientific Calculator", layout="wide")

# ---------------- safe evaluator (AST) ----------------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: lambda x: +x,
    ast.USub: lambda x: -x,
    ast.Invert: lambda x: ~x if isinstance(x, int) else ~int(x),
}

_SAFE_MATH = {
    "pi": math.pi,
    "e": math.e,
    "tau": 2 * math.pi,
    "phi": (1 + 5 ** 0.5) / 2,
    "abs": abs,
    "round": round,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": getattr(math, "asin", None),
    "acos": getattr(math, "acos", None),
    "atan": getattr(math, "atan", None),
    "sinh": getattr(math, "sinh", None),
    "cosh": getattr(math, "cosh", None),
    "tanh": getattr(math, "tanh", None),
    "exp": math.exp,
    "ln": math.log,
    "log": math.log,
    "log10": getattr(math, "log10", None),
    "sqrt": math.sqrt,
    "factorial": getattr(math, "factorial", None),
    "gamma": getattr(math, "gamma", None),
    "comb": getattr(math, "comb", None),
    "gcd": getattr(math, "gcd", None),
    "mean": mean,
    "median": median,
    "stdev": stdev,
    "var": variance,
    "rad": math.radians,
    "deg": math.degrees,
    "Frac": Fraction,
}

# remove None
_SAFE_MATH = {k: v for k, v in _SAFE_MATH.items() if v is not None}

class SafeEval(ast.NodeVisitor):
    def __init__(self, names=None):
        self.names = dict(_SAFE_MATH)
        if names:
            self.names.update(names)

    def visit(self, node):
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float, complex, bool)):
            return node.value
        raise ValueError("Constant type not allowed")

    def visit_Num(self, node):
        return node.n

    def visit_Name(self, node):
        if node.id in self.names:
            return self.names[node.id]
        raise NameError(f"Name '{node.id}' is not defined")

    def visit_BinOp(self, node):
        L = self.visit(node.left)
        R = self.visit(node.right)
        op_type = type(node.op)
        if op_type in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[op_type](L, R)
        raise ValueError("Binary operator not allowed")

    def visit_UnaryOp(self, node):
        val = self.visit(node.operand)
        t = type(node.op)
        if t in _ALLOWED_UNARYOPS:
            return _ALLOWED_UNARYOPS[t](val)
        raise ValueError("Unary operator not allowed")

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in self.names:
                raise NameError(f"Function '{name}' not found")
            func = self.names[name]
            args = [self.visit(a) for a in node.args]
            kwargs = {}
            for k in node.keywords:
                if k.arg is None:
                    raise ValueError("**kwargs not allowed")
                kwargs[k.arg] = self.visit(k.value)
            return func(*args, **kwargs)
        raise ValueError("Only direct function calls allowed")

    def visit_List(self, node):
        return [self.visit(e) for e in node.elts]

    def visit_Tuple(self, node):
        return tuple(self.visit(e) for e in node.elts)

    def generic_visit(self, node):
        raise ValueError(f"Element {node.__class__.__name__} not allowed")

def safe_eval(expr: str, names=None):
    tree = ast.parse(expr, mode="eval")
    ev = SafeEval(names)
    return ev.visit(tree)

# --------------- numeric helpers ----------------

def numeric_derivative(expr, x0=0.0, h=1e-6, var="x", names=None):
    def f(x):
        env = dict(names or {})
        env[var] = x
        return safe_eval(expr, names=env)
    return (f(x0 + h) - f(x0 - h)) / (2 * h)

def numeric_integral(expr, a, b, n=200, var="x", names=None):
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    env = dict(names or {})
    def f(x):
        e = dict(env); e[var] = x
        return safe_eval(expr, names=e)
    s = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        s += (4 if i % 2 else 2) * f(x)
    return s * h / 3.0

def newton_root(expr, x0, tol=1e-9, maxiter=50, var="x", names=None):
    x = x0
    env = dict(names or {})
    for _ in range(maxiter):
        val = safe_eval(expr, names={**env, var: x})
        der = numeric_derivative(expr, x0=x, var=var, names=env)
        if der == 0:
            raise ZeroDivisionError("Zero derivative")
        x_new = x - val / der
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    raise RuntimeError("Newton did not converge")

def bisect_root(expr, a, b, tol=1e-9, maxiter=200, var="x", names=None):
    env = dict(names or {})
    fa = safe_eval(expr, names={**env, var: a})
    fb = safe_eval(expr, names={**env, var: b})
    if fa * fb > 0:
        raise ValueError("Same sign at endpoints")
    low, high = a, b
    for _ in range(maxiter):
        mid = (low + high) / 2.0
        fm = safe_eval(expr, names={**env, var: mid})
        if abs(fm) < tol or (high - low)/2 < tol:
            return mid
        if fa * fm <= 0:
            high = mid
            fb = fm
        else:
            low = mid
            fa = fm
    raise RuntimeError("Bisection did not converge")

# --------------- matrix helpers ----------------

def mat_add(A, B):
    if len(A) != len(B) or any(len(A[i]) != len(B[i]) for i in range(len(A))):
        raise ValueError("Dimension mismatch")
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def mat_mul(A, B):
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    if n != n2:
        raise ValueError("Incompatible dims")
    C = [[0]*p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            s = 0
            for k in range(n):
                s += A[i][k]*B[k][j]
            C[i][j] = s
    return C

def mat_det(A):
    n = len(A)
    if n == 1:
        return A[0][0]
    if n == 2:
        return A[0][0]*A[1][1]-A[0][1]*A[1][0]
    det = 0
    for c in range(n):
        minor = [[A[i][j] for j in range(n) if j != c] for i in range(1, n)]
        det += ((-1)**c) * A[0][c] * mat_det(minor)
    return det

def mat_inv(A):
    n = len(A)
    M = [row[:] + [float(i==j) for j in range(n)] for i, row in enumerate(A)]
    for i in range(n):
        pivot = M[i][i]
        if pivot == 0:
            for r in range(i+1, n):
                if M[r][i] != 0:
                    M[i], M[r] = M[r], M[i]
                    pivot = M[i][i]
                    break
        if pivot == 0:
            raise ValueError("Singular matrix")
        M[i] = [v/pivot for v in M[i]]
        for r in range(n):
            if r != i:
                factor = M[r][i]
                M[r] = [M[r][c] - factor*M[i][c] for c in range(2*n)]
    return [row[n:] for row in M]

# ---------------- streamlit UI ----------------

st.title("Scientific Calculator (Streamlit)")
st.write("Safe AST-based evaluator. Use math functions, define variables, run numeric helpers.")

# session state: variables and history
if "vars" not in st.session_state:
    st.session_state.vars = {}
if "history" not in st.session_state:
    st.session_state.history = []

cols = st.columns([3, 1])
with cols[0]:
    expr = st.text_area("Expression or assignment (e.g. x = 2.5 or sin(pi/4))", height=120)
    evaluate = st.button("Evaluate")
    clear_hist = st.button("Clear History")
with cols[1]:
    st.subheader("Helpers")
    if st.button("Derivative helper"):
        st.text("Usage: derivative(expr, x0) e.g. derivative('x**2', 2)")
    if st.button("Integral helper"):
        st.text("Usage: integral(expr, a, b) e.g. integral('sin(x)', 0, pi)")
    if st.button("Root helper"):
        st.text("Usage: newton(expr, x0) or bisect(expr, a, b)")

# Evaluate logic
output = ""
error = None
if evaluate and expr.strip():
    raw = expr.strip()
    try:
        # assignment?
        if "=" in raw and not raw.strip().startswith(("derivative","integral","newton","bisect")):
            left, right = raw.split("=", 1)
            var = left.strip()
            if not var.isidentifier():
                raise ValueError("Invalid variable name")
            env = {**st.session_state.vars}
            val = safe_eval(right, names=env)
            st.session_state.vars[var] = val
            output = f"{var} = {val!r}"
        else:
            env = {**st.session_state.vars}
            # attach helpers to env for expression calls
            env.update({
                "derivative": lambda e, x0, h=1e-6: numeric_derivative(e, x0=x0, h=h, names=st.session_state.vars),
                "integral": lambda e, a, b, n=200: numeric_integral(e, a, b, n=n, names=st.session_state.vars),
                "newton": lambda e, x0: newton_root(e, x0, names=st.session_state.vars),
                "bisect": lambda e, a, b: bisect_root(e, a, b, names=st.session_state.vars),
                "m_add": mat_add, "m_mul": mat_mul, "det": mat_det, "inv": mat_inv,
            })
            val = safe_eval(raw, names=env)
            output = repr(val)
        # append history
        st.session_state.history.append({"time": datetime.now().isoformat(), "expr": raw, "result": output})
    except Exception as e:
        error = str(e)

if clear_hist:
    st.session_state.history = []

# outputs
if error:
    st.error(f"Error: {error}")
elif output:
    st.success(output)

# show variables and history
st.subheader("Variables")
if st.session_state.vars:
    st.json(st.session_state.vars)
else:
    st.write("(no variables)")

st.subheader("History (latest 50)")
for item in st.session_state.history[-50:][::-1]:
    st.markdown(f"- `{item['expr']}` → `{item['result']}`  _{item['time']}_")
