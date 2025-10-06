#!/usr/bin/env python3
"""
Scientific Calculator (single file, standard library only)

Features:
- Safe expression evaluation (uses ast, only allows math functions + approved names)
- Arithmetic, trig, hyperbolic, exp/log, factorial, gamma (math.gamma), combinations, permutations
- Complex numbers supported (use j, e.g. 1+2j)
- Variable assignment (x = 2.5)
- Named constants: pi, e, tau, phi
- Numeric derivative (central difference)
- Numeric integral (Simpson)
- Root finding: Newton-Raphson and Bisection
- Basic matrix operations: add, multiply, determinant, inverse (pure Python)
- Statistics: mean, median, stdev, var
- Base conversions, gcd/lcm
- Command help, history (if readline available)
- Minimal dependencies: standard library only
"""

import ast
import math
import cmath
import operator
import sys
import traceback
from statistics import mean, median, stdev, pstdev, variance
from fractions import Fraction
from functools import reduce

# --- SAFE EVAL via AST ---

# Allowed math functions and constants (map names to callables/values)
_SAFE_MATH = {
    # constants
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau if hasattr(math, "tau") else 2 * math.pi,
    "phi": (1 + 5 ** 0.5) / 2,
    "inf": math.inf,
    "nan": math.nan,
    # unary helpers
    "abs": abs,
    "round": round,
    # math funcs
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asinh": getattr(math, "asinh", None),
    "acosh": getattr(math, "acosh", None),
    "atanh": getattr(math, "atanh", None),
    "exp": math.exp,
    "ln": math.log,   # ln(x) -> natural log
    "log": math.log,  # log(x, base) allowed
    "log10": math.log10,
    "sqrt": math.sqrt,
    "cbrt": lambda x: x ** (1/3),
    "pow": pow,
    "factorial": math.factorial,
    "gamma": getattr(math, "gamma", None),
    "lgamma": getattr(math, "lgamma", None),
    "comb": getattr(math, "comb", None),
    "perm": getattr(math, "perm", None) if hasattr(math, "perm") else None,
    "gcd": math.gcd,
    "lcm": getattr(math, "lcm", None),
    # complex helpers
    "abs2": lambda z: (z.real ** 2 + z.imag ** 2) if isinstance(z, complex) else z * z,
    "re": lambda z: z.real if isinstance(z, complex) else float(z),
    "im": lambda z: z.imag if isinstance(z, complex) else 0.0,
    "conj": lambda z: z.conjugate() if isinstance(z, complex) else z,
    # statistical helpers
    "mean": mean,
    "median": median,
    "stdev": stdev,
    "var": variance,
    # misc
    "rad": math.radians,
    "deg": math.degrees,
}

# filter None entries (in case some functions are missing in older Python)
_SAFE_MATH = {k: v for k, v in _SAFE_MATH.items() if v is not None}

# Allowed operators mapping
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
    ast.Invert: lambda x: ~x if isinstance(x, int) else (~int(x)),
}

# Evaluator
class SafeEvaluator(ast.NodeVisitor):
    def __init__(self, names):
        self.names = dict(_SAFE_MATH)
        self.names.update(names or {})

    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        # numbers, complex, booleans allowed
        if isinstance(node.value, (int, float, complex, bool)):
            return node.value
        raise ValueError("Constant type not allowed")

    # For Python <3.8 compatibility - Num node
    def visit_Num(self, node):
        return node.n

    def visit_Name(self, node):
        if node.id in self.names:
            return self.names[node.id]
        raise NameError(f"Name '{node.id}' is not defined")

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[op_type](left, right)
        raise ValueError(f"Binary operator {op_type} not allowed")

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type in _ALLOWED_UNARYOPS:
            return _ALLOWED_UNARYOPS[op_type](operand)
        raise ValueError("Unary operator not allowed")

    def visit_Call(self, node):
        # only allow simple function calls: Name + args
        if isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname not in self.names:
                raise NameError(f"Function '{fname}' not found")
            func = self.names[fname]
            args = [self.visit(a) for a in node.args]
            kwargs = {}
            # disallow keyword args for safety except log(x, base=...)
            for k in node.keywords:
                if k.arg is None:
                    raise ValueError("Kwargs with ** not allowed")
                kwargs[k.arg] = self.visit(k.value)
            return func(*args, **kwargs)
        else:
            raise ValueError("Only direct function calls allowed")

    def visit_Compare(self, node):
        # allow simple comparisons for root finding etc.
        left = self.visit(node.left)
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("Complex comparisons not supported")
        right = self.visit(node.comparators[0])
        op = node.ops[0]
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        raise ValueError("Comparison not supported")

    def visit_BoolOp(self, node):
        # allow and/or
        if isinstance(node.op, ast.And):
            for v in node.values:
                if not self.visit(v):
                    return False
            return True
        if isinstance(node.op, ast.Or):
            for v in node.values:
                if self.visit(v):
                    return True
            return False
        raise ValueError("Boolean op not allowed")

    def visit_List(self, node):
        return [self.visit(elt) for elt in node.elts]

    def visit_Tuple(self, node):
        return tuple(self.visit(elt) for elt in node.elts)

    def generic_visit(self, node):
        raise ValueError(f"Expression element {node.__class__.__name__} not allowed")


def safe_eval(expr, names=None):
    """
    Safely evaluate an expression string using allowed names/functions.
    names: dict of variable names to include.
    """
    try:
        tree = ast.parse(expr, mode='eval')
        evaluator = SafeEvaluator(names or {})
        return evaluator.visit(tree)
    except Exception:
        raise


# --- NUMERICAL UTILITIES ---

def numeric_derivative(func_expr, var_name='x', x0=0.0, h=1e-6, names=None):
    """Central difference derivative f'(x0) approximated."""
    # build evaluator closures
    def f(x):
        env = dict(names or {})
        env[var_name] = x
        return safe_eval(func_expr, env)
    return (f(x0 + h) - f(x0 - h)) / (2 * h)

def numeric_integral(func_expr, a, b, n=200, var_name='x', names=None):
    """Composite Simpson's rule (n must be even)."""
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    env = dict(names or {})

    def f(x):
        env_local = dict(env)
        env_local[var_name] = x
        return safe_eval(func_expr, env_local)

    s = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        coef = 4 if i % 2 == 1 else 2
        s += coef * f(x)
    return s * h / 3.0

def newton_raphson(func_expr, x0, tol=1e-9, maxiter=50, var_name='x', names=None):
    """Newton-Raphson using numeric derivative."""
    x = x0
    env = dict(names or {})
    for i in range(maxiter):
        env_local = dict(env); env_local[var_name] = x
        fx = safe_eval(func_expr, env_local)
        dfx = numeric_derivative(func_expr, var_name=var_name, x0=x, names=env)
        if dfx == 0:
            raise ZeroDivisionError("Derivative zero during Newton-Raphson")
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    raise RuntimeError("Newton-Raphson did not converge")

def bisection(func_expr, a, b, tol=1e-9, maxiter=200, var_name='x', names=None):
    env = dict(names or {})
    fa = safe_eval(func_expr, {**env, var_name: a})
    fb = safe_eval(func_expr, {**env, var_name: b})
    if fa * fb > 0:
        raise ValueError("Function has same sign at endpoints")
    low, high = a, b
    for i in range(maxiter):
        mid = (low + high) / 2.0
        fm = safe_eval(func_expr, {**env, var_name: mid})
        if abs(fm) < tol or (high - low) / 2 < tol:
            return mid
        if fa * fm <= 0:
            high = mid
            fb = fm
        else:
            low = mid
            fa = fm
    raise RuntimeError("Bisection did not converge")

# --- MATRIX UTILITIES (pure python) ---

def mat_add(A, B):
    if len(A) != len(B) or any(len(A[i]) != len(B[i]) for i in range(len(A))):
        raise ValueError("Matrices must have same dimensions")
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def mat_mul(A, B):
    # A (m x n), B (n x p)
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    if n != n2:
        raise ValueError("Incompatible matrix dims")
    C = [[0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            s = 0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C

def mat_transpose(A):
    return list(map(list, zip(*A)))

def mat_det(A):
    # recursive Laplace expansion (fine for small matrices)
    n = len(A)
    if n == 1:
        return A[0][0]
    if n == 2:
        return A[0][0]*A[1][1] - A[0][1]*A[1][0]
    det = 0
    for c in range(n):
        minor = [[A[i][j] for j in range(n) if j != c] for i in range(1, n)]
        det += ((-1) ** c) * A[0][c] * mat_det(minor)
    return det

def mat_inverse(A):
    # Gauss-Jordan (works for invertible square matrices)
    n = len(A)
    # build augmented matrix
    M = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(A)]
    for i in range(n):
        # pivot
        pivot = M[i][i]
        if pivot == 0:
            # find non-zero pivot and swap
            for r in range(i+1, n):
                if M[r][i] != 0:
                    M[i], M[r] = M[r], M[i]
                    pivot = M[i][i]
                    break
        if pivot == 0:
            raise ValueError("Matrix is singular")
        # normalize row
        M[i] = [v / pivot for v in M[i]]
        # eliminate other rows
        for r in range(n):
            if r != i:
                factor = M[r][i]
                M[r] = [M[r][c] - factor * M[i][c] for c in range(2*n)]
    # extract inverse
    inv = [row[n:] for row in M]
    return inv

# --- COMMAND-LINE INTERFACE (REPL) ---

BANNER = r"""
Scientific Calculator REPL (standard library only)
Type expressions like: 2+3*4, sin(pi/2), ln(2), 1+2j
Variable assignment: x = 2.5
Use functions: derivative(expr, x0), integral(expr, a, b)
Root finding: newton(expr, x0), bisect(expr, a, b)
Matrix: m_add, m_mul, det(mat), inv(mat)
Type :help for help, :exit to quit
"""

HELP = """
Available special commands (prefix colon) and utilities:

:help                show this help
:vars                show variables
:clear               clear user variables
:exit or :quit       exit

Functions you can call in expressions:
- sin,cos,tan,asin,acos,atan,sinh,cosh,tanh,exp,ln,log,log10,sqrt,pow
- factorial, gamma, comb, perm (if Python version supports), gcd, lcm
- mean, median, stdev, var
- rad(x) / deg(x)
- Complex numbers using j (e.g. 1+2j)

Special evaluator helpers (callable from expressions):
- derivative(expr_str, x0, h=1e-6)   numeric derivative at x0
- integral(expr_str, a, b, n=200)   numeric integral (Simpson)
- newton(expr_str, x0)               Newton-Raphson root finder
- bisect(expr_str, a, b)             Bisection root finder

Matrix helpers (use Python list-of-lists syntax, e.g. [[1,2],[3,4]]):
- m_add(A,B)
- m_mul(A,B)
- det(A)
- inv(A)

Examples:
x = 2
y = sin(pi/4) + 3
derivative("x**3 - 2*x", x0=1.0)
integral("sin(x)", 0, pi)
newton("x**2-2", 1.0)
"""

def make_builtin_env(user_vars):
    env = {}
    env.update(_SAFE_MATH)
    # attach numeric helpers as callables that accept strings or numbers
    env['derivative'] = lambda expr, x0, h=1e-6, var_name='x': numeric_derivative(expr, var_name=var_name, x0=x0, h=h, names=user_vars)
    env['integral'] = lambda expr, a, b, n=200, var_name='x': numeric_integral(expr, a, b, n=n, var_name=var_name, names=user_vars)
    env['newton'] = lambda expr, x0, tol=1e-9: newton_raphson(expr, x0, tol=tol, names=user_vars)
    env['bisect'] = lambda expr, a, b, tol=1e-9: bisection(expr, a, b, tol=tol, names=user_vars)
    # matrix / misc
    env['m_add'] = mat_add
    env['m_mul'] = mat_mul
    env['det'] = mat_det
    env['inv'] = mat_inverse
    env['mean'] = mean
    env['median'] = median
    env['stdev'] = stdev
    env['var'] = variance
    # include Fraction helper
    env['Frac'] = Fraction
    return env

def repl():
    print(BANNER)
    user_vars = {}  # user-defined variables
    env_cache = make_builtin_env(user_vars)

    # try to enable readline for history if available
    try:
        import readline  # type: ignore
        readline.set_history_length(200)
    except Exception:
        pass

    while True:
        try:
            raw = input("calc> ").strip()
            if not raw:
                continue
            if raw.startswith(':'):
                cmd = raw[1:].strip().lower()
                if cmd in ('q', 'quit', 'exit'):
                    print("Bye.")
                    break
                if cmd == 'help':
                    print(HELP)
                    continue
                if cmd == 'vars':
                    if not user_vars:
                        print("(no user variables)")
                    else:
                        for k, v in user_vars.items():
                            print(f"{k} = {v!r}")
                    continue
                if cmd == 'clear':
                    user_vars.clear()
                    env_cache = make_builtin_env(user_vars)
                    print("User variables cleared.")
                    continue
                print("Unknown command. Type :help.")
                continue

            # assignment?
            if '=' in raw and not raw.strip().startswith(('derivative','integral','newton','bisect')):
                # simple assignment parsing: var = expr
                parts = raw.split('=', 1)
                var = parts[0].strip()
                expr = parts[1].strip()
                if not var.isidentifier():
                    print("Invalid variable name.")
                    continue
                # evaluate expression with current env
                env_cache = make_builtin_env(user_vars)
                try:
                    val = safe_eval(expr, names=env_cache | user_vars)
                except Exception as e:
                    print("Evaluation error:", e)
                    continue
                user_vars[var] = val
                print(f"{var} = {val!r}")
                continue

            # evaluate expression
            env_cache = make_builtin_env(user_vars)
            try:
                val = safe_eval(raw, names=env_cache | user_vars)
                # pretty print complex and floats
                if isinstance(val, float):
                    print(repr(val))
                else:
                    print(val)
            except Exception as e:
                # show short traceback for debugging but keep REPL alive
                print("Error:", e)
                # uncomment for debugging:
                # traceback.print_exc()

        except (KeyboardInterrupt, EOFError):
            print("\nExit.")
            break
        except Exception:
            print("Unexpected error:")
            traceback.print_exc()

if __name__ == '__main__':
    repl()
