"""
AST transformer to convert Pythonic for/if statements into IR builder calls
(scf.ForOp + InsertionPoint, scf.IfOp + InsertionPoint + scf.YieldOp).

Effect is isolated at function level: only the body of specified function(s)
is transformed; the rest of the module is unchanged.
"""

import ast
from typing import List, Optional, Sequence


def _is_pto_range(node: ast.AST) -> bool:
    """True if node is a call to pto.range(start=..., end=..., step=...)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if not isinstance(func.value, ast.Name) or func.value.id != "pto":
        return False
    if func.attr != "range":
        return False
    keywords = {kw.arg: kw.value for kw in (node.keywords or []) if kw.arg}
    return "start" in keywords and "end" in keywords and "step" in keywords


def _ast_bool_const(value: bool):
    """AST node for a boolean constant (Constant in 3.8+, NameConstant in 3.7)."""
    if hasattr(ast, "Constant"):
        return ast.Constant(value=value)
    return ast.NameConstant(value=value)  # type: ignore[attr-defined]


def _get_pto_range_bounds(node: ast.Call):
    """Return (start, end, step) AST values for pto.range(...)."""
    keywords = {kw.arg: kw.value for kw in (node.keywords or []) if kw.arg}
    return keywords["start"], keywords["end"], keywords["step"]


def _make_name(name: str):
    return ast.Name(name, ast.Store()) if hasattr(ast, "Store") else ast.Name(name, ast.Load())


def _store_ctx(node: ast.Name):
    """Return node with Store context (for assignment target)."""
    if hasattr(ast, "Store"):
        return ast.Name(node.id, ast.Store())
    return node


def _load_ctx(node: ast.Name):
    return ast.Name(node.id, ast.Load())


class IRBuilderTransformer(ast.NodeTransformer):
    """
    Converts Pythonic control flow to IR builder style:
    - for i in pto.range(start=a, end=b, step=c): body  ->  loop = scf.ForOp(a,b,c,[]); with InsertionPoint(loop.body): i = loop.induction_variable; body
    - if cond: then_body else: else_body  ->  _if = scf.IfOp(cond,[],hasElse=True); with InsertionPoint(_if.then_block): then_body; scf.YieldOp([]); with InsertionPoint(_if.else_block): else_body; scf.YieldOp([])
    """

    def __init__(self, function_names: Optional[Sequence[str]] = None):
        self._if_counter = 0
        self._loop_counter = 0
        self.function_names = set(function_names) if function_names else None

    def _next_loop_name(self) -> str:
        name = "_loop" if self._loop_counter == 0 else f"_loop_{self._loop_counter}"
        self._loop_counter += 1
        return name

    def _next_if_name(self) -> str:
        name = "_if" if self._if_counter == 0 else f"_if_{self._if_counter}"
        self._if_counter += 1
        return name

    def _transform_body(self, body: list) -> List[ast.AST]:
        """Transform a list of statements; expand for/if at top level."""
        new_body: List[ast.AST] = []
        for stmt in body:
            if isinstance(stmt, ast.For) and _is_pto_range(stmt.iter):
                new_body.extend(self._expand_for(stmt))
            elif isinstance(stmt, ast.If):
                new_body.extend(self._expand_if(stmt))
            else:
                new_body.append(self.visit(stmt))
        return new_body

    def _expand_for(self, node: ast.For) -> List[ast.AST]:
        """Expand for i in pto.range(...): body into scf.ForOp + with InsertionPoint."""
        start, end, step = _get_pto_range_bounds(node.iter)
        loop_var = self._next_loop_name()
        # loop_var = scf.ForOp(start, end, step, [])
        for_op = ast.Call(
            func=ast.Attribute(value=ast.Name("scf", ast.Load()), attr="ForOp", ctx=ast.Load()),
            args=[start, end, step, ast.List(elts=[], ctx=ast.Load())],
            keywords=[],
        )
        assign_loop = ast.Assign(
            targets=[ast.Name(loop_var, ast.Store())],
            value=for_op,
        )
        # i = loop.induction_variable (target may be tuple for "for i,j in ..." but we assume single name)
        induction_target = node.target
        if isinstance(induction_target, ast.Name):
            assign_induction = ast.Assign(
                targets=[ast.Name(induction_target.id, ast.Store())],
                value=ast.Attribute(
                    value=ast.Name(loop_var, ast.Load()),
                    attr="induction_variable",
                    ctx=ast.Load(),
                ),
            )
        else:
            # e.g. for (a,b) in ... - unsupported for now; keep single name
            assign_induction = ast.Assign(
                targets=[induction_target],
                value=ast.Attribute(
                    value=ast.Name(loop_var, ast.Load()),
                    attr="induction_variable",
                    ctx=ast.Load(),
                ),
            )
        inner_body = [assign_induction] + self._transform_body(node.body)
        # with InsertionPoint(loop.body): ...
        with_ctx = ast.Call(
            func=ast.Name("InsertionPoint", ast.Load()),
            args=[
                ast.Attribute(
                    value=ast.Name(loop_var, ast.Load()),
                    attr="body",
                    ctx=ast.Load(),
                )
            ],
            keywords=[],
        )
        with_item = ast.withitem(context_expr=with_ctx, optional_vars=None)
        with_node = ast.With(items=[with_item], body=inner_body)
        return [assign_loop, with_node]

    def _expand_if(self, node: ast.If) -> List[ast.AST]:
        """Expand if cond: then else: else_ into scf.IfOp + with InsertionPoint + scf.YieldOp."""
        if_var = self._next_if_name()
        has_else = bool(node.orelse)
        # _if = scf.IfOp(cond, [], hasElse=True/False)
        if_op = ast.Call(
            func=ast.Attribute(value=ast.Name("scf", ast.Load()), attr="IfOp", ctx=ast.Load()),
            args=[node.test, ast.List(elts=[], ctx=ast.Load())],
            keywords=[ast.keyword(arg="hasElse", value=_ast_bool_const(has_else))],
        )
        assign_if = ast.Assign(
            targets=[ast.Name(if_var, ast.Store())],
            value=if_op,
        )
        # then_block body + scf.YieldOp([])
        then_body = self._transform_body(node.body)
        then_body.append(
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(value=ast.Name("scf", ast.Load()), attr="YieldOp", ctx=ast.Load()),
                    args=[ast.List(elts=[], ctx=ast.Load())],
                    keywords=[],
                )
            )
        )
        with_then = ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(
                        func=ast.Name("InsertionPoint", ast.Load()),
                        args=[
                            ast.Attribute(
                                value=ast.Name(if_var, ast.Load()),
                                attr="then_block",
                                ctx=ast.Load(),
                            )
                        ],
                        keywords=[],
                    ),
                    optional_vars=None,
                )
            ],
            body=then_body,
        )
        result = [assign_if, with_then]
        if has_else:
            else_body = self._transform_body(node.orelse)
            else_body.append(
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(value=ast.Name("scf", ast.Load()), attr="YieldOp", ctx=ast.Load()),
                        args=[ast.List(elts=[], ctx=ast.Load())],
                        keywords=[],
                    )
                )
            )
            with_else = ast.With(
                items=[
                    ast.withitem(
                        context_expr=ast.Call(
                            func=ast.Name("InsertionPoint", ast.Load()),
                            args=[
                                ast.Attribute(
                                    value=ast.Name(if_var, ast.Load()),
                                    attr="else_block",
                                    ctx=ast.Load(),
                                )
                            ],
                            keywords=[],
                        ),
                        optional_vars=None,
                    )
                ],
                body=else_body,
            )
            result.append(with_else)
        return result

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Transform only the body of functions in self.function_names (or all if None)."""
        if self.function_names is not None and node.name not in self.function_names:
            return self.generic_visit(node)
        new_body = self._transform_body(node.body)
        return ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=new_body,
            decorator_list=node.decorator_list,
            returns=node.returns,
            type_comment=node.type_comment if hasattr(node, "type_comment") else None,
        )

    def visit_With(self, node: ast.With):
        """Transform body of with blocks (e.g. with InsertionPoint(entry):)."""
        new_body = self._transform_body(node.body)
        return ast.With(items=node.items, body=new_body)

    def visit_Module(self, node: ast.Module):
        """Module body: visit each stmt (transform only inside functions in function_names)."""
        new_body = [self.visit(stmt) for stmt in node.body]
        return ast.Module(body=new_body, type_ignores=getattr(node, "type_ignores", []))


def transform_function_source(source: str, function_name: str) -> str:
    """
    Parse module source, transform only the named function's body, return
    transformed module source. Effect is isolated at function level.
    """
    return transform_module_source(source, [function_name])


def transform_module_source(source: str, function_names: Optional[Sequence[str]] = None) -> str:
    """
    Parse module source and transform the body of specified functions.
    If function_names is None, transform all function definitions.
    Returns transformed module source.
    """
    tree = ast.parse(source)
    transformer = IRBuilderTransformer(function_names=function_names)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)
