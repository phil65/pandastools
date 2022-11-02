import ast
import itertools
import logging
import time
from typing import Any, Optional


logger = logging.getLogger(__name__)


def uniquify_names(new_cols, old_cols):
    cols = []
    for col in new_cols:
        if col not in old_cols:
            cols.append(col)
            continue
        for i in itertools.count():
            name = f"{col}_{i}"
            if name not in old_cols:
                cols.append(name)
                break
    return cols


def evaluate(
    code, context: Optional[dict[str, Any]] = None, return_val: Optional[str] = None
):
    """Evaluate several lines of input, returning the result of the last line."""
    now = time.time()
    if context is None:
        context = locals()
    code = str(code) if return_val is None else f"{code}\n{return_val}"
    logger.debug("Evaluating code:\n%s", code)
    tree = ast.parse(code)
    eval_expr = ast.Expression(tree.body[-1].value)  # type: ignore
    # exec_expr = ast.Module(tree.body[:-1])  # type: ignore
    exec_expr = ast.parse("")
    exec_expr.body = tree.body[:-1]
    compiled = compile(exec_expr, "file", "exec")
    exec(compiled, context)
    val = eval(compile(eval_expr, "file", "eval"), context)
    logger.debug("Code evaluation took %s seconds.", time.time() - now)
    return val
