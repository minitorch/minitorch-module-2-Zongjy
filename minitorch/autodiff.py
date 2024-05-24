from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Dict, Set

from typing_extensions import Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    l_vals = list(vals)
    r_vals = list(vals)
    r_vals[arg] += epsilon
    l_vals[arg] -= epsilon
    return (f(*r_vals) - f(*l_vals)) / (2.0 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    res: List[Variable] = []
    vis: Set[int] = set()

    def dfs(cur: Variable) -> None:
        if cur.is_constant() or cur.unique_id in vis:
            return

        vis.add(cur.unique_id)
        for pa in cur.parents:
            dfs(pa)
        res.append(cur)

    dfs(variable)
    return reversed(res)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    top_order = list(topological_sort(variable))
    derives: Dict[int, Any] = dict()
    derives[variable.unique_id] = deriv

    for var in top_order:
        if var.is_leaf():
            var.accumulate_derivative(derives[var.unique_id])
        else:
            if var.unique_id in derives:
                der = derives[var.unique_id]
                for pa, pa_der in var.chain_rule(der):
                    if pa.unique_id in derives:
                        derives[pa.unique_id] += pa_der
                    else:
                        derives[pa.unique_id] = pa_der


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
