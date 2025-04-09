from functools import reduce
from operator import mul as mul_py
from typing import Any, Dict, List, Set

import torch
from sympy import expand
from torch._prims_common import Dim
from torch.nn.functional import softmax as softmax_pytorch


class Node:
    """Node in a computational graph.

    Fields
    ------
    inputs: List[Node]
        The list of input nodes to this node.

    op: Op
        The op of this node.

    attrs: Dict[str, Any]
        The attribute dictionary of this node.
        E.g. "constant" is the constant operand of add_by_const.

    name: str
        Name of the node for debugging purposes.
    """

    inputs: List["Node"]
    op: "Op"
    attrs: Dict[str, Any]
    name: str

    def __init__(
        self, inputs: List["Node"], op: "Op", attrs: Dict[str, Any] = {}, name: str = ""
    ) -> None:
        self.inputs = inputs
        self.op = op
        self.attrs = attrs
        self.name = name

    def __add__(self, other):
        if isinstance(other, Node):
            return add(self, other)
        else:
            assert isinstance(other, (int, float))
            return add_by_const(self, other)

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __mul__(self, other):
        if isinstance(other, Node):
            return mul(self, other)
        else:
            assert isinstance(other, (int, float))
            return mul_by_const(self, other)

    def __pow__(self, other):
        if isinstance(other, Node):
            raise NotImplementedError(
                "Power operation with another node is not implemented."
            )
        else:
            assert isinstance(other, (int, float))
            return power(self, other)

    def __gt__(self, other):
        if isinstance(other, Node):
            return greater(self, other)
        else:
            assert isinstance(other, (int, float))
            return greater_by_const(self, other)

    def __truediv__(self, other):
        if isinstance(other, Node):
            return div(self, other)
        else:
            assert isinstance(other, (int, float))
            return div_by_const(self, other)

    # Allow left-hand-side add and multiplication.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow printing the node name."""
        return self.name

    def __getattr__(self, attr_name: str) -> Any:
        if attr_name in self.attrs:
            return self.attrs[attr_name]
        raise KeyError(f"Attribute {attr_name} does not exist in node {self}")

    __repr__ = __str__


class Variable(Node):
    """A variable node with given name."""

    def __init__(self, name: str) -> None:
        super().__init__(inputs=[], op=placeholder, name=name)


class Op:
    """The class of operations performed on nodes."""

    def __call__(self, *kwargs) -> Node:
        """Create a new node with this current op.

        Returns
        -------
        The created new node.
        """
        raise NotImplementedError

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Compute the output value of the given node with its input
        node values given.

        Parameters
        ----------
        node: Node
            The node whose value is to be computed

        input_values: List[torch.Tensor]
            The input values of the given node.

        Returns
        -------
        output: torch.Tensor
            The computed output value of the node.
        """
        raise NotImplementedError

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given a node and its output gradient node, compute partial
        adjoints with regards to each input node.

        Parameters
        ----------
        node: Node
            The node whose inputs' partial adjoints are to be computed.

        output_grad: Node
            The output gradient with regard to given node.

        Returns
        -------
        input_grads: List[Node]
            The list of partial gradients with regard to each input of the node.
        """
        raise NotImplementedError


class PlaceholderOp(Op):
    """The placeholder op to denote computational graph input nodes."""

    def __call__(self, name: str) -> Node:
        return Node(inputs=[], op=self, name=name)

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        raise RuntimeError(
            "Placeholder nodes have no inputs, and there values cannot be computed."
        )

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        raise RuntimeError("Placeholder nodes have no inputs.")


class AddOp(Op):
    """Op to element-wise add two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}+{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise addition of input values."""
        assert len(input_values) == 2
        return input_values[0] + input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to each input."""
        return [output_grad, output_grad]


class AddByConstOp(Op):
    """Op to element-wise add a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}+{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise addition of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] + node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to the input."""
        return [output_grad]


class MulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}*{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise multiplication of input values."""
        assert len(input_values) == 2
        return input_values[0] * input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to each input."""
        return [output_grad * node.inputs[1], output_grad * node.inputs[0]]


class MulByConstOp(Op):
    """Op to element-wise multiply a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}*{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise multiplication of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] * node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to the input."""
        return [output_grad * node.constant]


class GreaterThanOp(Op):
    """Op to compare if node_A > node_B element-wise."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}>{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return element-wise comparison result as float tensor."""
        assert len(input_values) == 2
        return (input_values[0] > input_values[1]).float()

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Comparison operations have gradient of 0."""
        return [zeros_like(node.inputs[0]), zeros_like(node.inputs[1])]


class GreaterThanByConstOp(Op):
    """Op to compare if node_A > const_val element-wise."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}>{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return element-wise comparison result as float tensor."""
        assert len(input_values) == 1
        return (input_values[0] > node.constant).float()

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Comparison operations have gradient of 0."""
        return [zeros_like(node.inputs[0])]


class SubOp(Op):
    """Op to element-wise subtract two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}-{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise subtraction of input values."""
        assert len(input_values) == 2
        return input_values[0] - input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of subtraction node, return partial adjoint to each input."""
        return [output_grad, mul_by_const(output_grad, -1)]


class ZerosLikeOp(Op):
    """Zeros-like op that returns an all-zero array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"ZerosLike({node_A.name})")

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return an all-zero tensor with the same shape as input."""
        assert len(input_values) == 1
        return torch.zeros_like(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]


class OnesLikeOp(Op):
    """Ones-like op that returns an all-one array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"OnesLike({node_A.name})")

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return an all-one tensor with the same shape as input."""
        assert len(input_values) == 1
        return torch.ones_like(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]


class SumOp(Op):
    """
    Op to compute sum along specified dimensions.

    Note: This is a reference implementation for SumOp.
        If it does not work in your case, you can modify it.
    """

    def __call__(
        self,
        node_A: Node,
        dim: tuple[int, ...] | List[int] | int | None,
        keepdim: bool = False,
    ) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim, "keepdim": keepdim},
            name=f"Sum({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return input_values[0].sum(dim=node.dim, keepdim=node.keepdim)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        dim = node.attrs["dim"]
        keepdim = node.attrs["keepdim"]

        if keepdim:
            return [output_grad]
        else:
            reshape_grad = expand_as_3d(output_grad, node.inputs[0])
            return [reshape_grad]


class ExpandAsOp(Op):
    """Op to broadcast a tensor to the shape of another tensor.

    Note: This is a reference implementation for ExpandAsOp.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"broadcast({node_A.name} -> {node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 2
        input_tensor, target_tensor = input_values
        return input_tensor.expand_as(target_tensor)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the broadcast node, compute partial adjoint to input."""

        return [sum_op(output_grad, dim=0), zeros_like(output_grad)]


class ExpandAsOp3d(Op):
    """Op to broadcast a tensor to the shape of another tensor.

    Note: This is a reference implementation for ExpandAsOp3d.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"broadcast({node_A.name} -> {node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 2
        input_tensor, target_tensor = input_values
        print("expand_op", input_tensor.shape, target_tensor.shape)
        return input_tensor.unsqueeze(1).expand_as(target_tensor)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the broadcast node, compute partial adjoint to input."""

        return [sum_op(output_grad, dim=(0, 1)), zeros_like(output_grad)]


class LogOp(Op):
    """Logarithm (natural log) operation."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"Log({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the natural logarithm of the input."""
        assert len(input_values) == 1, "Log operation requires one input."
        return torch.log(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the Log node, return the partial adjoint to the input."""
        input_node = node.inputs[0]
        return [output_grad / input_node]


class BroadcastOp(Op):
    def __call__(
        self, node_A: Node, input_shape: List[int], target_shape: List[int]
    ) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"input_shape": input_shape, "target_shape": target_shape},
            name=f"Broadcast({node_A.name}, {target_shape})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 1
        return input_values[0].expand(node.attrs["target_shape"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of broadcast node, return partial adjoint to input.

        For broadcasting, we need to sum out the broadcasted dimensions to get
        back to the original shape.
        """
        if "input_shape" not in node.attrs:
            raise ValueError(
                "Input shape is not set. Make sure compute() is called before gradient()"
            )

        input_shape = node.attrs["input_shape"]
        output_shape = node.attrs["target_shape"]

        dims_to_sum: List[int] = []
        for i, (in_size, out_size) in enumerate(
            zip(input_shape[::-1], output_shape[::-1])
        ):
            if in_size != out_size:
                dims_to_sum.append(len(output_shape) - 1 - i)

        grad = output_grad
        if dims_to_sum:
            grad = sum_op(grad, dim=dims_to_sum, keepdim=True)

        if len(output_shape) > len(input_shape):
            grad = sum_op(
                grad,
                dim=list(range(len(output_shape) - len(input_shape))),
                keepdim=False,
            )

        return [grad]


class DivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}/{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise division of input values."""
        assert len(input_values) == 2
        return input_values[0] / input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to each input."""
        node_A, node_B = node.inputs
        return [
            output_grad / node_B,
            mul_by_const(output_grad * node_A / power(node_B, 2), -1),
        ]


class DivByConstOp(Op):
    """Op to element-wise divide a nodes by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}/{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise division of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] / node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to the input."""
        return [output_grad / node.constant]


class TransposeOp(Op):
    """Op to transpose a matrix."""

    def __call__(self, node_A: Node, dim0: int, dim1: int) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim0": dim0, "dim1": dim1},
            name=f"transpose({node_A.name}, {dim0}, {dim1})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the transpose of the input by swapping two dimensions.

        For example:
        - transpose(x, 1, 0) swaps first two dimensions
        """
        assert len(input_values) == 1
        dim0 = node.attrs["dim0"]
        dim1 = node.attrs["dim1"]
        return input_values[0].transpose(dim0, dim1)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of transpose node, return partial adjoint to input."""
        dim0 = node.attrs["dim0"]
        dim1 = node.attrs["dim1"]
        return [transpose(output_grad, dim0, dim1)]


class MatMulOp(Op):
    """Matrix multiplication op of two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        """Create a matrix multiplication node.

        Parameters
        ----------
        node_A: Node
            The lhs matrix.
        node_B: Node
            The rhs matrix

        Returns
        -------
        result: Node
            The node of the matrix multiplication.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the matrix multiplication result of input values."""
        assert len(input_values) == 2
        return input_values[0] @ input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of matmul node, return partial adjoint to each input."""
        node_A, node_B = node.inputs  # (..., m, k) @ (..., k, n) = (..., m, n)
        return [
            matmul(
                output_grad, transpose(node_B, -2, -1)
            ),  # (..., m, n) @ (..., n, k) = (..., m, k)
            matmul(
                transpose(node_A, -2, -1), output_grad
            ),  # (..., k, m) @ (..., m, n) = (..., k, n)
        ]


class SoftmaxOp(Op):
    """Softmax operation on input node."""

    def __call__(self, node_A: Node, dim: int = -1) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim},
            name=f"Softmax({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return softmax of input along specified dimension."""
        assert len(input_values) == 1
        dim = node.attrs["dim"]
        x = input_values[0]
        return softmax_pytorch(x, dim=dim)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of softmax node, return partial adjoint to input.

        softmax(xi) = exp(xi) / sum(exp(xj))

        Let f = exp(xi), g = 1 / sum(exp(xj))

        dL/d_xi = sum { dL/d_softmax(xj) * d_softmax(xj)/d_xi }

        d_softmax(xj)/d_xi = d_softmax(xj)/df * df/d_xi + d_softmax(xj)/dg * dg/d_xi
        
        d_softmax(xj)/df = g, 
        df/d_xi = exp(xj) * delta, where delta = 1 if i == j else 0

        ==> d_softmax(xj)/df * df/d_xi = exp(xi) * delta * g = softmax(xi) * delta

        d_softmax(xj)/dg = - exp(xj) / (s ^ 2)
        dg/d_xi = exp(xi)

        ==> d_softmax(xj)/dg * dg/d_xi = - exp(xj) * exp(xi) / (s ^ 2)
                                       = - softmax(xi) * softmax(xj)


        ==> d_softmax(xj)/d_xi = softmax(xi) * delta - softmax(xi) * softmax(xj)
                               = softmax(xi) * (delta - softmax(xj))

        ==> sum_j { dL/d_softmax(xj) * d_softmax(xj)/d_xi } 
          = sum_j { dL/d_softmax(xj) * softmax(xi) * (delta - softmax(xj)) }
          = softmax(xi) * sum_j { delta(xj) * dL/d_softmax(xj) } - softmax(xi) * sum_j { dL/d_softmax(xj) * softmax(xj) }
          = softmax(xi) * dL/d_softmax(xi) - softmax(xi) * sum(output_grad * softmax(x))
          = softmax(xi) * (dL/d_softmax(xi) - sum(output_grad * softmax(x)))
        """
        x = node.inputs[0]
        dim = node.attrs["dim"]

        # Compute softmax of x
        x_exp = exp(x)
        s = sum_op(x_exp, dim=dim, keepdim=True)
        s = expand_as(s, x_exp)
        softmax_x = x_exp / s

        grad_mult_softmax_x_sum = sum_op(output_grad * softmax_x, dim=dim, keepdim=True)
        grad_mult_softmax_x_sum = expand_as(grad_mult_softmax_x_sum, x)

        return [softmax_x * (output_grad - grad_mult_softmax_x_sum)]


class LayerNormOp(Op):
    """Layer normalization operation."""

    def __call__(self, node_A: Node, dim: tuple[int], eps: float = 1e-5) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim, "eps": eps},
            name=f"LayerNorm({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return layer normalized input."""
        assert len(input_values) == 1

        x = input_values[0]
        dim: tuple[int] = node.attrs["dim"]
        eps: float = node.attrs["eps"]

        # Compute mean and variance
        mu = x.mean(dim=dim, keepdim=True)
        var = x.var(dim=dim, keepdim=True, unbiased=False)
        x_normalized = (x - mu) / torch.sqrt(var + eps)
        return x_normalized

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """
        Given gradient of the LayerNorm node wrt its output, return partial
        adjoint (gradient) wrt the input x.
        Uses the formula derived from chain rule:
        dL/dx_i = (1/sigma) * (dL/dy_i - mean(dL/dy) - y_i * mean(dL/dy * y))
        where y = (x - mu) / sigma is the normalized output.
        """
        x = node.inputs[0]
        dim: tuple[int] = node.attrs["dim"]
        eps: float = node.attrs["eps"]

        # Recompute necessary forward values using autograd ops
        mu = mean(x, dim=dim, keepdim=True)
        # Expand mu to match x's shape for subtraction
        mu = expand_as(mu, x)

        # Calculate variance and standard deviation (sigma)
        variance = var(x, dim=dim, keepdim=True)
        sigma = sqrt(variance + eps)
        # Expand sigma to match x's shape for division
        sigma = expand_as(sigma, x)

        # Recompute normalized output y (x_norm)
        x_norm = (x - mu) / sigma

        # Calculate terms needed for the gradient formula
        # 1. mean(dL/dy)
        grad_mean = mean(output_grad, dim=dim, keepdim=True)
        grad_mean = expand_as(grad_mean, x)

        # 2. mean(dL/dy * y)
        grad_mult_x_norm_mean = mean(output_grad * x_norm, dim=dim, keepdim=True)
        grad_mult_x_norm_mean = expand_as(grad_mult_x_norm_mean, x)

        # Combine terms: dL/dy_i - mean(dL/dy) - y_i * mean(dL/dy * y)
        combined_grads = (
            output_grad - grad_mean - x_norm * grad_mult_x_norm_mean
        )

        # Final gradient: (1 / sigma) * combined_grads
        return [(sigma**-1) * combined_grads]


class ReLUOp(Op):
    """ReLU activation function."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"ReLU({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return ReLU of input."""
        assert len(input_values) == 1
        return torch.relu(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of ReLU node, return partial adjoint to input."""
        input_node = node.inputs[0]
        grad = output_grad * (input_node > 0)
        return [grad]


class SqrtOp(Op):
    """Op to compute element-wise square root."""

    def __call__(self, node: Node) -> Node:
        return Node(
            inputs=[node],
            op=self,
            name=f"Sqrt({node.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return torch.sqrt(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of sqrt node, return partial adjoint to input."""
        input_node = node.inputs[0]
        grad = output_grad / (2 * sqrt(input_node))
        return [grad]


class PowerOp(Op):
    """Op to compute element-wise power."""

    def __call__(self, node_A: Node, exponent: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"exponent": exponent},
            name=f"Power({node_A.name}, {exponent})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return input_values[0] ** node.attrs["exponent"]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of power node, return partial adjoint to input."""
        input_node = node.inputs[0]
        exponent = node.attrs["exponent"]
        grad = output_grad * exponent * (input_node ** (exponent - 1))
        return [grad]


class ExpOp(Op):
    """Op to compute element-wise exponentiation."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"Exp({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return torch.exp(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of exp node, return partial adjoint to input."""
        x = node.inputs[0]
        return [output_grad * exp(x)]

class MeanOp(Op):
    """Op to compute mean along specified dimensions."""

    def __call__(self, node_A: Node, dim: tuple, keepdim: bool = False) -> Node:
        assert all(d >= 0 for d in dim), "Dimensions must be non-negative."
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim, "keepdim": keepdim},
            name=f"Mean({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return input_values[0].mean(
            dim=node.attrs["dim"], keepdim=node.attrs["keepdim"]
        )

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of mean node, return partial adjoint to the input."""
        node_A = node.inputs[0]
        dim = node.attrs["dim"]
        keepdim = node.attrs["keepdim"]

        d = count_over_dim(node_A, dim)  # total number of elements in the dimension
        expanded_d = expand_as(
            d, output_grad
        )

        grad = output_grad / expanded_d

        if not keepdim:
            for d in sorted(list(dim)):
                grad = unsqueeze(grad, d)

        return [expand_as(grad, node_A)]


class VarOp(Op):
    """Op to compute variance along specified dimensions."""

    def __call__(self, node_A: Node, dim: tuple[int, ...], keepdim: bool) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim, "keepdim": keepdim},
            name=f"Var({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        dim = node.attrs["dim"]
        keepdim = node.attrs["keepdim"]

        return input_values[0].var(dim=dim, unbiased=False, keepdim=keepdim)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of var node, return partial adjoint to input."""

        x = node.inputs[0]
        dim = node.attrs["dim"]
        keepdim = node.attrs["keepdim"]

        # Calculate N (count over dimensions)
        n = count_over_dim(x, dim)
        n = expand_as(n, x)

        # Calculate mean, keeping dimensions initially for correct expansion
        mu = mean(x, dim, keepdim=True)
        mu = expand_as(mu, x)

        # Expand the incoming gradient to the shape of x
        if not keepdim:
            for d in sorted(list(dim)):
                output_grad = unsqueeze(output_grad, d)

        output_grad = expand_as(output_grad, x)

        return [output_grad * ((x - mu) * 2) / n]


class CountOp(Op):
    """Op to count the number of elements given dimensions."""

    def __call__(self, node_A: Node, dim: tuple[int, ...] | int | None = None) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim},
            name=f"Count({node_A.name}, {dim})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        dim: tuple[int, ...] | int | None = node.attrs["dim"]
        shape = input_values[0].shape

        if dim is None:
            # If dim is None, return the total number of elements
            total = reduce(mul_py, shape, 1)
            return torch.tensor(total, dtype=torch.float32)

        if isinstance(dim, int):
            dim = (dim,)

        total = reduce(mul_py, [shape[d] for d in dim], 1)
        return torch.tensor(total, dtype=torch.float32)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of count node, return partial adjoint to input."""
        # Count operation is not differentiable, so return zeros_like
        return [zeros_like(node.inputs[0])]


class UnsqueezeOp(Op):
    """Op to unsqueeze a tensor along specified dimensions."""

    def __call__(self, node_A: Node, dim: int) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim},
            name=f"Unsqueeze({node_A.name}, {dim})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1

        dim = node.attrs["dim"]
        return input_values[0].unsqueeze(dim)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of unsqueeze node, return partial adjoint to input."""
        return [squeeze(output_grad, dim=node.attrs["dim"])]


class SqueezeOp(Op):
    """Op to squeeze a tensor along specified dimensions."""

    def __call__(self, node_A: Node, dim: int) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim},
            name=f"Squeeze({node_A.name}, {dim})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1

        dim = node.attrs["dim"]
        return input_values[0].squeeze(dim)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of squeeze node, return partial adjoint to input."""
        return [unsqueeze(output_grad, dim=node.attrs["dim"])]


# Create global instances of ops.
# Your implementation should just use these instances, rather than creating new instances.
placeholder = PlaceholderOp()
add = AddOp()
mul = MulOp()
div = DivOp()
add_by_const = AddByConstOp()
mul_by_const = MulByConstOp()
div_by_const = DivByConstOp()
matmul = MatMulOp()
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()
softmax = SoftmaxOp()
layernorm = LayerNormOp()
relu = ReLUOp()
transpose = TransposeOp()
mean = MeanOp()
exp = ExpOp()
var = VarOp()
sum_op = SumOp()
sqrt = SqrtOp()
power = PowerOp()
greater = GreaterThanOp()
greater_by_const = GreaterThanByConstOp()
expand_as = ExpandAsOp()
expand_as_3d = ExpandAsOp3d()
log = LogOp()
sub = SubOp()
count_over_dim = CountOp()
broadcast = BroadcastOp()
squeeze = SqueezeOp()
unsqueeze = UnsqueezeOp()


class Evaluator:
    """The node evaluator that computes the values of nodes in a computational graph."""

    eval_nodes: List[Node]

    def __init__(self, eval_nodes: List[Node]) -> None:
        """Constructor, which takes the list of nodes to evaluate in the computational graph.

        Parameters
        ----------
        eval_nodes: List[Node]
            The list of nodes whose values are to be computed.
        """
        self.eval_nodes = eval_nodes

    def run(self, input_values: Dict[Node, torch.Tensor]) -> List[torch.Tensor]:
        """Computes values of nodes in `eval_nodes` field with
        the computational graph input values given by the `input_values` dict.

        Parameters
        ----------
        input_values: Dict[Node, torch.Tensor]
            The dictionary providing the values for input nodes of the
            computational graph.
            Throw ValueError when the value of any needed input node is
            not given in the dictionary.

        Returns
        -------
        eval_values: List[torch.Tensor]
            The list of values for nodes in `eval_nodes` field.
        """

        node_values: Dict[Node, torch.Tensor] = {}

        def compute(node: Node) -> torch.Tensor:
            if node in node_values:
                return node_values[node]

            if node in input_values:
                node_values[node] = input_values[node]
                return node_values[node]

            values = []
            for node_input in node.inputs:
                if node_input not in node_values:
                    raise ValueError(
                        f"Input node {node_input} not found in input values."
                    )
                values.append(node_values[node_input])

            node_values[node] = node.op.compute(node, values)
            return node_values[node]

        graph_nodes = explore_graph(self.eval_nodes)
        for node in reverse_topological_sort(graph_nodes):
            compute(node)

        return [node_values[node] for node in self.eval_nodes]


def explore_graph(out_nodes: List[Node]) -> List[Node]:
    visited: Set[Node] = set()
    nodes: List[Node] = []

    def dfs(node: Node):
        if node in visited:
            return

        visited.add(node)

        if not isinstance(node.op, PlaceholderOp):
            for node_input in node.inputs:
                dfs(node_input)

        nodes.append(node)

    for node in out_nodes:
        dfs(node)

    return nodes


def reverse_topological_sort(nodes: List[Node]) -> List[Node]:
    """Helper function to perform topological sort on nodes.

    Parameters
    ----------
    nodes : List[Node] or Node
        Node(s) to sort

    Returns
    -------
    List[Node]
        Nodes in topological order
    """
    sorted_nodes = []
    visited = set()

    def visit(node: Node):
        if node in visited:
            return

        visited.add(node)

        if not isinstance(node.op, PlaceholderOp):
            for input_node in node.inputs:
                visit(input_node)

        sorted_nodes.append(node)

    for node in nodes:
        visit(node)

    return sorted_nodes


def gradients(output_node: Node, nodes: List[Node]) -> List[Node]:
    """Construct the backward computational graph, which takes gradient
    of given output node with respect to each node in input list.
    Return the list of gradient nodes, one for each node in the input list.

    Parameters
    ----------
    output_node: Node
        The output node to take gradient of, whose gradient is 1.

    nodes: List[Node]
        The list of nodes to take gradient with regard to.

    Returns
    -------
    grad_nodes: List[Node]
        A list of gradient nodes, one for each input nodes respectively.
    """

    node_to_grad: Dict[Node, Node] = {}
    all_nodes = explore_graph([output_node])

    node_to_grad[output_node] = ones_like(output_node)
    for node in reverse_topological_sort(all_nodes)[::-1]:
        if node not in node_to_grad:
            raise ValueError(f"Node {node} is not in the gradient computation graph.")

        if isinstance(node.op, PlaceholderOp):
            continue

        node_to_grad[node].name = f"grad_({node.name})"
        for node_input, node_adjoint in zip(
            node.inputs, node.op.gradient(node, node_to_grad[node])
        ):
            if node_input not in node_to_grad:
                node_to_grad[node_input] = node_adjoint
            else:
                node_to_grad[node_input] += node_adjoint

    return [node_to_grad[node] for node in nodes]
