import sys
from pathlib import Path
from typing import List

import torch

sys.path.append(str(Path(__file__).parent.parent))
import auto_diff as ad


def check_compute_output(
    node: ad.Node, input_values: List[torch.Tensor], expected_output: torch.Tensor
) -> None:
    output = node.op.compute(node, input_values)
    torch.testing.assert_close(output, expected_output, atol=1e-4, rtol=1e-4)


def test_mul():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.mul(x1, x2)

    check_compute_output(
        y,
        [
            torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            torch.tensor([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
        ],
        torch.tensor([[-2.80, 1.40, -0.05, 0.00], [0.18, 0.00, -18.56, 9.61]]),
    )


def test_mul_by_const():
    x1 = ad.Variable("x1")
    y = ad.mul_by_const(x1, 2.7)

    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])],
        torch.tensor([[-2.70, 5.40, 1.35, 9.18], [0.81, 0.00, -15.66, 8.37]]),
    )


def test_div():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.div(x1, x2)

    check_compute_output(
        y,
        [
            torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            torch.tensor([[2.5, 4.0, -0.1, 0.1], [-8.0, 5.0, -2.5, -1.0]]),
        ],
        torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
    )


def test_div_by_const():
    x1 = ad.Variable("x1")
    y = ad.div_by_const(x1, 5.0)

    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])],
        torch.tensor([[-0.2, 0.4, 0.1, 0.68], [0.06, 0.0, -1.16, 0.62]]),
    )


def test_matmul():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.matmul(x1, x2)

    x1_val = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x2_val = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    check_compute_output(
        y,
        [x1_val, x2_val],
        torch.tensor([[27.0, 30.0, 33.0], [61.0, 68.0, 75.0], [95.0, 106.0, 117.0]]),
    )


def test_matmul_3d():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.matmul(x1, x2)

    x1_val = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
        ]
    )

    x2_val = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
        ]
    )

    expected = torch.tensor(
        [
            [[30.0, 36.0, 42.0], [66.0, 81.0, 96.0], [102.0, 126.0, 150.0]],
            [[150.0, 126.0, 102.0], [96.0, 81.0, 66.0], [42.0, 36.0, 30.0]],
        ]
    )

    check_compute_output(y, [x1_val, x2_val], expected)


def test_layernorm():
    x = ad.Variable("x")
    y = ad.layernorm(x, dim=(1,))

    check_compute_output(
        y,
        [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)],
        torch.tensor(
            [[-1.224745, 0.0, 1.224745], [-1.224745, 0.0, 1.224745]],
            dtype=torch.float32,
        ),
    )


def test_relu():
    x = ad.Variable("x")
    y = ad.relu(x)

    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.0], [3.0, -4.0, 5.0]], dtype=torch.float32)],
        torch.tensor([[0.0, 2.0, 0.0], [3.0, 0.0, 5.0]], dtype=torch.float32),
    )


def test_transpose():
    x = ad.Variable("x")
    y = ad.transpose(x, 1, 0)

    check_compute_output(
        y,
        [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])],
        torch.tensor([[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [7.0, 8.0]]]),
    )


def test_softmax():
    x = ad.Variable("x")
    y = ad.softmax(x)

    check_compute_output(
        y,
        [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)],
        torch.tensor(
            [[0.0900, 0.2447, 0.6652], [0.0900, 0.2447, 0.6652]], dtype=torch.float32
        ),
    )


def test_broadcast():
    x = ad.Variable("x")
    y = ad.broadcast(x, input_shape=[3, 2], target_shape=[2, 3, 2])

    check_compute_output(
        y,
        [torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])],
        torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]
        ),
    )


def test_sqrt():
    x = ad.Variable("x")
    y = ad.sqrt(x)

    check_compute_output(
        y,
        [torch.tensor([[4.0, 9.0], [16.0, 25.0]], dtype=torch.float32)],
        torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float32),
    )


def test_power():
    x = ad.Variable("x")
    y = ad.power(x, 2)

    check_compute_output(
        y,
        [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)],
        torch.tensor([[1.0, 4.0], [9.0, 16.0]], dtype=torch.float32),
    )


def test_mean():
    x = ad.Variable("x")
    y = ad.mean(x, dim=(1,), keepdim=True)
    z = ad.mean(x, dim=(1,), keepdim=False)
    complex_t = torch.randn(3, 4, 5, 6, dtype=torch.float32)

    check_compute_output(
        y,
        [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)],
        torch.tensor([[1.5], [3.5]], dtype=torch.float32),
    )

    check_compute_output(
        z,
        [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)],
        torch.tensor([1.5, 3.5], dtype=torch.float32),
    )

    check_compute_output(
        ad.mean(x, dim=(1, 2), keepdim=True),
        [complex_t],
        complex_t.mean(dim=(1, 2), keepdim=True),
    )
    check_compute_output(
        ad.mean(x, dim=(1, 2), keepdim=False),
        [complex_t],
        complex_t.mean(dim=(1, 2), keepdim=False),
    )


def test_count():
    x = ad.Variable("x")
    y = ad.count_over_dim(x, dim=None)
    z = ad.count_over_dim(x, dim=(1, 2))
    w = ad.count_over_dim(x, dim=1)

    check_compute_output(
        y,
        [torch.randn(2, 3, 4, dtype=torch.float32)],
        torch.tensor(2 * 3 * 4, dtype=torch.float32),
    )

    check_compute_output(
        z,
        [torch.randn(2, 3, 4, dtype=torch.float32)],
        torch.tensor(3 * 4, dtype=torch.float32),
    )

    check_compute_output(
        w,
        [torch.randn(2, 3, 4, dtype=torch.float32)],
        torch.tensor(3, dtype=torch.float32),
    )


def test_unsqueeze():
    x = ad.Variable("x")
    y = ad.unsqueeze(x, dim=2)

    t = torch.randn(2, 3, 1, 4, dtype=torch.float32)

    check_compute_output(y, [t], t.unsqueeze(2))


def test_squeeze():
    x = ad.Variable("x")
    y = ad.squeeze(x, dim=2)

    t = torch.randn(2, 3, 4, dtype=torch.float32)

    check_compute_output(y, [t], t.squeeze(2))


def test_var():
    x = ad.Variable("x")
    # Test with keepdim=True
    y = ad.var(x, dim=(1,), keepdim=True)
    # Test with keepdim=False
    z = ad.var(x, dim=(1,), keepdim=False)
    # Test with multiple dimensions
    w = ad.var(x, dim=(1, 2), keepdim=True)

    # Simple tensor with known values
    t1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)

    # Complex tensor for multiple dimension testing
    t2 = torch.randn(2, 3, 4, 5, dtype=torch.float32)

    check_compute_output(y, [t1], t1.var(dim=1, keepdim=True, unbiased=False))

    check_compute_output(z, [t1], t1.var(dim=1, keepdim=False, unbiased=False))

    check_compute_output(w, [t2], t2.var(dim=(1, 2), keepdim=True, unbiased=False))


if __name__ == "__main__":
    test_mul()
    test_mul_by_const()
    test_div()
    test_div_by_const()
    test_layernorm()
    test_relu()
    test_softmax()
    test_matmul()
    test_mat
