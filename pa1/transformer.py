from enum import Enum, auto
from math import sqrt
from typing import Callable, Dict, List, Tuple, cast

import auto_diff as ad
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from torchvision import datasets, transforms

max_len = 28


class NodeType(Enum):
    # Linear layer
    LINEAR_W = auto()
    LINEAR_B = auto()
    # Attention layer
    ATTENTION_Q = auto()
    ATTENTION_K = auto()
    ATTENTION_V = auto()
    ATTENTION_O = auto()
    # Feed-forward layer
    FFN_W = auto()
    FFN_B = auto()


def linear(
    batch_size: int,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    X: ad.Node,
    W: ad.Node,
    b: ad.Node | None = None,
) -> ad.Node:
    """
    Linear layer.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, in_dim, hidden_dim), denoting the input data.
    W: ad.Node
        A node in shape (hidden_dim, out_dim), denoting the weight matrix.
    b: ad.Node
        A node in shape (out_dim,), denoting the bias vector.

    Returns
    -------
    output: ad.Node
        A node in shape (batch_size, seq_length, out_dim), denoting the output of the linear layer.
    """

    W = ad.unsqueeze(W, 0)  # (1, hidden_dim, out_dim)
    W = ad.broadcast(
        W,
        input_shape=[1, hidden_dim, out_dim],
        target_shape=[batch_size, hidden_dim, out_dim],
    )  # (batch_size, hidden_dim, out_dim)

    if b is None:
        return ad.matmul(X, W)  # (batch_size, in_dim, out_dim)

    b = ad.unsqueeze(ad.unsqueeze(b, 0), 0)  # (1, 1, out_dim)
    b = ad.broadcast(
        b,
        input_shape=[1, 1, out_dim],
        target_shape=[batch_size, in_dim, out_dim],
    )  # (batch_size, in_dim, out_dim)

    return ad.matmul(X, W) + b


def single_head_attention(
    X: ad.Node,
    W_q: ad.Node,
    W_k: ad.Node,
    W_v: ad.Node,
    W_o: ad.Node,
    batch_size: int,
    seq_length: int,  # 28
    model_dim: int,
) -> ad.Node:
    """
    Single head attention.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, model_dim), denoting the input data.
    W_q, W_k, W_v, W_o: ad.Node
        A node in shape (model_dim, model_dim), denoting the weight matrix for Q, K, V, and O.

    Returns
    -------
    output: ad.Node
        A node in shape (batch_size, seq_length, model_dim), denoting the output of the single head attention.
    """

    Q = linear(
        batch_size, seq_length, seq_length, model_dim, X, W_q
    )  # (batch_size, seq_length, model_dim)
    K = linear(
        batch_size, seq_length, seq_length, model_dim, X, W_k
    )  # (batch_size, seq_length, model_dim)
    V = linear(
        batch_size, seq_length, seq_length, model_dim, X, W_v
    )  # (batch_size, seq_length, model_dim)

    out = ad.matmul(Q, ad.transpose(K, 1, 2)) / sqrt(
        model_dim
    )  # (batch_size, seq_length, seq_length)
    out = ad.softmax(out, dim=2)  # (batch_size, seq_length, seq_length)
    out = ad.matmul(out, V)  # (batch_size, seq_length, model_dim)
    out = linear(batch_size, seq_length, model_dim, model_dim, out, W_o)

    return out  # (batch_size, seq_length, model_dim)


def transformer(
    X: ad.Node,
    nodes: Dict[NodeType, ad.Node],
    input_dim: int,
    model_dim: int,
    seq_length: int,
    eps,
    batch_size,
    num_classes,
) -> ad.Node:
    """Construct the computational graph for a single transformer layer with sequence classification.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, input_dim, input_dim), denoting the input data.
    nodes: List[ad.Node]
        Nodes you would need to initialize the transformer.
    model_dim: int
        Dimension of the model (hidden size).
    seq_length: int
        Length of the input sequence.

    Returns
    -------
    output: ad.Node
        The output of the transformer layer, averaged over the sequence length for classification, in shape (batch_size, num_classes).
    """

    # Single head attention
    x = single_head_attention(
        X,
        nodes[NodeType.ATTENTION_Q],
        nodes[NodeType.ATTENTION_K],
        nodes[NodeType.ATTENTION_V],
        nodes[NodeType.ATTENTION_O],
        batch_size,
        seq_length,
        model_dim,
    )  # (batch_size, seq_length, model_dim)

    # Residual connection
    x = x + ad.layernorm(x, dim=(2,), eps=eps)

    # Feed-forward network
    x = linear(
        batch_size,
        seq_length,
        model_dim,
        model_dim,
        x,
        nodes[NodeType.LINEAR_W],
        nodes[NodeType.LINEAR_B],
    )  # (batch_size, seq_length, model_dim)

    # Residual connection
    x = x + ad.layernorm(x, dim=(2,), eps=eps)

    # Linear layer for classification
    x = linear(
        batch_size,
        seq_length,
        model_dim,
        num_classes,
        x,
        nodes[NodeType.FFN_W],
        nodes[NodeType.FFN_B],
    )  # (batch_size, seq_length, num_classes)

    return ad.mean(x, dim=(1,))  # (batch_size, num_classes)


def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        logits for the batch of instances.

    y_one_hot: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        one-hot encoding of the ground truth label for the batch of instances.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch.
        When evaluating, it should be a zero-rank array (i.e., shape is `()`).

    Note
    ----
    1. In this homework, you do not have to implement a numerically
    stable version of softmax loss.
    2. You may find that in other machine learning frameworks, the
    softmax loss function usually does not take the batch size as input.
    Try to think about why our softmax loss may need the batch size.
    """
    softmax_logits = ad.softmax(Z, dim=1)  # (batch_size, num_classes)
    log_softmax_logits = ad.log(softmax_logits)  # (batch_size, num_classes)
    loss = ad.sum_op(y_one_hot * log_softmax_logits, dim=(1,)) / batch_size  # ()
    return loss * -1


def sgd_epoch(
    f_run_model: Callable[
        [torch.Tensor, torch.Tensor, Dict[NodeType, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]],
    ],
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: Dict[NodeType, torch.Tensor],
    batch_size: int,
    lr: float,
) -> Tuple[Dict[NodeType, torch.Tensor], torch.Tensor]:
    """Run an epoch of SGD for the logistic regression model
    on training data with regard to the given mini-batch size
    and learning rate.

    Parameters
    ----------
    f_run_model: Callable
        The function to run the forward and backward computation
        at the same time for logistic regression model.
        It takes the training data, training label, model weight
        and bias as inputs, and returns the logits, loss value,
        weight gradient and bias gradient in order.
        Please check `f_run_model` in the `train_model` function below.

    X: torch.Tensor
        The training data in shape (num_examples, in_features).

    y: torch.Tensor
        The training labels in shape (num_examples,).

    model_weights: List[torch.Tensor]
        The model weights in the model.

    batch_size: int
        The mini-batch size.

    lr: float
        The learning rate.

    Returns
    -------
    model_weights: List[torch.Tensor]
        The model weights after update in this epoch.

    loss: torch.Tensor
        The average training loss of this epoch.
    """

    """TODO: Your code here"""
    num_examples = X.shape[0]
    num_batches = (
        num_examples + batch_size - 1
    ) // batch_size  # Compute the number of batches
    total_loss: torch.Tensor = torch.tensor(0.0)

    for i in range(num_batches):
        # Get the mini-batch data
        start_idx = i * batch_size
        if start_idx + batch_size > num_examples:
            continue

        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx, :max_len]
        y_batch = y[start_idx:end_idx]

        # Compute forward and backward passes
        _, loss, weight_grads = f_run_model(X_batch, y_batch, model_weights)
        idx_to_node_type = {
            i: node_type for i, node_type in enumerate(model_weights.keys())
        }

        # Update weights and biases
        # Hint: You can update the tensor using something like below:
        # W_Q -= lr * grad_W_Q.sum(dim=0)
        for i, w_grad in enumerate(weight_grads):
            node_type = idx_to_node_type[i]
            w_shape, w_grad_shape = (
                model_weights[node_type].shape,
                w_grad.shape,
            )

            assert w_shape == w_grad_shape, f"{node_type} {w_shape} != {w_grad_shape}"
            model_weights[node_type] -= lr * w_grad

        # Accumulate the loss
        total_loss += loss.sum()

    # Compute the average loss
    average_loss = total_loss / num_examples
    print("Avg_loss:", average_loss)

    # You should return the list of parameters and the loss
    return model_weights, average_loss


def train_model():
    """Train a logistic regression model with handwritten digit dataset.

    Note
    ----
    Your implementation should NOT make changes to this function.
    """
    # Set up model params

    # TODO: Tune your hyperparameters here
    # Hyperparameters
    input_dim = 28  # Each row of the MNIST image
    seq_length = max_len  # Number of rows in the MNIST image
    num_classes = 10  #
    model_dim = 128  #
    eps = 1e-5

    # Set up the training settings.
    num_epochs = 20
    batch_size = 50
    lr = 0.02

    # Define the forward graph.
    input_x = ad.Variable(name="x")
    y_groundtruth = ad.Variable(name="y")

    print(f"batch_size: {batch_size}, lr: {lr}, num_epochs: {num_epochs}")

    model_weight_nodes: Dict[NodeType, ad.Node] = {
        NodeType.LINEAR_W: ad.Variable(name="linear_w"),
        NodeType.LINEAR_B: ad.Variable(name="linear_b"),
        NodeType.ATTENTION_Q: ad.Variable(name="attn_q"),
        NodeType.ATTENTION_K: ad.Variable(name="attn_k"),
        NodeType.ATTENTION_V: ad.Variable(name="attn_v"),
        NodeType.ATTENTION_O: ad.Variable(name="attn_o"),
        NodeType.FFN_W: ad.Variable(name="ffn_w"),
        NodeType.FFN_B: ad.Variable(name="ffn_b"),
    }

    y_predict = transformer(
        input_x,
        model_weight_nodes,
        input_dim,
        model_dim,
        seq_length,
        eps,
        batch_size,
        num_classes,
    )
    loss = softmax_loss(y_predict, y_groundtruth, batch_size)

    # Construct the backward graph.
    model_weight_node_list = list(model_weight_nodes.values())
    grads = ad.gradients(loss, model_weight_node_list)

    # Create the evaluator.
    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])

    # Load the dataset.
    # Take 80% of data for training, and 20% for testing.
    # Prepare the MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )

    # Convert the train dataset to NumPy arrays
    X_train = (
        train_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    )  # Flatten to 784 features
    y_train = train_dataset.targets.numpy()

    # Convert the test dataset to NumPy arrays
    X_test = (
        test_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    )  # Flatten to 784 features
    y_test = test_dataset.targets.numpy()

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(
        sparse_output=False
    )  # Use sparse=False to get a dense array

    # Fit and transform y_train, and transform y_test
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    num_classes = 10

    # Initialize model weights.
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    W_Q_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_K_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_V_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_O_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_1_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_2_val = np.random.uniform(-stdv, stdv, (model_dim, num_classes))
    b_1_val = np.random.uniform(-stdv, stdv, (model_dim,))
    b_2_val = np.random.uniform(-stdv, stdv, (num_classes,))

    model_weights: Dict[NodeType, torch.Tensor] = {
        NodeType.LINEAR_W: torch.tensor(W_1_val),
        NodeType.LINEAR_B: torch.tensor(b_1_val),
        NodeType.ATTENTION_Q: torch.tensor(W_Q_val),
        NodeType.ATTENTION_K: torch.tensor(W_K_val),
        NodeType.ATTENTION_V: torch.tensor(W_V_val),
        NodeType.ATTENTION_O: torch.tensor(W_O_val),
        NodeType.FFN_W: torch.tensor(W_2_val),
        NodeType.FFN_B: torch.tensor(b_2_val),
    }

    def f_run_model(
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        model_weights: Dict[NodeType, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """The function to compute the forward and backward graph.

        It returns the logits, loss, and gradients for model weights.
        """

        grad_inputs = {
            node: model_weights[node_type]
            for node_type, node in model_weight_nodes.items()
        }

        logits, loss, *grads = evaluator.run(
            input_values={
                **grad_inputs,
                input_x: X_train,
                y_groundtruth: y_train,
            }
        )
        return logits, loss, grads

    def f_eval_model(X_val: torch.Tensor, model_weights: Dict[NodeType, torch.Tensor]):
        """The function to compute the forward graph only and returns the prediction."""
        num_examples = X_val.shape[0]
        num_batches = (
            num_examples + batch_size - 1
        ) // batch_size  # Compute the number of batches

        total_loss = 0.0
        all_logits = []
        for i in range(num_batches):
            # Get the mini-batch data
            start_idx = i * batch_size
            if start_idx + batch_size > num_examples:
                continue

            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]

            grad_inputs = {
                node: model_weights[node_type]
                for node_type, node in model_weight_nodes.items()
            }

            logits = test_evaluator.run(
                input_values={
                    **grad_inputs,
                    input_x: X_batch,
                }
            )
            all_logits.append(logits[0])

        # Concatenate all logits and return the predicted classes
        concatenated_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions

    # Train the model.
    X_train, X_test, y_train, y_test = (
        torch.tensor(X_train),
        torch.tensor(X_test),
        torch.DoubleTensor(y_train),
        torch.DoubleTensor(y_test),
    )

    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)  # type: ignore
        X_train = cast(torch.Tensor, X_train)
        y_train = cast(torch.Tensor, y_train)

        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, model_weights, batch_size, lr
        )

        # Evaluate the model on the test data.
        predict_label = f_eval_model(X_test, model_weights)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label== y_test.numpy())}, "
            f"loss = {loss_val}"
        )

    # Return the final test accuracy.
    predict_label = f_eval_model(X_test, model_weights)
    return np.mean(predict_label == y_test.numpy())


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")


"""
Avg_loss: tensor(0.0513)
Epoch 0: test accuracy = 0.3615, loss = 0.051286373287439346
Avg_loss: tensor(0.0341)
Epoch 1: test accuracy = 0.4302, loss = 0.03409101814031601
Avg_loss: tensor(0.0318)
Epoch 2: test accuracy = 0.4198, loss = 0.031806014478206635
Avg_loss: tensor(0.0307)
Epoch 3: test accuracy = 0.4654, loss = 0.03065194934606552
Avg_loss: tensor(0.0299)
Epoch 4: test accuracy = 0.4831, loss = 0.029914211481809616
Avg_loss: tensor(0.0293)
Epoch 5: test accuracy = 0.4865, loss = 0.029279977083206177
Avg_loss: tensor(0.0288)
Epoch 6: test accuracy = 0.4895, loss = 0.028813334181904793
Avg_loss: tensor(0.0284)
Epoch 7: test accuracy = 0.4668, loss = 0.028393130749464035
Avg_loss: tensor(0.0280)
Epoch 8: test accuracy = 0.5237, loss = 0.028042368590831757
Avg_loss: tensor(0.0280)
Epoch 9: test accuracy = 0.5294, loss = 0.027956528589129448
Avg_loss: tensor(0.0277)
Epoch 10: test accuracy = 0.5086, loss = 0.027731336653232574
Avg_loss: tensor(0.0276)
Epoch 11: test accuracy = 0.5185, loss = 0.0275511983782053
Avg_loss: tensor(0.0274)
Epoch 12: test accuracy = 0.5059, loss = 0.027358775958418846
Avg_loss: tensor(0.0273)
Epoch 13: test accuracy = 0.5548, loss = 0.02729284018278122
Avg_loss: tensor(0.0271)
Epoch 14: test accuracy = 0.5282, loss = 0.027083780616521835
Avg_loss: tensor(0.0270)
Epoch 15: test accuracy = 0.5327, loss = 0.026964979246258736
Avg_loss: tensor(0.0268)
Epoch 16: test accuracy = 0.5556, loss = 0.026788707822561264
Avg_loss: tensor(0.0267)
Epoch 17: test accuracy = 0.5517, loss = 0.026665808632969856
Avg_loss: tensor(0.0266)
Epoch 18: test accuracy = 0.5665, loss = 0.026565730571746826
Avg_loss: tensor(0.0264)
Epoch 19: test accuracy = 0.5715, loss = 0.026430971920490265
Final test accuracy: 0.5715
"""
