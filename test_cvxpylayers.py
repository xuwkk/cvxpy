"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer


def build_projection_layer(n: int) -> tuple[CvxpyLayer, cp.Problem, cp.Parameter, cp.Variable]:
    """Build a simple differentiable projection problem.

    min_x ||x - p||_2^2
    s.t.  x >= 0, sum(x) == 1
    """
    x = cp.Variable(n)
    p = cp.Parameter(n)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - p)), [x >= 0, cp.sum(x) == 1])
    assert problem.is_dcp(dpp=True)
    layer = CvxpyLayer(problem, parameters=[p], variables=[x])
    return layer, problem, p, x


def test_cvxpylayers_torch_forward_and_backward() -> None:
    torch.manual_seed(0)
    n = 5
    layer, problem, p_param, x_var = build_projection_layer(n)

    p_torch = torch.tensor([0.3, -0.4, 0.9, 0.1, -0.2], dtype=torch.double, requires_grad=True)
    x_torch, = layer(p_torch, solver_args={"eps": 1e-9, "max_iters": 20000})

    # Feasibility checks.
    assert torch.all(x_torch >= -1e-6)
    assert torch.isclose(torch.sum(x_torch), torch.tensor(1.0, dtype=torch.double), atol=1e-6)

    # Gradient check.
    loss = torch.sum(x_torch**2)
    loss.backward()
    assert p_torch.grad is not None
    assert torch.all(torch.isfinite(p_torch.grad))

    # Compare to direct CVXPY solve on the same problem.
    p_param.value = p_torch.detach().numpy()
    problem.solve(solver=cp.CLARABEL)
    assert problem.status == cp.OPTIMAL
    diff = np.linalg.norm(x_torch.detach().numpy() - x_var.value)
    assert diff <= 5e-4


def test_cvxpylayers_torch_batched_forward() -> None:
    torch.manual_seed(1)
    n = 4
    batch_size = 3
    layer, _, _, _ = build_projection_layer(n)

    p_batch = torch.tensor(
        [
            [1.0, -1.0, 0.5, -0.3],
            [0.2, 0.3, -0.1, 0.8],
            [-0.5, 0.0, 0.5, 1.5],
        ],
        dtype=torch.double,
        requires_grad=True,
    )
    x_batch, = layer(p_batch, solver_args={"eps": 1e-8, "max_iters": 20000})

    assert x_batch.shape == (batch_size, n)
    assert torch.all(x_batch >= -1e-6)
    assert torch.allclose(
        torch.sum(x_batch, dim=1),
        torch.ones(batch_size, dtype=torch.double),
        atol=1e-6,
    )

    # Batched backward pass.
    loss = torch.sum((x_batch - 0.25) ** 2)
    loss.backward()
    assert p_batch.grad is not None
    assert torch.all(torch.isfinite(p_batch.grad))
