#pragma once
#include <torch/torch.h>
#include <functional>
#include "utils.h"

// Euler ODE integration for flow-matching action generation
// Mirrors the odeint() call in Python generate_flow_action()
class EulerODESolver {
public:
    // velocity_fn: takes (timestep_scalar, current_state) -> velocity
    // state: initial state [batch, horizon, action_dim]
    // times: time schedule [num_steps] (e.g., linspace(dt, 1.0, num_steps))
    //        - times[0] is the SECOND timestep (after first step done in prefill)
    //        - times[-1] = 1.0 is the final time
    // Returns: final state at t=1.0
    using VelocityFn = std::function<torch::Tensor(float, const torch::Tensor&)>;

    static torch::Tensor solve(VelocityFn velocity_fn,
                                const torch::Tensor& initial_state,
                                const torch::Tensor& times);

    // Full trajectory (returns all intermediate states)
    static std::vector<torch::Tensor> solve_trajectory(
        VelocityFn velocity_fn,
        const torch::Tensor& initial_state,
        const torch::Tensor& times);
};
