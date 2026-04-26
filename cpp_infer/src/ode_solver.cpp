#include "ode_solver.h"

torch::Tensor EulerODESolver::solve(VelocityFn velocity_fn,
                                     const torch::Tensor& initial_state,
                                     const torch::Tensor& times) {
    // Euler method: y_{n+1} = y_n + dt * f(t_n, y_n)
    // times: [num_remaining_steps] - the timesteps to integrate over
    // (times[0] is already the second step since first step done in prefill)

    auto state = initial_state.clone();
    int num_steps = times.size(0);
    auto times_cpu = times.to(torch::kCPU).to(torch::kFloat32);
    auto t_ptr = times_cpu.data_ptr<float>();

    for (int i = 0; i < num_steps; i++) {
        float t_current = t_ptr[i];
        float dt;
        if (i < num_steps - 1) {
            dt = t_ptr[i + 1] - t_ptr[i];
        } else {
            dt = 1.0f - t_ptr[i];  // Integrate to t=1.0
            if (dt <= 0) break;     // Already at t=1.0
        }

        // Compute velocity at current state
        auto velocity = velocity_fn(t_current, state);

        // Euler step
        state = state + dt * velocity;
    }

    return state;
}

std::vector<torch::Tensor> EulerODESolver::solve_trajectory(
    VelocityFn velocity_fn,
    const torch::Tensor& initial_state,
    const torch::Tensor& times) {

    std::vector<torch::Tensor> trajectory;
    auto state = initial_state.clone();
    trajectory.push_back(state.clone());

    int num_steps = times.size(0);
    auto times_cpu = times.to(torch::kCPU).to(torch::kFloat32);
    auto t_ptr = times_cpu.data_ptr<float>();

    for (int i = 0; i < num_steps; i++) {
        float t_current = t_ptr[i];
        float dt;
        if (i < num_steps - 1) {
            dt = t_ptr[i + 1] - t_ptr[i];
        } else {
            dt = 1.0f - t_ptr[i];
            if (dt <= 0) break;
        }

        auto velocity = velocity_fn(t_current, state);
        state = state + dt * velocity;
        trajectory.push_back(state.clone());
    }

    return trajectory;
}
