import torch

###############################################################################
# MPPI Controller with Adaptive Noise
###############################################################################
class MPPI:
    """
    Model Predictive Path Integral (MPPI) controller in PyTorch, with the option
    of adaptive noise for control exploration.
    """

    def __init__(
        self,
        dynamics_func,
        cost_func,
        nx,                  # state dimension
        nu,                  # control dimension
        horizon=30,
        num_samples=1000,
        lambda_=1.0,         # temperature / exploration parameter
        u_min=None,
        u_max=None,
        noise_sigma=1.0,     # initial std dev for control noise
        device='cuda',
        ###########################################
        # Additional parameters for adaptive noise
        ###########################################
        adaptive_noise=True,
        target_cost=5.0,     # if min cost < target_cost, reduce noise
        noise_adapt_factor=0.95,  # multiply or divide noise by this factor
        noise_min=0.01,
        noise_max=100.0,
        return_trajectory=False
    ):
        """
        Args:
            dynamics_func: f(x, u, dt) -> x_next (vectorized)
            cost_func:     cost function cost(x_current, x_next, u, t, horizon, dt)
            nx:            state dimension
            nu:            control dimension
            horizon:       MPPI horizon
            num_samples:   number of random rollouts
            lambda_:       temperature for cost exponentiation
            u_min, u_max:  bounds for controls
            noise_sigma:   initial scale for random control noise
            device:        'cuda' or 'cpu'
            adaptive_noise: If True, adapt the noise scale each iteration
            target_cost:   If the best cost is below this, shrink the noise
            noise_adapt_factor: factor by which to multiply or divide the noise
            noise_min, noise_max: clamp the noise to these bounds
        """
        self.dynamics_func = dynamics_func
        self.cost_func = cost_func
        self.nx = nx
        self.nu = nu
        self.horizon = horizon
        self.num_samples = num_samples
        self.lambda_ = lambda_
        self.device = device
        self.return_trajectory = return_trajectory

        if u_min is not None:
            self.u_min = torch.tensor(u_min, device=device, dtype=torch.float32)
        else:
            self.u_min = None
        if u_max is not None:
            self.u_max = torch.tensor(u_max, device=device, dtype=torch.float32)
        else:
            self.u_max = None

        # Mean control initialization
        self.mean_control = torch.zeros(horizon, nu, device=device, dtype=torch.float32)

        # Noise scale
        self.noise_sigma = noise_sigma

        # Adaptive noise
        self.adaptive_noise = adaptive_noise
        self.target_cost = target_cost
        self.noise_adapt_factor = noise_adapt_factor
        self.noise_min = noise_min
        self.noise_max = noise_max

    @torch.no_grad()
    def command(self, x, dt):
        """
        Perform one MPPI iteration from current state x, return the first control
        of the updated sequence (apply it) and shift for receding horizon.

        x: [nx] current state
        dt: float
        """
        # Expand current state for parallel simulation
        x_batch = x.unsqueeze(0).repeat(self.num_samples, 1)  # [num_samples, nx]

        # Sample control noise: shape [horizon, num_samples, nu]
        noise = torch.randn((self.horizon, self.num_samples, self.nu), device=self.device) * self.noise_sigma
        # noise = torch.zeros((self.horizon, self.num_samples, self.nu), device=self.device).uniform_(-1, 1) * self.noise_sigma

        U = self.mean_control.unsqueeze(1) +  noise
        noise = noise

        # Enforce control bounds if provided
        if self.u_min is not None:
            U = torch.maximum(U, self.u_min)
        if self.u_max is not None:
            U = torch.minimum(U, self.u_max)

        # Rollout in parallel, accumulate cost
        if self.return_trajectory:
            costs, trajectories = self._rollout_and_cost(x_batch, U, dt)
            best_idx = torch.argmin(costs)
        else:
            costs = self._rollout_and_cost(x_batch, U, dt)


        # Compute weights
        min_cost = torch.min(costs)
        costs -= min_cost
        costs /= -self.lambda_
        torch.exp(costs, out=costs)
        weights = costs / torch.sum(costs)
        weights = weights.unsqueeze(1).expand(self.horizon, self.num_samples, 1)
        weights_temp = weights.mul(noise)
        weights = torch.sum(weights_temp, dim=1)
        self.mean_control += weights
        self.mean_control.clamp_(self.u_min, self.u_max)
        # weights = torch.exp(- (costs - min_cost) / self.lambda_)  # [num_samples]
        # w_sum = torch.sum(weights) + 1e-10

        # # Weighted update of mean control
        # # shape(U) = [horizon, num_samples, nu]
        # weighted_u = torch.einsum('s,hsn->hn', weights, U)  # [horizon, nu]
        # new_mean_control = weighted_u / w_sum
 
        # self.mean_control = new_mean_control

        # The control to apply now is the first in the updated sequence
        u0 = self.mean_control[0].clone()

        # Shift horizon
        self.mean_control[:-1] = self.mean_control[1:].clone()
        self.mean_control[-1].zero_()

        # ADAPTIVE NOISE UPDATE
        if self.adaptive_noise:
            # If the best cost is < target, reduce noise (exploit).
            # Otherwise, increase noise (explore more).
            if min_cost < self.target_cost:
                # reduce noise
                self.noise_sigma *= self.noise_adapt_factor
            else:
                # increase noise
                self.noise_sigma /= self.noise_adapt_factor

            # Clamp noise
            self.noise_sigma = max(self.noise_min, min(self.noise_sigma, self.noise_max))

        if self.return_trajectory:
            return u0, trajectories, best_idx
        else:
            return u0

    @torch.no_grad()
    def _rollout_and_cost(self, x_batch, U, dt):
        """
        Roll out each of the sampled control trajectories in parallel and
        accumulate cost on-the-fly.

        x_batch: [num_samples, nx]
        U:       [horizon, num_samples, nu]
        dt:      float
        Returns: cost: [num_samples]
        """
        cost = torch.zeros(self.num_samples, device=self.device)
        trajectories = torch.zeros((self.horizon, self.num_samples, self.nx), device=self.device)

        current_x = x_batch

        for t in range(self.horizon):
            u_t = U[t]  # shape [num_samples, nu]
            next_x = self.dynamics_func(current_x, u_t, dt)

            step_cost = self.cost_func(current_x, next_x, u_t, t, self.horizon, dt)
            cost += step_cost
            current_x = next_x
        
        if self.return_trajectory:

            return cost, trajectories
        else:
            return cost

'''
###############################################################################
# Cost Function
###############################################################################
def cost_function(current_x, next_x, u_t, t, horizon, dt):
    """
    Computes the cost at each timestep.
    State = [roll, pitch, roll_rate, pitch_rate, wheel_speed, steer_angle].
    Control = [delta_wheel_speed, delta_steer_angle].

    Goal: get roll, pitch to desired angles with minimal angular velocity by the final time.
    Also handle large angle differences (flips).
    """
    # Desired final angles
    desired_roll  = 0.0
    desired_pitch = 0.0

    # Extract next_x states
    roll       = next_x[:, 0]
    pitch      = next_x[:, 1]
    roll_rate  = next_x[:, 2]
    pitch_rate = next_x[:, 3]
    wheel_spd  = next_x[:, 4]

    # Are we at the final step?
    is_final = (t == horizon - 1)

    # We can do angle wrapping to allow flips: compute the minimal angle difference
    # For instance, define a helper for angle error:
    def angle_diff(a, b):
        # returns the signed difference in [-pi, pi]
        diff = (a - b + torch.pi) % (2*torch.pi) - torch.pi
        return diff

    roll_error  = angle_diff(roll,  desired_roll)
    pitch_error = angle_diff(pitch, desired_pitch)

    # Weighted differently for final vs. intermediate steps
    if is_final:
        w_angle = 50.0
        w_rate  = 20.0
    else:
        w_angle = 0.5
        w_rate  = 0.2

    # Angular error cost
    angle_cost = (roll_error**2 + pitch_error**2)

    # Angular velocity cost
    rate_cost = (roll_rate**2 + pitch_rate**2)

    # Control use penalty (small)
    w_ctrl = 0.001
    ctrl_cost = (u_t**2).sum(dim=-1)

    # We might also penalize saturations or being “stuck” at extremes. For example:
    #   If wheel speed is near max, and we keep commanding +delta, that might be wasteful.
    #   Or if near 0 wheel speed but keep commanding negative delta, also wasteful.
    # A simple approach: add cost if sign(delta_wheel_speed) tries to push beyond saturation:
    #   If wheel_spd ~ wheel_spd_max and delta_wheel_speed>0 => cost
    #   If wheel_spd ~ wheel_spd_min and delta_wheel_speed<0 => cost

    # We'll define some small threshold for "near max" or "near min"
    near_thresh = 10.0
    wheel_spd_min = 0.0
    wheel_spd_max = 8000.0

    delta_ws = u_t[:, 0]
    penalize_saturation = torch.zeros_like(delta_ws)
    # near max & accelerating more
    near_max = (wheel_spd > (wheel_spd_max - near_thresh))
    penalize_saturation = torch.where(
        (near_max & (delta_ws > 0)),
        penalize_saturation + 200.0 * delta_ws.abs(),  # big penalty
        penalize_saturation
    )
    # near min & decelerating more
    near_min = (wheel_spd < (wheel_spd_min + near_thresh))
    penalize_saturation = torch.where(
        (near_min & (delta_ws < 0)),
        penalize_saturation + 200.0 * delta_ws.abs(),
        penalize_saturation
    )

    # Combine cost terms
    cost = w_angle * angle_cost + w_rate * rate_cost + w_ctrl * ctrl_cost + penalize_saturation

    return cost

###############################################################################
# Car Mid-Air Dynamics with Wheel Speed Saturation
###############################################################################
def car_dynamics_step(states, controls, dt):
    """
    Vectorized car mid-air dynamics step with saturations on wheel speed.
    States:  [batch_size, nx]
        [roll, pitch, roll_rate, pitch_rate, wheel_speed, steer_angle]
    Controls:[batch_size, nu]
        [delta_wheel_speed, delta_steer_angle]

    Returns next_states: [batch_size, nx]
    """
    # Unpack
    roll       = states[:, 0]
    pitch      = states[:, 1]
    roll_rate  = states[:, 2]
    pitch_rate = states[:, 3]
    wheel_spd  = states[:, 4]
    steer_ang  = states[:, 5]

    delta_ws   = controls[:, 0]
    delta_st   = controls[:, 1]

    # Bounds for wheel speed (example: 0 to 8000)
    wheel_spd_min = 0.0
    wheel_spd_max = 8000.0

    # Update wheel speed (and saturate)
    new_wheel_spd = wheel_spd + delta_ws
    new_wheel_spd = torch.clip(new_wheel_spd, wheel_spd_min, wheel_spd_max)

    # Update steering angle (optionally saturate if needed)
    steer_ang_min, steer_ang_max = -1.0, 1.0
    new_steer_ang = steer_ang + delta_st
    new_steer_ang = torch.clip(new_steer_ang, steer_ang_min, steer_ang_max)

    # -------------------------------------------------------------
    #  Simplified model of inertia + precession:
    #    - Steering at center => changes in wheel speed => pitch changes.
    #    - Steering off-center => changes in wheel speed => pitch + roll.
    #    - At constant wheel speed => changing steer => roll changes.
    #
    #  We'll define nominal constants (you must tune them).
    # -------------------------------------------------------------
    alpha_pitch = 0.02  # effect of wheel speed acceleration on pitch
    alpha_roll  = 0.02  # effect of wheel speed acceleration on roll if steer != 0
    beta_roll   = 0.01  # effect of steering change on roll (scaled by wheel speed)
    damp_roll_rate  = 0.01
    damp_pitch_rate = 0.01

    # Wheel speed change this timestep is (new_wheel_spd - wheel_spd)
    actual_delta_ws = new_wheel_spd - wheel_spd

    # Precession from wheel speed changes
    pitch_acc_due_to_ws = alpha_pitch * actual_delta_ws
    roll_acc_due_to_ws  = alpha_roll * actual_delta_ws * torch.abs(new_steer_ang)

    # Steering-induced roll, scaled by current wheel speed
    roll_acc_due_to_steer = beta_roll * (0.5*(new_wheel_spd + wheel_spd)) * delta_st

    # Net roll acceleration
    roll_acc = roll_acc_due_to_ws + roll_acc_due_to_steer - damp_roll_rate * roll_rate
    # Net pitch acceleration
    pitch_acc = pitch_acc_due_to_ws - damp_pitch_rate * pitch_rate

    # Integrate
    new_roll_rate  = roll_rate  + roll_acc  * dt
    new_pitch_rate = pitch_rate + pitch_acc * dt

    new_roll  = roll  + new_roll_rate  * dt
    new_pitch = pitch + new_pitch_rate * dt

    # Next state
    next_states = torch.stack([
        new_roll, new_pitch,
        new_roll_rate, new_pitch_rate,
        new_wheel_spd, new_steer_ang
    ], dim=-1)

    return next_states

###############################################################################
# Example Usage
###############################################################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # State dimension: [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, wheel_speed, steer_angle]
    nx = 8
    # Control dimension: [delta_wheel_speed, delta_steer_angle]
    nu = 2
    
    # MPPI hyperparameters
    horizon = 30
    num_samples = 1000
    lambda_ = 1.0
    initial_noise_sigma = 50.0

    # Control bounds
    u_min = [-2000., -0.5]
    u_max = [ 2000.,  0.5]

    # Instantiate the MPPI with adaptive noise
    mppi = MPPI(
        dynamics_func=car_dynamics_step,
        cost_func=cost_function,
        nx=nx,
        nu=nu,
        horizon=horizon,
        num_samples=num_samples,
        lambda_=lambda_,
        u_min=u_min,
        u_max=u_max,
        noise_sigma=initial_noise_sigma,
        device=device,
        ################################
        # Adaptive noise parameters
        ################################
        adaptive_noise=True,
        target_cost=5.0,         # tune to your scenario
        noise_adapt_factor=0.95, # shrink or grow by 5%
        noise_min=0.1,
        noise_max=200.0
    )

    # Initial state
    init_state = torch.tensor([0.3, 0.5, 0.0, 0.0, 3000.0, 0.0], device=device)

    dt = 0.01
    steps_to_sim = 40
    current_state = init_state.clone()

    for step in range(steps_to_sim):
        # 1) MPPI picks the next control
        u = mppi.command(current_state, dt)  # shape [nu]

        # 2) Apply to "real" or sim dynamics
        next_state = car_dynamics_step(current_state.unsqueeze(0), u.unsqueeze(0), dt)
        current_state = next_state.squeeze(0)

        # Print info
        if step % 5 == 0:
            print(f"Step {step}: control={u}, noise_sigma={mppi.noise_sigma:.2f}")
'''