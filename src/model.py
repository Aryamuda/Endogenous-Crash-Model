import numpy as np
from scipy.stats import cauchy


class KuramotoModel:
    """
    Deterministic Kuramoto Model for phase synchronization.
    """
    
    def __init__(self, N=1000, omega=None, seed=42):
        self.N = N
        self.seed = seed
        np.random.seed(seed)
        
        # Initialize natural frequencies
        if omega is None:
            self.omega = cauchy.rvs(loc=0, scale=1, size=N)
        else:
            self.omega = omega
            
        # Initialize phases uniformly
        self.theta = np.random.uniform(0, 2*np.pi, N)
    
    def reset_phases(self, theta=None):
        """Reset phases to initial or specified distribution."""
        if theta is None:
            self.theta = np.random.uniform(0, 2*np.pi, self.N)
        else:
            self.theta = theta.copy()
    
    def compute_order_parameter(self):
        """
        Compute the complex order parameter z = r*exp(i*psi).
        """
        z = np.mean(np.exp(1j * self.theta))
        return np.abs(z), np.angle(z)
    
    def step(self, K, dt):
        """
        Perform one deterministic Kuramoto step.
        """
        r, psi = self.compute_order_parameter()
        dtheta = self.omega + K * r * np.sin(psi - self.theta)
        self.theta += dtheta * dt
    
    def evolve(self, K, T, dt):
        """
        Evolve the system for time T and return order parameter history.
        """
        steps = int(T / dt)
        r_history = np.zeros(steps)
        
        for i in range(steps):
            self.step(K, dt)
            r_history[i], _ = self.compute_order_parameter()
        
        return r_history
    
    def find_steady_state(self, K, T, dt):
        """
        Run simulation to steady state and return final r.
        """
        self.evolve(K, T, dt)
        r_final, _ = self.compute_order_parameter()
        return r_final


class StochasticKuramotoModel(KuramotoModel):
    """
    Stochastic Kuramoto Model with additive Wiener noise.
    """
    
    def __init__(self, N=1000, sigma=0.5, omega=None, seed=42):
        super().__init__(N, omega, seed)
        self.sigma = sigma
    
    def step(self, K, dt):
        """
        Stochastic Euler-Maruyama step.
        """
        r, psi = self.compute_order_parameter()
        
        # Deterministic drift
        drift = self.omega + K * r * np.sin(psi - self.theta)
        
        # Stochastic diffusion (Wiener process)
        diffusion = self.sigma * np.random.normal(0, np.sqrt(dt), self.N)
        
        # Update
        self.theta += drift * dt + diffusion
        
        return r  # Return current r for convenience


class MarketCrashModel(StochasticKuramotoModel):
    """
    Kuramoto-based market crash model with jump intensity.
    """
    
    def __init__(self, N=1000, sigma=0.5, alpha=20.0, lambda_base=0.5, omega=None, seed=42):
        super().__init__(N, sigma, omega, seed)
        self.alpha = alpha
        self.lambda_base = lambda_base
    
    def compute_jump_intensity(self, r):
        """
        Compute current market jump intensity.
        """
        return self.lambda_base + self.alpha * r
    
    def step_with_jumps(self, K, dt):
        """
        Evolve system and check for market jump event.
        """
        # Standard stochastic step
        r = self.step(K, dt)
        
        # Compute jump intensity
        lambda_t = self.compute_jump_intensity(r)
        
        # Poisson jump check
        p_jump = lambda_t * dt
        jumped = np.random.random() < p_jump
        
        return r, lambda_t, jumped
    
    def simulate_with_ramp(self, K_start, K_end, T, dt):
        """
        Simulate with linearly ramping coupling strength.
        """
        steps = int(T / dt)
        K_ramp = np.linspace(K_start, K_end, steps)
        time_axis = np.linspace(0, T, steps)
        
        r_history = np.zeros(steps)
        lambda_history = np.zeros(steps)
        jump_times = []
        
        for i in range(steps):
            r, lambda_t, jumped = self.step_with_jumps(K_ramp[i], dt)
            
            r_history[i] = r
            lambda_history[i] = lambda_t
            
            if jumped:
                jump_times.append(time_axis[i])
        
        return {
            'time_axis': time_axis,
            'r_history': r_history,
            'lambda_history': lambda_history,
            'jump_times': jump_times,
            'K_ramp': K_ramp
        }


def phase_transition_sweep(K_values, N=1000, T=50, dt=0.05, seed=42):
    """
    Sweep coupling strength and measure steady-state order parameter.
    """
    model = KuramotoModel(N=N, seed=seed)
    # Store the initial condition (generated once with seed=42)
    theta_init = model.theta.copy()
    r_steady = []
    
    for K in K_values:
        # Reset to the same initial condition for fair comparison
        model.reset_phases(theta_init)
        r_final = model.find_steady_state(K, T, dt)
        r_steady.append(r_final)
    
    return np.array(r_steady)


def compare_regimes(K_regimes, N=1000, T=50, dt=0.05, seed=42):
    """
    Generate time evolution for multiple coupling regimes.
    """
    model = KuramotoModel(N=N, seed=seed)
    steps = int(T / dt)
    time_axis = np.linspace(0, T, steps)
    time_series = []
    
    # Store initial condition
    theta_init = model.theta.copy()
    
    for K in K_regimes:
        model.reset_phases(theta_init)
        r_history = model.evolve(K, T, dt)
        time_series.append(r_history)
    
    return time_axis, time_series
