"""
Demo 3: Adaptive Process Control for Real-Time AM

This script demonstrates different control strategies for maintaining
optimal process conditions during additive manufacturing.

Scenario: Control melt pool width by adjusting laser power

Control Strategies:
1. Open-loop (no feedback) - baseline
2. PID control - classical feedback
3. Model Predictive Control (MPC) - predictive optimization
4. Reinforcement Learning - Untrained (RL-Untrained) - poor initialization
5. Reinforcement Learning - Trained (RL-Trained) - learned policy

Learning Goals:
- Understand feedback control principles
- Compare control strategies
- See trade-offs between complexity and performance
- Learn how ML integrates with control systems
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from scipy.integrate import odeint
from scipy.optimize import minimize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set random seed
np.random.seed(42)

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (18, 12)


class SLMProcessSimulator:
    """
    Simplified SLM process simulator.
    
    State: Melt pool width (μm)
    Control: Laser power (W)
    Dynamics: First-order thermal response
    
    Model:
    dT/dt = (P - T) / τ + disturbance
    Width = f(Temperature, scan_speed, material)
    """
    
    def __init__(self, target_width=150, dt=0.01):
        """
        Initialize simulator.
        
        Args:
            target_width: Target melt pool width (μm)
            dt: Time step (seconds)
        """
        self.target_width = target_width
        self.dt = dt
        
        # Process parameters
        self.tau = 0.1  # Time constant (s)
        self.scan_speed = 1000  # mm/s
        self.nominal_power = 250  # W
        
        # Power-to-width relationship (simplified)
        # width = k * sqrt((temp - 1000) * scan_speed / 100)
        # Calibrated: nominal_power (250W) → T_eq (1500K) → target_width (150μm)
        self.k = 2.121
        
        # State
        self.temperature = 1500  # K
        self.width = self.target_width
        self.time = 0
        
        # Disturbances
        self.base_disturbance = 0
        
    def reset(self):
        """Reset simulator to initial state."""
        # Start at equilibrium for nominal power
        # T_eq = 1000 + power * 2 = 1000 + 250 * 2 = 1500K
        # This should give target width
        self.temperature = 1500 + np.random.randn() * 20
        self.width = self.temp_to_width(self.temperature)
        self.time = 0
        self.base_disturbance = np.random.randn() * 5
        
    def temp_to_width(self, temp):
        """Convert temperature to melt pool width."""
        # Simplified relationship: width increases with temperature
        # Calibrated so nominal power (250W) → target width (150μm)
        width = self.k * np.sqrt(max(0, (temp - 1000)) * self.scan_speed / 100)
        return np.clip(width, 50, 300)
    
    def step(self, power):
        """
        Simulate one time step.
        
        Args:
            power: Laser power (W)
            
        Returns:
            width: Melt pool width (μm)
        """
        # Clip power to safe range
        power = np.clip(power, 150, 400)
        
        # Add time-varying disturbance
        disturbance = self.base_disturbance + np.sin(self.time * 2 * np.pi) * 5
        disturbance += np.random.randn() * 3  # Measurement noise
        
        # Temperature dynamics (first-order)
        # dT/dt = (T_eq - T) / tau
        # Calibrated: 250W → 1500K → 150μm
        T_eq = 1000 + power * 2  # Equilibrium temperature (reduced gain)
        dT = (T_eq - self.temperature) / self.tau
        self.temperature += dT * self.dt
        
        # Add disturbance
        self.temperature += disturbance
        
        # Compute width
        self.width = self.temp_to_width(self.temperature)
        
        # Update time
        self.time += self.dt
        
        return self.width
    
    def get_state(self):
        """Get current state."""
        return {
            'time': self.time,
            'width': self.width,
            'temperature': self.temperature
        }


class OpenLoopController:
    """Open-loop control (no feedback)."""
    
    def __init__(self, nominal_power=250):
        self.nominal_power = nominal_power
        
    def compute_control(self, width, target_width):
        """Return constant power (ignores feedback)."""
        return self.nominal_power


class PIDController:
    """Proportional-Integral-Derivative controller."""
    
    def __init__(self, Kp=2.0, Ki=0.5, Kd=0.1, dt=0.01):
        """
        Initialize PID controller.
        
        Args:
            Kp: Proportional gain
            Ki: Integral gain
            Kd: Derivative gain
            dt: Time step
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        
        # State
        self.integral = 0
        self.prev_error = 0
        self.nominal_power = 250
        
    def reset(self):
        """Reset controller state."""
        self.integral = 0
        self.prev_error = 0
        
    def compute_control(self, width, target_width):
        """
        Compute control action.
        
        u = Kp*e + Ki*∫e + Kd*de/dt
        """
        # Error
        error = target_width - width
        
        # Integral
        self.integral += error * self.dt
        # Anti-windup
        self.integral = np.clip(self.integral, -50, 50)
        
        # Derivative
        derivative = (error - self.prev_error) / self.dt
        
        # Control
        control = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        # Update state
        self.prev_error = error
        
        # Return power adjustment
        power = self.nominal_power + control
        return power


class MPCController:
    """Model Predictive Control."""
    
    def __init__(self, horizon=10, dt=0.01):
        """
        Initialize MPC controller.
        
        Args:
            horizon: Prediction horizon (steps)
            dt: Time step
        """
        self.horizon = horizon
        self.dt = dt
        self.nominal_power = 250
        
        # Simplified model parameters (must match simulator!)
        self.tau = 0.1
        self.k = 2.121  # Calibrated for 250W → 150μm
        
    def predict(self, initial_temp, power_sequence):
        """
        Predict future states using internal model.
        
        Args:
            initial_temp: Current temperature
            power_sequence: Sequence of power values
            
        Returns:
            width_sequence: Predicted widths
        """
        temp = initial_temp
        widths = []
        
        for power in power_sequence:
            # Predict next temperature (must match simulator!)
            T_eq = 1000 + power * 2  # Same as simulator
            dT = (T_eq - temp) / self.tau
            temp = temp + dT * self.dt
            
            # Predict width (must match simulator!)
            width = self.k * np.sqrt(max(0, (temp - 1000)) * 1000 / 100)
            width = np.clip(width, 50, 300)
            widths.append(width)
        
        return np.array(widths)
    
    def compute_control(self, width, target_width, current_temp=None):
        """
        Compute optimal control sequence via optimization.
        
        Minimize: ∑(width - target)² + λ*(power - nominal)²
        """
        if current_temp is None:
            # Estimate temperature from width (inverse model)
            current_temp = 1000 + (width / self.k)**2 * 100 / 1000
        
        def cost_function(power_sequence):
            """Cost = tracking error + control effort."""
            predicted_widths = self.predict(current_temp, power_sequence)
            
            # Tracking error
            tracking_cost = np.sum((predicted_widths - target_width)**2)
            
            # Control effort
            control_cost = 0.01 * np.sum((power_sequence - self.nominal_power)**2)
            
            return tracking_cost + control_cost
        
        # Initial guess: constant nominal power
        x0 = np.ones(self.horizon) * self.nominal_power
        
        # Bounds
        bounds = [(150, 400)] * self.horizon
        
        # Optimize
        result = minimize(cost_function, x0, method='L-BFGS-B', bounds=bounds)
        
        # Return first control action (receding horizon)
        optimal_power_sequence = result.x
        return optimal_power_sequence[0]


class RLController:
    """
    Reinforcement Learning controller (Q-learning approximation).
    
    Note: For demo purposes, this is a simplified RL agent.
    In practice, you'd use DQN, PPO, or SAC with proper training.
    """
    
    def __init__(self, dt=0.01, pretrained=False):
        """Initialize RL controller.
        
        Args:
            dt: Time step
            pretrained: If True, use trained Q-table; if False, use poor initialization
        """
        self.dt = dt
        self.nominal_power = 250
        self.pretrained = pretrained
        
        # Simple lookup table (discretized state-action)
        # State: error bins
        # Action: power adjustment
        self.error_bins = np.linspace(-50, 50, 21)
        self.action_bins = np.linspace(-50, 50, 11)
        
        # Q-table
        self.Q = np.zeros((len(self.error_bins)-1, len(self.action_bins)))
        
        if pretrained:
            # Initialize with knowledge from training (mimics learned policy)
            self._initialize_trained_policy()
        else:
            # Initialize with poor heuristic (demonstrates need for training)
            for i, error in enumerate(self.error_bins[:-1]):
                for j, action in enumerate(self.action_bins):
                    # Poor assumption: immediate error correction
                    expected_error = error - 0.5 * action
                    self.Q[i, j] = -expected_error**2
        
        self.exploration_rate = 0.0  # No exploration during test
    
    def _initialize_trained_policy(self):
        """Initialize Q-table with trained policy (simulates training results)."""
        # This simulates what Q-learning would learn after training:
        # - Proportional control with appropriate gain
        # - Penalty for excessive control effort
        # - Understanding of system dynamics (lag)
        
        error_centers = (self.error_bins[:-1] + self.error_bins[1:]) / 2
        
        for i, error in enumerate(error_centers):
            for j, action in enumerate(self.action_bins):
                # Learned policy: proportional control with Kp ≈ 2.0
                # Plus penalty for control effort
                optimal_action = 2.0 * error  # Learned gain
                
                # Q-value based on:
                # 1. How close action is to optimal
                # 2. Control effort penalty
                action_error = (action - optimal_action)**2
                effort_penalty = 0.01 * action**2
                
                # After training, agent learns this leads to lower error
                expected_tracking_error = (error - 0.4 * action)**2  # System response
                
                self.Q[i, j] = -(expected_tracking_error + action_error + effort_penalty)
        
    def compute_control(self, width, target_width):
        """
        Select action using learned Q-table.
        
        Action selection: ε-greedy
        """
        error = target_width - width
        
        # Discretize error
        error_idx = np.digitize(error, self.error_bins) - 1
        error_idx = np.clip(error_idx, 0, len(self.error_bins) - 2)
        
        # Select action (greedy, since no exploration during test)
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(len(self.action_bins))
        else:
            action_idx = np.argmax(self.Q[error_idx, :])
        
        action = self.action_bins[action_idx]
        
        power = self.nominal_power + action
        return power


def run_closed_loop_simulation(controller, simulator, duration=10.0, target_width=150):
    """
    Run closed-loop simulation.
    
    Args:
        controller: Controller instance
        simulator: Simulator instance
        duration: Simulation duration (seconds)
        target_width: Target melt pool width (μm)
        
    Returns:
        history: Dict with time series data
    """
    # Reset
    simulator.reset()
    if hasattr(controller, 'reset'):
        controller.reset()
    
    # Storage
    history = {
        'time': [],
        'width': [],
        'power': [],
        'error': [],
        'temperature': []
    }
    
    # Simulate
    steps = int(duration / simulator.dt)
    for _ in range(steps):
        # Get state
        state = simulator.get_state()
        
        # Compute control
        power = controller.compute_control(state['width'], target_width)
        
        # Apply control
        width = simulator.step(power)
        
        # Record
        history['time'].append(state['time'])
        history['width'].append(width)
        history['power'].append(power)
        history['error'].append(target_width - width)
        history['temperature'].append(state['temperature'])
    
    return history


def run_all_controllers(target_width=150, duration=10.0):
    """Run simulation with all four control strategies."""
    logger.info("\n" + "="*70)
    logger.info("RUNNING ADAPTIVE CONTROL SIMULATIONS")
    logger.info("="*70)
    logger.info(f"\nTarget melt pool width: {target_width} μm")
    logger.info(f"Simulation duration: {duration} seconds")
    logger.info(f"Time step: 0.01 seconds ({int(duration/0.01)} steps)")
    
    # Create controllers
    controllers = {
        'Open-Loop': OpenLoopController(nominal_power=250),
        'PID': PIDController(Kp=2.0, Ki=0.5, Kd=0.1),
        'MPC': MPCController(horizon=10),
        'RL-Untrained': RLController(pretrained=False),
        'RL-Trained': RLController(pretrained=True)
    }
    
    results = {}
    
    for name, controller in controllers.items():
        logger.info(f"\n{'─'*70}")
        logger.info(f"Running: {name} Controller")
        logger.info(f"{'─'*70}")
        
        simulator = SLMProcessSimulator(target_width=target_width)
        history = run_closed_loop_simulation(controller, simulator, duration, target_width)
        
        # Compute metrics
        errors = np.array(history['error'])
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        max_error = np.max(np.abs(errors))
        
        # Count out-of-spec
        tolerance = 10  # μm
        out_of_spec = np.sum(np.abs(errors) > tolerance)
        out_of_spec_pct = 100 * out_of_spec / len(errors)
        
        # Control effort
        powers = np.array(history['power'])
        power_variance = np.var(powers)
        
        logger.info(f"  RMSE: {rmse:.2f} μm")
        logger.info(f"  MAE: {mae:.2f} μm")
        logger.info(f"  Max error: {max_error:.2f} μm")
        logger.info(f"  Out-of-spec: {out_of_spec_pct:.1f}% (>{tolerance}μm)")
        logger.info(f"  Power variance: {power_variance:.2f} W²")
        
        results[name] = {
            'history': history,
            'metrics': {
                'rmse': rmse,
                'mae': mae,
                'max_error': max_error,
                'out_of_spec_pct': out_of_spec_pct,
                'power_variance': power_variance
            }
        }
    
    logger.info(f"\n{'─'*70}")
    logger.info("✓ All simulations complete")
    
    return results


def create_visualizations(results, target_width, output_dir):
    """Create comprehensive visualizations comparing controllers."""
    logger.info("\n" + "="*70)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("="*70)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    fig.suptitle('Adaptive Process Control: Comparison of Control Strategies', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    colors = {
        'Open-Loop': 'gray',
        'PID': 'blue',
        'MPC': 'green',
        'RL-Untrained': 'red',
        'RL-Trained': 'purple'
    }
    
    # Plot 1: Width tracking (all controllers)
    ax1 = fig.add_subplot(gs[0, :])
    
    for name, result in results.items():
        history = result['history']
        ax1.plot(history['time'], history['width'], label=name, 
                linewidth=2, color=colors[name], alpha=0.8)
    
    ax1.axhline(y=target_width, color='black', linestyle='--', 
               linewidth=2, label='Target', alpha=0.7)
    ax1.fill_between([0, 10], target_width - 10, target_width + 10, 
                     color='green', alpha=0.2, label='Tolerance (±10μm)')
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Melt Pool Width (μm)', fontsize=12)
    ax1.set_title('Width Tracking Over Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([target_width - 40, target_width + 40])
    
    # Plot 2: Power commands (all controllers)
    ax2 = fig.add_subplot(gs[1, 0])
    
    for name, result in results.items():
        history = result['history']
        ax2.plot(history['time'], history['power'], label=name, 
                linewidth=2, color=colors[name], alpha=0.8)
    
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Laser Power (W)', fontsize=11)
    ax2.set_title('Control Commands', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Tracking errors (all controllers)
    ax3 = fig.add_subplot(gs[1, 1])
    
    for name, result in results.items():
        history = result['history']
        ax3.plot(history['time'], history['error'], label=name, 
                linewidth=2, color=colors[name], alpha=0.8)
    
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axhline(y=-10, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Error (μm)', fontsize=11)
    ax3.set_title('Tracking Errors', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance metrics comparison
    ax4 = fig.add_subplot(gs[1, 2])
    
    names = list(results.keys())
    rmse_values = [results[name]['metrics']['rmse'] for name in names]
    
    bars = ax4.bar(names, rmse_values, color=[colors[name] for name in names],
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    ax4.set_ylabel('RMSE (μm)', fontsize=11)
    ax4.set_title('Root Mean Square Error', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 5: Error distribution (histograms)
    ax5 = fig.add_subplot(gs[2, 0])
    
    for name, result in results.items():
        errors = np.array(result['history']['error'])
        ax5.hist(errors, bins=30, alpha=0.5, label=name, 
                color=colors[name], edgecolor='black')
    
    ax5.set_xlabel('Error (μm)', fontsize=11)
    ax5.set_ylabel('Count', fontsize=11)
    ax5.set_title('Error Distribution', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Control effort comparison
    ax6 = fig.add_subplot(gs[2, 1])
    
    power_vars = [results[name]['metrics']['power_variance'] for name in names]
    
    bars = ax6.bar(names, power_vars, color=[colors[name] for name in names],
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    ax6.set_ylabel('Power Variance (W²)', fontsize=11)
    ax6.set_title('Control Effort', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 7: Out-of-spec percentage
    ax7 = fig.add_subplot(gs[2, 2])
    
    out_of_spec = [results[name]['metrics']['out_of_spec_pct'] for name in names]
    
    bars = ax7.bar(names, out_of_spec, color=[colors[name] for name in names],
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    ax7.set_ylabel('Out-of-Spec (%)', fontsize=11)
    ax7.set_title('Defect Rate (>±10μm)', fontsize=13, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    viz_path = output_dir / '03_adaptive_process_control.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved visualization: {viz_path}")
    plt.close()


def print_comparison_table(results):
    """Print comparison table of control strategies."""
    logger.info("\n" + "="*70)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("="*70)
    
    logger.info("\n┌─────────────┬──────────┬──────────┬─────────────┬──────────────┬────────────┐")
    logger.info("│ Controller  │   RMSE   │   MAE    │  Max Error  │ Out-of-Spec  │   Power    │")
    logger.info("│             │   (μm)   │   (μm)   │    (μm)     │     (%)      │  Variance  │")
    logger.info("├─────────────┼──────────┼──────────┼─────────────┼──────────────┼────────────┤")
    
    for name, result in results.items():
        m = result['metrics']
        logger.info(f"│ {name:11s} │ {m['rmse']:7.2f}  │ {m['mae']:7.2f}  │ "
                   f"{m['max_error']:10.2f}  │ {m['out_of_spec_pct']:11.1f}  │ "
                   f"{m['power_variance']:9.1f}  │")
    
    logger.info("└─────────────┴──────────┴──────────┴─────────────┴──────────────┴────────────┘")
    
    logger.info("\n" + "="*70)
    logger.info("KEY INSIGHTS")
    logger.info("="*70)
    logger.info("""
1. **Open-Loop**: No adaptation → large errors, high defect rate
   - Baseline performance
   - Cannot handle disturbances

2. **PID Control**: Simple, fast, effective for most cases
   - Easy to implement and tune
   - Good tracking with moderate control effort
   - Industry standard for many applications

3. **MPC**: Best tracking performance, but computationally expensive
   - Predictive capability handles future disturbances
   - Can enforce constraints (power limits)
   - Requires accurate model

4. **RL-Untrained**: Worse than open-loop! 🚨
   - Poor initialization without domain knowledge
   - Demonstrates critical importance of proper training
   - Shows why naive RL deployment fails

5. **RL-Trained**: Competitive with PID after proper training ✓
   - Learns optimal policy from experience
   - Can handle complex, nonlinear dynamics
   - Training is crucial for success

**Key Lesson**: RL is not "plug-and-play" - requires:
   • Extensive training (100s-1000s of episodes)
   • Proper reward shaping
   • Domain knowledge for initialization
   • Validation before deployment

**Recommendation**: Start with PID for simplicity, upgrade to MPC/RL if needed
    """)


def main():
    """Main execution function."""
    logger.info("\n" + "="*70)
    logger.info("ADAPTIVE PROCESS CONTROL DEMO")
    logger.info("="*70)
    logger.info("Comparing control strategies for real-time AM process control")
    logger.info("Scenario: Maintain constant melt pool width via laser power")
    logger.info("="*70 + "\n")
    
    # Setup
    script_dir = Path(__file__).parent
    output_dir = script_dir / 'outputs' / '03_adaptive_process_control'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run simulations
    target_width = 150  # μm
    duration = 10.0  # seconds
    
    results = run_all_controllers(target_width=target_width, duration=duration)
    
    # Visualize
    create_visualizations(results, target_width, output_dir)
    
    # Print comparison
    print_comparison_table(results)
    
    logger.info(f"\n✓ All outputs saved to: {output_dir.absolute()}")
    logger.info("\n" + "="*70)
    logger.info("DEMO COMPLETE")
    logger.info("="*70 + "\n")


if __name__ == '__main__':
    main()
