"""
Test file for scalable MCMC fitting with synthetic data.

This script demonstrates how to:
1. Generate synthetic data for multiple participants
2. Fit both Markov and Quantum models using the improved scalable approach
3. Use subsampling for large datasets (100+ participants)
4. Use HMCECS kernel for better convergence

Usage:
    python test_fit_scalable.py

Configuration:
    - Adjust NUM_PARTICIPANTS to test different scales (3, 10, 50, 100, 1000)
    - Adjust NUM_TRIALS per participant
    - Adjust ENABLE_SUBSAMPLING to True for 100+ participants
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import time

# Add parent directory to path to import cme modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from cme.utils import fit_model as fm
from cme.decision_models import confidence_accumulation as ca
from cme.utils import common_logging as cl

log = cl.get_logger("test_fit_scalable")

# ============================================================================
# Configuration for scalable testing
# ============================================================================

# Number of participants to test - ADJUST THIS VALUE TO TEST SCALING
# Try: 3, 5, 10, 20, 50, 100, 200, 1000
NUM_PARTICIPANTS = 10  # Start small, then increase

NUM_TRIALS = 30  # Trials per participant
NUM_STATES = 11  # State space size
MEASUREMENT_PROB = 0.7  # Measurement/observation probability
DELTA = 1.0  # Time step

# Model fitting parameters
NUM_WARMUP = 500  # Warmup iterations
SAMPLES_N = 500  # Final samples to draw
NUM_CHAINS = 4

# SCALING CONFIGURATION
# For small datasets (< 50 participants), use regular NUTS
# For large datasets (> 50 participants), use HMCECS with subsampling
ENABLE_SUBSAMPLING = NUM_PARTICIPANTS > 50
SUBSAMPLE_SIZE = None  # Auto-detected if None, set to 50-100 for manual control
USE_HMCECS = True if NUM_PARTICIPANTS > 50 else False

# ============================================================================
# Synthetic Data Generation
# ============================================================================

def generate_synthetic_data(n_participants, n_trials, n_states, 
                           measurement_prob, delta, model_type="Markov",
                           seed=42):
    """
    Generate simple synthetic decision-making data.
    
    Args:
        n_participants: Number of participants
        n_trials: Trials per participant
        n_states: Number of states in the walk
        measurement_prob: Probability of observing state at each step
        delta: Time step
        model_type: "Markov" or "Quantum"
        seed: Random seed for reproducibility
    
    Returns:
        RT: Response times (n_participants, n_trials)
        X: Correct/incorrect responses (n_participants, n_trials)
        IDs: Participant IDs
    """
    np.random.seed(seed)
    
    # Generate simple synthetic data:
    # - RTs from log-normal distribution (realistic for decision making)
    # - Responses as Bernoulli (correct/incorrect)
    # - Correlate: faster RTs slightly more likely to be correct
    
    RT = np.random.lognormal(mean=0.5, sigma=0.5, size=(n_participants, n_trials))
    RT = np.clip(RT, 0.5, 20.0)  # Realistic RT range (500ms - 20s)
    
    # Generate responses with slight speed-accuracy tradeoff
    # Faster RTs (lower percentile) → higher error rate
    accuracy_rates = 0.5 + 0.3 * (1 - (RT - RT.min()) / (RT.max() - RT.min()))  # 0.5 to 0.8 accuracy
    X = np.random.random((n_participants, n_trials)) < accuracy_rates
    X = X.astype(int)  # Convert to 0/1
    
    IDs = list(range(n_participants))
    
    log.info(f"Generated synthetic {model_type} data:")
    log.info(f"  Participants: {n_participants}")
    log.info(f"  Trials: {n_trials}")
    log.info(f"  RT shape: {RT.shape}")
    log.info(f"  X shape: {X.shape}")
    log.info(f"  RT range: [{RT.min():.3f}, {RT.max():.3f}] seconds")
    log.info(f"  Accuracy: {X.mean():.2%}")
    
    return RT, X, IDs


def save_synthetic_data(RT, X, IDs, output_dir="data_synthetic"):
    """Save synthetic data to CSV files for use with fit_model."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrames with trial numbers as column names and participant IDs as index
    df_RT = pd.DataFrame(RT, index=IDs)
    df_X = pd.DataFrame(X, index=IDs)
    
    rt_file = f"{output_dir}/synthetic_rt.csv"
    x_file = f"{output_dir}/synthetic_ra.csv"
    
    # Save with index (participant IDs) in first column
    # This ensures proper data structure when loaded
    df_RT.to_csv(rt_file, index=True, index_label="id")
    df_X.to_csv(x_file, index=True, index_label="id")
    
    log.info(f"Saved synthetic data:")
    log.info(f"  RT: {rt_file} shape={df_RT.shape}")
    log.info(f"  X: {x_file} shape={df_X.shape}")
    log.info(f"  Sample RT row: {df_RT.iloc[0].values}")
    log.info(f"  Sample X row: {df_X.iloc[0].values}")
    
    return output_dir


# ============================================================================
# Main Testing Function
# ============================================================================

def test_scalable_fitting():
    """Test fitting with scalable configuration."""
    
    log.info("=" * 80)
    log.info("SCALABLE MCMC FITTING TEST")
    log.info("=" * 80)
    log.info(f"Configuration:")
    log.info(f"  Participants: {NUM_PARTICIPANTS}")
    log.info(f"  Trials: {NUM_TRIALS}")
    log.info(f"  States: {NUM_STATES}")
    log.info(f"  Warmup: {NUM_WARMUP}")
    log.info(f"  Samples: {SAMPLES_N}")
    log.info(f"  Use HMCECS: {USE_HMCECS}")
    log.info(f"  Subsample Size: {SUBSAMPLE_SIZE}")
    log.info("=" * 80)
    
    # Create export directory
    os.makedirs("export", exist_ok=True)
    
    # Generate synthetic data for both models
    for model_type in ["Markov", "Quantum"]:
        log.info(f"\n{'='*80}")
        log.info(f"Testing {model_type} Model")
        log.info(f"{'='*80}")
        
        # Generate data
        start_time = time.perf_counter()
        RT, X, IDs = generate_synthetic_data(NUM_PARTICIPANTS, NUM_TRIALS, NUM_STATES,
                                             MEASUREMENT_PROB, DELTA, model_type=model_type)
        
        # Save data to CSV files
        data_dir = save_synthetic_data(RT, X, IDs)
        data_gen_time = time.perf_counter() - start_time
        log.info(f"Data generation completed in {data_gen_time:.2f} seconds")
        
        # Create model configuration - use file-based data loading
        # The file_pre should match the saved file prefix
        # Use empty string as key so filename is synthetic_rt.csv (not synthetic<key>_rt.csv)
        model_details = fm.ModelDetails(
            folder=data_dir,
            file_pre="synthetic",  # Files named synthetic_rt.csv and synthetic_ra.csv
            data={"": None},  # Empty key - file loading looks for synthetic_rt.csv, synthetic_ra.csv
            version=0.1,
            n_states=[NUM_STATES],  # Must be list for zipping
            start_width=None,
            response_width=[1],  # Must be list for zipping
            delta=DELTA,
            measurement_prob=MEASUREMENT_PROB,
            predictive_n=10,
            batch_size=None,
            num_warmup=NUM_WARMUP,
            samples_n=SAMPLES_N,
            params_type="NonCentralized",
            model_type=[model_type],  # Already correct
            transition_type="TIMESTEP",
            likelihood_type="SINGLE",
            sampling_type="GEN",
            estimation_type="MCMC",
            execution_type="Both",
            scale=None,
            csv_header=True,  # Now True since we save with proper headers
            is_test=False,
            is_parallel=False,
            conf_scale=[None],  # Must be list for zipping
            # NEW scalable parameters
            subsample_size=SUBSAMPLE_SIZE,
            use_hmcecs=USE_HMCECS,
        )
        
        # Run fitting
        log.info(f"\nStarting MCMC fitting for {NUM_PARTICIPANTS} participants...")
        start_time = time.perf_counter()
        
        try:
            fm.fit_model(model_details)
            fitting_time = time.perf_counter() - start_time
            log.info(f"✓ {model_type} fitting completed in {fitting_time:.2f} seconds "
                    f"({fitting_time/60:.2f} minutes)")
        except Exception as e:
            fitting_time = time.perf_counter() - start_time
            log.error(f"✗ {model_type} fitting failed after {fitting_time:.2f} seconds")
            log.error(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    log.info("\n" + "="*80)
    log.info("TEST COMPLETED")
    log.info("="*80)


# ============================================================================
# Scaling Recommendations
# ============================================================================

"""
SCALING RECOMMENDATIONS:

For NUM_PARTICIPANTS = 3:
  - Use standard NUTS kernel
  - No subsampling
  - Expected time: ~1-2 minutes

For NUM_PARTICIPANTS = 10:
  - Use standard NUTS kernel
  - No subsampling
  - Expected time: ~5-10 minutes

For NUM_PARTICIPANTS = 50:
  - Consider HMCECS with subsampling
  - SUBSAMPLE_SIZE = None (auto-detect)
  - Expected time: ~10-20 minutes

For NUM_PARTICIPANTS = 100:
  - Use HMCECS with subsampling
  - SUBSAMPLE_SIZE = 50-100
  - Expected time: ~20-40 minutes

For NUM_PARTICIPANTS = 1000:
  - Use HMCECS with heavy subsampling
  - SUBSAMPLE_SIZE = 100
  - Consider increasing NUM_CHAINS
  - Expected time: ~1-3 hours

KEY IMPROVEMENTS FOR QUANTUM CONVERGENCE:
  1. Better priors in non_centralized_parameters (now model-specific)
  2. Improved softplus scaling (avoids numerical issues)
  3. HMCECS kernel (better empirical centering)
  4. Automatic subsampling detection
  
If Quantum model still doesn't converge:
  1. Increase NUM_WARMUP to 1000-2000
  2. Increase SAMPLES_N to 1000-2000
  3. Reduce step_size (handled by MCMC automatically)
  4. Try with fewer states initially (e.g., NUM_STATES = 7)
"""


if __name__ == "__main__":
    test_scalable_fitting()
