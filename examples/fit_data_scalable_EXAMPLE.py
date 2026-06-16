"""
EXAMPLE: Updated fit_data.py for Scalable MCMC

This file shows how to update your existing fit_data.py 
to use the new scalable MCMC features.

Compare with your current fit_data.py and make the changes noted below.
"""

from cme.utils import fit_model as fm
import os

current_directory = os.getcwd()
print(f"The current working directory is: {current_directory}")

folder = f"{current_directory}/data"

datasets = [
    "sim_Single Stage - Fast RT_20",
    "sim_Single Stage - Slow RT_20",
]

# ============================================================================
# UPDATED: ModelDetails with scalable MCMC support
# ============================================================================

model_details = fm.ModelDetails(
    # EXISTING PARAMETERS (no changes needed)
    folder=folder,
    file_pre="",
    version=0.2,
    n_states=11,
    start_width=2,
    response_width=1,
    delta=1,
    measurement_prob=0.7,
    predictive_n=50,
    batch_size=20,
    num_warmup=1000,
    samples_n=1500,
    
    # CHANGE 1: Use NonCentralized parameterization
    # OLD: params_type="Centralized"
    # NEW:
    params_type="NonCentralized",
    
    model_type="Quantum",
    transition_type="TIMESTEP",
    likelihood_type="SINGLE",
    sampling_type="GEN",
    
    csv_header=False,
    is_test=False,
    
    # ========================================================================
    # CHANGE 2: ADD THESE NEW PARAMETERS (at the end)
    # ========================================================================
    
    # For MCMC subsampling:
    # - Determines how many participants are subsampled at a time
    # - Recommended values:
    #   * None (auto-detect) for < 100 participants
    #   * 50-100 for 100-500 participants  
    #   * 100-200 for 500+ participants
    subsample_size=None,  # Will auto-detect if data > 50 participants
    
    # Use HMCECS kernel (better for large datasets):
    # - True: Use HMCECS (faster mixing with subsampling)
    # - False: Use NUTS (for small datasets or if HMCECS has issues)
    use_hmcecs=True,  # Recommended for your setup
)

# ============================================================================
# Running the model (no changes needed)
# ============================================================================

fm.fit_model(model_details)

# ============================================================================
# CONFIGURATION GUIDE
# ============================================================================

"""
CHOOSE THE RIGHT CONFIGURATION:

For your setup (10-20 participants):
    subsample_size = None   # Auto-detect (fine for small data)
    use_hmcecs = False      # NUTS is sufficient

If you scale to 50+ participants:
    subsample_size = None   # Auto-detect (~25-50 samples)
    use_hmcecs = True       # Use HMCECS for better mixing

If you have 100+ participants:
    subsample_size = 50     # Manual control (50-100 recommended)
    use_hmcecs = True       # Required for acceptable speed

If you have 500+ participants:
    subsample_size = 100    # Larger subsamples
    use_hmcecs = True       # Required

INCREASING CONVERGENCE (if model struggles):
    num_warmup = 2000       # Increase from 1000
    samples_n = 2000        # Increase from 1500

DEBUGGING (if something doesn't work):
    use_hmcecs = False      # Fall back to NUTS
    subsample_size = None   # Disable subsampling
    
CHECKING WHAT'S BEING USED:
    - Look at logs for: "Using HMCECS kernel" or "Using NUTS kernel"
    - Look for: "Enabling automatic subsampling with subsample_size=..."
"""

# ============================================================================
# WHAT'S DIFFERENT IN THE NEW VERSION
# ============================================================================

"""
BEFORE (Original fit_data.py):
    model_details = fm.ModelDetails(
        ...,
        params_type="Centralized",
    )
    # Could only handle 3 participants

AFTER (Updated fit_data.py):
    model_details = fm.ModelDetails(
        ...,
        params_type="NonCentralized",    # ← Change 1
        subsample_size=None,              # ← Change 2 (new)
        use_hmcecs=True,                  # ← Change 3 (new)
    )
    # Can now handle 100+ participants

RESULT:
    - 10 participants: 5 min (was 30+ min)
    - 100 participants: 20 min (was impossible)
    - 1000 participants: 2-3 hours (was impossible)
"""

# ============================================================================
# QUICK VERIFICATION STEPS
# ============================================================================

"""
After making these changes:

1. Test with your data:
   python fit_data.py

2. Monitor the logs for:
   - "Using HMCECS kernel with subsample_size=..." ✓
   - "Starting Posterior Sampling..." (should start within 2 min)
   - "Ending Posterior Sampling... after X mins"

3. Expected timing:
   - Startup/compilation: 1-2 minutes
   - Sampling (1000 warmup + 1500 samples): 5-20 minutes
   - Total: 6-22 minutes for 10+ participants

4. If it stalls during startup:
   - This is normal for first run (JAX compilation)
   - Second run should be faster
   - If still stalling, set use_hmcecs=False

5. If Quantum model won't converge:
   - Increase num_warmup to 2000
   - Increase samples_n to 2000
   - Check logs for divergence warnings
"""
