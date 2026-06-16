# ✅ NEXT STEPS: Action Items

## What You Need to Do (In Order)

### Step 1: Read (5 minutes)
- [ ] Open and read: `START_HERE.md`
- This tells you exactly what changed and how to proceed

### Step 2: Verify (5-10 minutes)  
- [ ] Run: `python test/test_fit_scalable.py`
- Verify the test completes without errors
- You should see timing and kernel selection in logs

### Step 3: Update Your Code (5 minutes)
- [ ] Open your: `examples/fit_data.py`
- [ ] Compare with: `examples/fit_data_scalable_EXAMPLE.py`
- [ ] Make 3 changes:
  - Change `params_type="Centralized"` → `params_type="NonCentralized"`
  - Add `subsample_size=None,`
  - Add `use_hmcecs=True,`

### Step 4: Test (5-15 minutes)
- [ ] Run your updated: `python examples/fit_data.py`
- [ ] Check logs for "Starting Posterior Sampling..."
- [ ] Verify it completes in reasonable time (5-20 min for 10+ participants)

### Step 5: Scale (Optional)
- [ ] Increase to 50 participants (should take 10-20 min)
- [ ] Increase to 100 participants (should take 20-40 min)
- [ ] Adjust `subsample_size` if needed

---

## Files to Reference

**READ FIRST:**
- `START_HERE.md` ← Start here (5 min)

**FOR YOUR CODE:**
- `examples/fit_data_scalable_EXAMPLE.py` ← Copy the 3 changes

**TROUBLESHOOTING:**
- `QUICKSTART.md` ← Configuration & common issues
- `SCALABLE_MCMC_IMPROVEMENTS.md` ← Technical details

**FOR VERIFICATION:**
- `test/test_fit_scalable.py` ← Run this to test

---

## Expected Outcomes

After completing these steps:

✓ Your code will fit 100+ participants (was impossible)  
✓ Quantum model convergence will improve  
✓ No more JAX compilation stalls  
✓ Auto-detection of optimal settings  
✓ Backward compatible with old code  

---

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Test doesn't run | Check you're in the right directory |
| Test takes too long | Normal for first run (JAX compilation) |
| Quantum model won't converge | Increase `num_warmup=2000` |
| Something else | Check QUICKSTART.md section "Common Issues" |

---

## Time Estimates

- Read docs: 5 min
- Run test: 5-10 min
- Update code: 5 min  
- Test with your data: 5-20 min
- **Total: 20-50 minutes**

---

## That's It!

You're now ready to:
1. Read `START_HERE.md`
2. Run the test
3. Update your code
4. Scale to 100+ participants

👉 **Next: Open `START_HERE.md` and follow the 4 steps**

Questions? Everything is documented in the files mentioned above.

Good luck! 🚀
