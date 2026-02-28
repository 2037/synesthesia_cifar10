# MPS Training Hang Fix

## Problem
Training was getting stuck at 40% (2/5 epochs) during the ablation study on Apple Silicon (MPS device).

## Root Causes
1. **Non-blocking transfers**: `non_blocking=True` with MPS can cause synchronization issues and hangs
2. **Memory accumulation**: MPS cache not being cleared, leading to memory pressure
3. **Progress bar overhead**: Too frequent updates overwhelming MPS
4. **Large batch size**: 128 batch size can be unstable on MPS

## Fixes Applied

### 1. **Changed to Blocking Transfers** (`src/trainer.py` & `src/experiments.py`)
```python
# Before (caused hangs):
x_batch = x_batch.to(device, non_blocking=cfg.PIN_MEMORY)

# After (stable):
x_batch = x_batch.to(device, non_blocking=False)
```

### 2. **Added MPS Memory Management**
```python
# Clear MPS cache every 20 batches to prevent memory buildup
if is_mps and batch_idx % 20 == 0:
    torch.mps.empty_cache()

# Final synchronization after epoch
if is_mps:
    torch.mps.synchronize()
    torch.mps.empty_cache()
```

### 3. **Reduced Progress Bar Update Frequency**
```python
# Update progress bar every 10 batches instead of every batch
if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
    bar.set_postfix(loss=f"{loss_val:.5f}")

# Increase minimum update interval for MPS
bar = tqdm(loader, mininterval=1.0 if is_mps else 0.1)
```

### 4. **Reduced Batch Size for MPS** (`src/config.py`)
```python
# Use smaller batch size on MPS for stability
BATCH_SIZE: int = 64 if str(DEVICE) == "mps" else 128
```

### 5. **Extract Loss Before Accumulation**
```python
# Ensure loss is computed and extracted immediately
loss_val = loss.item()  # Force synchronization
total_loss += loss_val * x_batch.size(0)
```

## Files Modified
- ✅ `src/trainer.py` - Fixed `_run_epoch()` function
- ✅ `src/experiments.py` - Fixed `_train_one_epoch()` function
- ✅ `src/config.py` - Reduced `BATCH_SIZE` for MPS

## Testing
After applying these fixes, the training should:
1. ✅ Not hang between batches or epochs
2. ✅ Complete all epochs successfully
3. ✅ Show stable memory usage
4. ✅ Display progress updates smoothly

## Additional Recommendations

### If Still Experiencing Issues:

1. **Further reduce batch size**:
   ```python
   BATCH_SIZE = 32  # in src/config.py
   ```

2. **Reduce model complexity temporarily**:
   ```python
   # Test with smaller model first
   model = ColorPredictor(dropout_rate=0.0)
   ```

3. **Monitor memory**:
   ```bash
   # Check memory usage in another terminal
   watch -n 1 'ps aux | grep python'
   ```

4. **Use CPU if MPS continues to have issues**:
   ```python
   # In src/config.py, force CPU:
   DEVICE = torch.device("cpu")
   ```

## Expected Behavior Now

Training should now proceed smoothly:
```
INFO | === Hyperparameter Sweep  target='B'  epochs=5  configs=4 ===
INFO |   ▶ 'No BN, No Dropout'               lr=0.01  BN=False  Drop=False  p=0.0
  No BN, No Dropout: 100%|██████████| 5/5 [02:15<00:00, 27.02s/epoch, train=0.01234, val=0.01123]
INFO |   ▶ 'BN only'                         lr=0.01  BN=True   Drop=False  p=0.0
  BN only: 100%|██████████| 5/5 [02:18<00:00, 27.64s/epoch, train=0.01156, val=0.01078]
...
```

## Why These Fixes Work

1. **Blocking transfers**: Ensures data is fully on device before proceeding
2. **Memory clearing**: Prevents MPS cache from growing unbounded
3. **Reduced updates**: Less overhead = more stable execution
4. **Smaller batches**: Reduces memory pressure and improves stability
5. **Explicit synchronization**: Ensures all MPS operations complete before continuing

---

## Quick Restart Instructions

1. **Stop the stuck process**: Press `Ctrl+C` in the terminal
2. **Clear any cached data**: 
   ```bash
   rm -rf models/*.pth  # Optional: remove old checkpoints
   ```
3. **Restart training**:
   ```bash
   python -m src.experiments --target B --epochs 5
   ```

The training should now complete without hanging! 🎉
