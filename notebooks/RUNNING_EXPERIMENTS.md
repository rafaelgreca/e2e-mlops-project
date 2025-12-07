# Running Training Experiments using script - Guide

### Get Container ID

```bash
docker ps
# Find the container with Jupyter (usually named e2e-mlops-project-dev-1)
```

### Run the Script

```bash
# Enter container
docker exec -it <DEV_CONTAINER_ID> bash

# Go to notebooks directory
cd /e2e-project/notebooks

# Run in background with logging
nohup python3 -u run_experiments.py > training.log 2>&1 &

# You'll see: [1] <process_id>
```

### Step 3: Monitor Progress

**Best Method - MLflow UI:**

- Open `http://localhost:5000` in browser
- See experiments appear in real-time
- Compare metrics as they're logged

**Alternative - Log File:**

```bash
# Inside container
training.log
```

### Step 4: Check Status Anytime

````bash
# Reconnect to container (if disconnected)
docker exec -it <DEV_CONTAINER_ID> bash

# Check if still running
ps aux | grep python

## What Gets Trained
The script automatically runs:

1. **Feature Selection Experiments** (~124 experiments)
   - 4 models × ~31 feature counts
   - DecisionTree, RandomForest, XGBoost, LightGBM

2. **Hyperparameter Tuning** (~400 experiments)
   - 4 models × 100 trials each
   - Optuna optimization

**Total: ~524 experiments**

## Monitoring Progress

**Real-time Monitoring:**
- Open MLflow UI: `http://localhost:5000`
- Navigate to experiments tab
- See runs appear with metrics as they complete
- No need to wait for script to finish!

**Check Logs:**
```bash
docker exec -it <CONTAINER_ID> bash
cd /e2e-project/notebooks
tail -f training.log
````
