# TensorFlow Installation Fix for Python 3.14

## ⚠️ Problem
TensorFlow doesn't officially support Python 3.14 yet. The latest supported version is Python 3.12.

## ✅ Solutions (Choose One)

### Option 1: Use Python 3.11 with Conda (RECOMMENDED)

```powershell
# Create new environment with Python 3.11
conda create -n railway_env python=3.11 -y

# Activate environment
conda activate railway_env

# Install dependencies
pip install -r requirements.txt
pip install tensorflow

# Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"

# Configure Jupyter to use this environment
python -m ipykernel install --user --name railway_env --display-name "Python 3.11 (Railway)"
```

Then in VS Code/Jupyter:
- Select Kernel → "Python 3.11 (Railway)"

---

### Option 2: Try TensorFlow Nightly (Experimental)

```powershell
# Install nightly build (may have Python 3.14 support)
pip install tf-nightly

# Verify
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

**Warning**: Nightly builds are unstable and may have bugs.

---

### Option 3: Use sklearn's MLPClassifier (No TensorFlow)

The notebook already has a fallback! When TensorFlow isn't available, it automatically uses `sklearn.neural_network.MLPClassifier`.

**No action needed** - just run the cells and they'll work with sklearn.

**Pros**:
- ✅ Works immediately with Python 3.14
- ✅ No additional installation
- ✅ Good performance for most tasks

**Cons**:
- ❌ Less flexible than TensorFlow
- ❌ Slower for very large datasets
- ❌ Fewer advanced features

---

### Option 4: Downgrade Python (System-wide)

```powershell
# Uninstall Python 3.14
# Download and install Python 3.11 from python.org
# Then reinstall packages
pip install -r requirements.txt
```

---

## 🎯 Recommended Approach

**Use Option 1 (Conda Environment)** because:
- ✅ Keeps your Python 3.14 installation
- ✅ Isolated environment (no conflicts)
- ✅ Easy to switch between projects
- ✅ Full TensorFlow support
- ✅ Reproducible setup

---

## 📝 What's Changed in requirements.txt

TensorFlow lines are now commented out:
```python
# Deep Learning (Note: TensorFlow requires Python 3.8-3.12)
# tensorflow>=2.12.0
# keras>=2.12.0
```

**To install** (after switching to Python 3.11):
```powershell
pip install tensorflow keras
```

---

## 🚀 Quick Start (Recommended)

```powershell
# Step 1: Create Python 3.11 environment
conda create -n railway_env python=3.11 pandas numpy scikit-learn matplotlib seaborn jupyter -y

# Step 2: Activate
conda activate railway_env

# Step 3: Install remaining packages
pip install tensorflow keras shap imbalanced-learn plotly ipywidgets tqdm

# Step 4: Install Jupyter kernel
python -m ipykernel install --user --name railway_env --display-name "Python 3.11 (Railway)"

# Step 5: Verify
python -c "import tensorflow as tf; print('✅ TensorFlow', tf.__version__)"
```

**Then in VS Code:**
1. Open notebook
2. Click kernel selector (top right)
3. Select "Python 3.11 (Railway)"
4. Run cells normally

---

## 🔍 Check Your Python Version

```powershell
python --version
```

If it shows `Python 3.14.x`, use **Option 1** to create a Python 3.11 environment.

---

## ✅ Verification

After setup, verify everything works:

```python
import tensorflow as tf
import keras
import sklearn
import pandas as pd
import numpy as np

print(f"✅ TensorFlow: {tf.__version__}")
print(f"✅ Keras: {keras.__version__}")
print(f"✅ sklearn: {sklearn.__version__}")
print(f"✅ pandas: {pd.__version__}")
print(f"✅ numpy: {np.__version__}")
```

---

## 📞 Need Help?

If you get errors:
1. Check Python version: `python --version`
2. Check if in conda env: `conda info --envs`
3. Try reinstalling: `pip install --upgrade tensorflow`
4. Use sklearn fallback (already in notebook)

**The notebook will work regardless** - it falls back to sklearn automatically! 🎉
