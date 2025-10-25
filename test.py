from pathlib import Path
import numpy as np

p = Path.home() / "OneDrive" / "Desktop" / "College" / "lockIn" / "testKeyframes" / "all_smoothed_angles.npy"
arr = np.load(p, allow_pickle=True)
print("path:", p)
print("shape:", arr.shape)
print("sample rows (first 3):\n", arr[:3])
