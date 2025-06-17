import numpy as np
import pandas as pd

np.random.seed(42)

# Simulating sensor signals
n = 500
temperature = np.random.normal(70, 5, n)  # Mean 70, SD 5
pressure = np.random.normal(30, 2, n)     # Mean 30, SD 2
vibration = np.random.normal(5, 1, n)     # Mean 5, SD 1

# Failure occurs if temperature > 80 or pressure > 35 or vibration > 7
failure = ((temperature > 80) | (pressure > 35) | (vibration > 7)).astype(int)

df = pd.DataFrame({"temperature": temperature, "pressure": pressure, "vibration": vibration, "failure": failure})

df.to_csv("data/sensor_data.csv", index=False)
print("Synthetic sensor data generated.")
