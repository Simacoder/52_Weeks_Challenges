import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from urllib.request import urlopen
import pickle

# Option A: Use a cached/alternative data source
def load_california_housing_alternative():
    """
    Load California housing data from an alternative source
    or use synthetic data as fallback
    """
    try:
        # Try downloading from an alternative URL
        url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
        data = pd.read_csv(url)
        
        # Prepare as sklearn format
        from sklearn.utils import Bunch
        X = data.drop('median_house_value', axis=1)
        y = data['median_house_value'] / 100000  # Normalize to match original
        
        return Bunch(data=X.values, target=y.values, 
                    feature_names=X.columns.tolist(),
                    target_names=['MedHouseVal'])
    except:
        # Fallback: create synthetic California housing-like data
        print("Using synthetic data (couldn't fetch from online sources)")
        np.random.seed(42)
        n_samples = 20640
        
        synthetic_data = np.random.randn(n_samples, 8) * [2, 15, 5, 1, 1000, 3, 5, 5] + \
                        [3.88, 35.4, 5.43, 1.1, 1427, 3.07, 35.6, -119.6]
        synthetic_target = (2 + 0.5*synthetic_data[:, 0] - 0.01*synthetic_data[:, 1] + 
                          np.random.randn(n_samples)*0.5)
        
        return Bunch(
            data=synthetic_data,
            target=synthetic_target,
            feature_names=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                          'Population', 'AveOccup', 'Latitude', 'Longitude'],
            target_names=['MedHouseVal']
        )

# Option B: If you have internet but behind a proxy
def load_with_retry():
    """Try multiple times with timeout"""
    from sklearn.datasets import fetch_california_housing
    import socket
    
    socket.setdefaulttimeout(30)  # 30 second timeout
    
    for attempt in range(3):
        try:
            print(f"Attempt {attempt+1}/3...")
            data = fetch_california_housing()
            return data
        except Exception as e:
            if attempt == 2:
                print("Failed to download. Using alternative source...")
                return load_california_housing_alternative()
            print(f"Attempt {attempt+1} failed: {e}")
            import time
            time.sleep(2)  # Wait before retry

# In your code, replace this line:
#   data = fetch_california_housing()
# With this:
data = load_california_housing_alternative()

print("Dataset loaded successfully!")
print(f"Shape: {data.data.shape}")
print(f"Features: {data.feature_names}")
# save local 
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target
df.to_csv("california_housing_alternative.csv", index=False)        