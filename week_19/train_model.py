import json
import numpy as np

from skl2onnx import to_onnx
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()

X,y = iris.data, iris.target

X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = RandomForestClassifier()
clr.fit(X_train, y_train)

onx = to_onnx(clr, X[:1])
with open("docker-example/model_artifacts/rf_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

sample_payload = {"data": X_test.tolist()}
with open("docker-example/model_artifacts/sample_payload.json", "w") as f:
    json.dump(sample_payload, f)

