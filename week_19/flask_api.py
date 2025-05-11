from flask import Flask, jsonify, request
import numpy as np
import onnxruntime as rt


app = Flask(__name__)

@app.route("/invocations", methods=["POST"])

def invocations():
    payload = request.get_json()
    data = np.array(payload["data"]).astype(np.float32)

    # Load the ONNX model
    onnx_model_path = "/model_artifacts/rf_iris.onnx"
    sess = rt.InferenceSession(onnx_model_path, providers= ["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    prediction = sess.run([label_name], {input_name: data})[0]

    response = {"Prediction": prediction.tolist()}

    return jsonify(response), 200

if __name__ =="__main__":
    app.run(host="0.0.0.0", port=8080)
