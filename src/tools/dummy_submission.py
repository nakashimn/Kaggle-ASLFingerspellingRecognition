import datetime
import os
import pathlib
import shutil
import sys
import traceback
from typing import Any

import numpy as np
import onnx
import onnx_tf
import onnx2keras
import onnxruntime
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import torch
from numpy.typing import NDArray

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from components.models import DummyASLModel
from config.sample import config

dim_input = config["model"]["dim_input"]
class TFLiteModel(tf.Module):
    def __init__(self, model):
        super(TFLiteModel, self).__init__()
        self.infer = model.signatures["serving_default"]

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, dim_input], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs, training=False):
        # Preprocess Data
        x = tf.cast(inputs, tf.float32)
        x = x[None]
        x = tf.identity(x)
        x = tf.cond(
            tf.shape(x)[1] == 0,
            lambda: tf.zeros((1, 1, dim_input)),
            lambda: tf.identity(x),
        )
        x = self.infer(x)
        return {'outputs': x["outputs"][0]}

if __name__ == "__main__":

    # path
    dirpath_model: str = "/kaggle/input/dummy_model/"
    onnx_model_path: str = f"{dirpath_model}/model.onnx"
    tf_model_path: str = f"{dirpath_model}/model.tf"
    tflite_basemodel_path: str = f"{dirpath_model}/basemodel_for_tflite.tf"
    tflite_model_path: str = f"{dirpath_model}/model.tflite"

    # preprocess
    if os.path.exists(onnx_model_path):
        os.remove(onnx_model_path)
    if os.path.exists(tf_model_path):
        shutil.rmtree(tf_model_path)
    if os.path.exists(tflite_basemodel_path):
        shutil.rmtree(tflite_basemodel_path)
    if os.path.exists(tflite_model_path):
        os.remove(tflite_model_path)

    # main
    # convert torch to onnx
    inputs_sample: torch.Tensor = torch.randn(1, 1, 184).cuda()
    inputs_sample: torch.Tensor = torch.randn(1, 184).cuda()
    model = DummyASLModel(config["model"]).cuda()
    torch.onnx.export(
        model,
        inputs_sample,
        onnx_model_path,
        opset_version=12,
        export_params=True,
        input_names=["inputs"],
        output_names=["outputs"],
        dynamic_axes={
            "inputs": {0: "length"},
            "outputs": {0: "length"},
        },
    )

    # convert onnx to tf
    onnx_model = onnx.load(onnx_model_path)
    tf_model = onnx_tf.backend.prepare(onnx_model)
    tf_model.export_graph(tf_model_path)

    # convert tf to tflite_base(with_preprocess)
    tflite_basemodel = TFLiteModel(tf.saved_model.load(tf_model_path))
    tf.saved_model.save(tflite_basemodel, tflite_basemodel_path)

    # convert tflite_bese to tflite
    # converter = tf.lite.TFLiteConverter.from_saved_model(tflite_basemodel_path)
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    ####################################################################################
    ### debug
    ####################################################################################
    # dummy_input
    normal_inputs: NDArray = np.random.randn(10, 184).astype(np.float32)
    single_input: NDArray = np.random.randn(1, 184).astype(np.float32)
    null_input: NDArray = np.random.randn(0, 184).astype(np.float32)
    dim1_inputs: NDArray = np.random.randn(184).astype(np.float32)
    nan_inputs: NDArray = normal_inputs.copy()
    nan_inputs[0, :20] = torch.nan

    # tflite
    signatures: str = "serving_default"
    output_name: str = "outputs"
    interpreter = tflite.Interpreter(f"{tflite_model_path}")
    prediction_fn = interpreter.get_signature_runner(signatures)

    result_normal_inputs = prediction_fn(inputs=normal_inputs)[output_name]
    print(f"normal_inputs: {result_normal_inputs.shape}")

    result_single_input = prediction_fn(inputs=single_input)[output_name]
    print(f"single_input: {result_single_input.shape}")

    result_null_input = prediction_fn(inputs=null_input)[output_name]
    print(f"null_input: {result_null_input.shape}")

    result_dim1_inputs = prediction_fn(inputs=dim1_inputs)[output_name]
    print(f"dim1_inputs: {result_dim1_inputs.shape}")

    result_nan_inputs = prediction_fn(inputs=nan_inputs)[output_name]
    print(f"nan_inputs: {result_nan_inputs.shape}",
          f"(hasnan: {np.isnan(result_nan_inputs).any()})")
