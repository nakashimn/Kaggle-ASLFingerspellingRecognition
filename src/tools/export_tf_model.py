import argparse
import datetime
import importlib
import json
import os
import pathlib
import shutil
import sys
import traceback
import zipfile
from argparse import Namespace
from typing import Any

import numpy as np
import onnx
import onnx_tf
import onnxruntime
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import torch
from numpy.typing import NDArray
from tensorflow.dtypes import DType
from torch import nn
from pytorch_lightning import LightningModule

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))


def display_now() -> str:
    return datetime.datetime.now().strftime("%Y/%m/%d(%a) %H:%M:%S")


def convert_torch_to_onnx(
    torch_model_path: str,
    onnx_model_path: str,
    Model: LightningModule,
    config: dict[str, Any],
    *,
    device: str = "cpu",
    opset_version: int = 12,
) -> None:
    print("\n\33[32m[convert torch to onnx]\33[0m")
    print(f"  start: {display_now()}")
    try:
        model: nn.Module = Model.load_from_checkpoint(
            torch_model_path,
            config=config["model"],
        ).to(device)
        inputs_sample: torch.Tensor = torch.randn(1, 10, config["model"]["dim_input"]).to(
            device
        )
        torch.onnx.export(
            model,
            inputs_sample,
            onnx_model_path,
            opset_version=opset_version,
            export_params=True,
            input_names=["inputs"],
            output_names=["outputs"],
            dynamic_axes={
                "inputs": {1: "length"},
                "outputs": {1: "length"},
            },
        )
    except:
        print(f"\33[31m{traceback.format_exc()}\33[0m")
        raise
    print(f"  end: {display_now()}")


def convert_onnx_to_tf(onnx_model_path: str, tf_model_path: str) -> None:
    print("\n\33[32m[convert onnx to tf]\33[0m")
    print(f"  start: {display_now()}")
    try:
        onnx_model = onnx.load(onnx_model_path)
        tf_model = onnx_tf.backend.prepare(onnx_model)
        tf_model.export_graph(tf_model_path)
    except:
        print(f"\33[31m{traceback.format_exc()}\33[0m")
        raise
    print(f"  end: {display_now()}")

def convert_tf_to_tflitebase(
    tf_model_path: str,
    tflite_basemodel_path: str,
    dim_input: int,
):
    print("\n\33[32m[convert tf to tflitebase]\33[0m")
    print(f"  start: {display_now()}")

    class TFLiteModel(tf.Module):
        def __init__(self, model):
            super(TFLiteModel, self).__init__()
            self.infer = model.signatures["serving_default"]

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, dim_input], dtype=tf.float32, name='inputs')])
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

    try:
        tflite_basemodel = TFLiteModel(tf.saved_model.load(tf_model_path))
        tf.saved_model.save(tflite_basemodel, tflite_basemodel_path)
    except:
        print(f"\33[31m{traceback.format_exc()}\33[0m")
        raise
    print(f"  end: {display_now()}")

def convert_tf_to_tflite(
    tf_model_path: str,
    tflite_model_path: str,
    *,
    dtype: str = "float32",
):
    print("\n\33[32m[convert tf to tflite]\33[0m")
    print(f"  start: {display_now()}")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [eval(f"tf.{dtype}")]
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            # tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        tflite_model = converter.convert()
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)
    except:
        print(f"\33[31m{traceback.format_exc()}\33[0m")
        raise
    print(f"  end: {display_now()}")


def write_json(filepath, data):
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except:
        print(f"\33[31m{traceback.format_exc()}\33[0m")


def define_inference_args(config: dict[str, Any]) -> dict[str, list[str]]:
    inference_args: dict[str, list[str]] = {
        "selected_columns": config["datamodule"]["dataset"]["select_col"]
    }
    return inference_args


def create_dummy_inputs(length: int, dim: int, dtype: str) -> NDArray:
    dummy_inputs: NDArray = np.random.randn(length, dim).astype(eval(f"np.{dtype}"))
    return dummy_inputs


def run_onnx(
    onnx_model_path: str,
    inputs: NDArray,
    *,
    input_name: str = "inputs",
) -> NDArray:
    try:
        ort_session = onnxruntime.InferenceSession(onnx_model_path)
        result = ort_session.run(None, {input_name: inputs})
        return result[0]
    except:
        print(f"\33[31m{traceback.format_exc()}\33[0m")
        raise


def run_tf(
    tf_model_path: str,
    inputs: NDArray,
    *,
    signatures: str = "serving_default",
    output_name: str = "outputs",
) -> NDArray:
    try:
        tf_model = tf.saved_model.load(tf_model_path)
        infer = tf_model.signatures[signatures]
        result = infer(tf.constant(inputs))
        return result[output_name].numpy()
    except:
        print(f"\33[31m{traceback.format_exc()}\33[0m")
        raise


def run_tflite(
    tflite_model_path: str,
    inputs: NDArray,
    *,
    signature: str = "serving_default",
    output_name: str = "outputs",
) -> NDArray:
    try:
        interpreter = tflite.Interpreter(f"{tflite_model_path}")
        prediction_fn = interpreter.get_signature_runner(signature)
        result = prediction_fn(inputs=inputs)
        return result[output_name]
    except:
        print(f"\33[31m{traceback.format_exc()}\33[0m")
        raise


def rmse(result_0: NDArray, result_1: NDArray) -> float:
    return np.sqrt(np.sum((result_0 - result_1) ** 2))


def import_classes(config: dict[str, Any]) -> LightningModule:
    # import Classes dynamically
    Model = getattr(
        importlib.import_module(f"components.models"), config["model"]["ClassName"]
    )
    return Model


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="stem of config filepath.", type=str, required=True
    )
    parser.add_argument(
        "-i",
        "--input",
        help='input modelname. if input is None, config["modelname"].ckpt is exported.',
        default=None,
    )
    parser.add_argument(
        "-d", "--debug", help="debugging exported models.", action="store_true"
    )
    return parser.parse_args()

def debug_preprocess():
    from config.sample import config
    debug = True
    modelname = config["modelname"]

if __name__ == "__main__":

    # args
    args = get_args()
    config = importlib.import_module(f"config.{args.config}").config

    # const
    device: str = config["pred_device"]
    dim_input: int = config["model"]["dim_input"]
    debug: bool = args.debug
    modelname: str = args.input or config["modelname"]

    # path
    dirpath_model: str = config["path"]["model_dir"]
    torch_model_path: str = f"{dirpath_model}/{modelname}.ckpt"
    onnx_model_path: str = f"{dirpath_model}/{modelname}.onnx"
    tf_model_path: str = f"{dirpath_model}/{modelname}.tf"
    tflite_basemodel_path: str = f"{dirpath_model}/basemodel_for_tflite.tf"
    tflite_model_path: str = f"{tf_model_path}/model.tflite"
    inference_args_path: str = f"{dirpath_model}/inference_args.json"
    submission_path: str = f"{dirpath_model}/submission.zip"

    # preprocess
    if os.path.exists(onnx_model_path):
        os.remove(onnx_model_path)
    if os.path.exists(tf_model_path):
        shutil.rmtree(tf_model_path)
    if os.path.exists(tflite_basemodel_path):
        shutil.rmtree(tflite_basemodel_path)
    if os.path.exists(inference_args_path):
        os.remove(inference_args_path)
    if os.path.exists(submission_path):
        os.remove(submission_path)
    Model = import_classes(config)

    # convert pytorch_checkpoint to tflite
    convert_torch_to_onnx(
        torch_model_path,
        onnx_model_path,
        Model,
        config,
        device=device,
    )
    convert_onnx_to_tf(onnx_model_path, tf_model_path)
    convert_tf_to_tflitebase(tf_model_path, tflite_basemodel_path, dim_input=dim_input)
    convert_tf_to_tflite(tflite_basemodel_path, tflite_model_path)
    # convert_tf_to_tflite(tf_model_path, tflite_model_path, dtype=dtype)
    inference_args: dict[str, list[str]] = define_inference_args(config)
    write_json(inference_args_path, inference_args)

    # output submission
    with zipfile.ZipFile(submission_path, "w") as z:
        z.write(tflite_model_path, arcname=pathlib.Path(tflite_model_path).name)
        z.write(inference_args_path, arcname=pathlib.Path(inference_args_path).name)

    # debug
    if debug:
        dummy_inputs: NDArray = create_dummy_inputs(
            length=np.random.randint(0, 10),
            dim=len(inference_args["selected_columns"]),
            dtype="float32",
        )
        result_onnx = run_onnx(onnx_model_path, dummy_inputs[None])
        result_tf = run_tf(tf_model_path, dummy_inputs[None])
        result_tflite = run_tflite(tflite_model_path, dummy_inputs)
        print("\n\33[32m[debug]\33[0m")
        print(f"  RMSE(onnx, tf): {rmse(result_onnx, result_tf):.3e}")
        print(f"  RMSE(tf, tflite): {rmse(result_tf, result_tflite):.3e}")
        print(f"  RMSE(onnx, tflite): {rmse(result_onnx, result_tflite):.3e}")
