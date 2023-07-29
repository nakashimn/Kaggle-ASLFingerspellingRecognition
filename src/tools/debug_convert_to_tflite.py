import datetime
import os
import pathlib
import shutil
import sys
import traceback
from tqdm import tqdm
from typing import Any

import numpy as np
import onnx
import onnx_tf
import torch
import torch.optim as optim
import tflite_runtime.interpreter as tflite
import tensorflow as tf
import torchvision.transforms as T
from numpy.typing import NDArray
from torch import nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from components.models import Transformer

def display_now() -> str:
    return datetime.datetime.now().strftime("%Y/%m/%d(%a) %H:%M:%S")

def convert_torch_to_onnx(
    model: str,
    onnx_model_path: str,
    dim_input: int,
    *,
    device: str = "cpu",
    opset_version: int = 12,
) -> None:
    print("\n\33[32m[convert torch to onnx]\33[0m")
    print(f"  start: {display_now()}")
    try:
        inputs_sample: torch.Tensor = torch.randn(1, 10, dim_input).to(device)
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


########################################################################################
# model
########################################################################################
class DummyModel(nn.Module):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__()

        # const
        self.dim_model: int = 184
        self.dim_output: int = 62
        self.vocab_size: int = 61
        self.bos_idx: int = 58
        self.pad_idx: int = 60
        self.dim_ff: int = 256
        self.n_layer: int = 1
        self.n_head: int = 1
        self.max_length_encoder: int = 10000
        self.max_length_decoder: int = 100
        self.dropout_rate: float = 0.0
        self.layer_norm_eps: float = 1e-7
        self.device: str = device
        self.transformer = self._create_model()

    def _create_model(self) -> tuple[nn.Module]:
        transformer = Transformer(
            self.dim_model, self.dim_output, self.dim_ff,
            self.vocab_size, self.bos_idx, self.pad_idx,
            self.max_length_encoder, self.max_length_decoder,
            self.n_layer, self.n_head, self.dropout_rate,
            self.layer_norm_eps, self.device
        )
        return transformer

    def forward(
        self,
        enc_embeds: torch.Tensor,
        dec_indices: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        enc_embeds[torch.isnan(enc_embeds)] = 0.0
        output_embeds = self.transformer(enc_embeds, dec_indices, mask)
        return output_embeds



if __name__ == "__main__":
    # path
    dirpath_model: str = "/kaggle/input/dummy_model/"
    modelname: str = "dummy_model"
    onnx_model_path: str = f"{dirpath_model}/{modelname}.onnx"
    tf_model_path: str = f"{dirpath_model}/{modelname}.tf"
    tflite_basemodel_path: str = f"{dirpath_model}/basemodel_for_tflite.tf"
    tflite_model_path: str = f"{tf_model_path}/model.tflite"

    # preprocess
    if os.path.exists(onnx_model_path):
        os.remove(onnx_model_path)
    if os.path.exists(tf_model_path):
        shutil.rmtree(tf_model_path)
    if os.path.exists(tflite_basemodel_path):
        shutil.rmtree(tflite_basemodel_path)

    # dummy model
    device = "cpu"
    model = DummyModel(device)
    dummy_enc_embeds: torch.Tensor = torch.randn([1, 100, 184]).to(device)
    dummy_dec_indices: torch.Tensor = torch.randint(0, 61, [1, 30]).to(device)
    dummy_mask: torch.Tensor = torch.tensor([[0]*25 + [1]*75]).bool().to(device)

    model.eval()
    result = model(dummy_enc_embeds)
    print(f"inference: {result.shape}")
    train_result = model(dummy_enc_embeds, dummy_dec_indices)
    print(f"train: {train_result.shape}")
    train_result = model(dummy_enc_embeds, dummy_dec_indices, dummy_mask)
    print(f"train(with_mask): {train_result.shape}")

    # convert
    convert_torch_to_onnx(model, onnx_model_path, dim_input=184, device=device)
    convert_onnx_to_tf(onnx_model_path, tf_model_path)
    convert_tf_to_tflitebase(tf_model_path, tflite_basemodel_path, dim_input=184)
    convert_tf_to_tflite(tflite_basemodel_path, tflite_model_path)

    # debug
    dummy_input: torch.Tensor = dummy_enc_embeds[0].detach().numpy()
    result_tflite = run_tflite(tflite_model_path, dummy_input)
    print(f"RMSE: {rmse(result[0].detach().numpy(), result_tflite):.03e}")
