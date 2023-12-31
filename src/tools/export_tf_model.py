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
    model: nn.Module,
    inputs_sample: tuple[torch.Tensor] | torch.Tensor,
    input_names: list[str] | str,
    output_names: list[str] | str,
    dynamic_axes: dict[str, dict[int, str]],
    onnx_model_path: str,
    *,
    opset_version: int = 12,
) -> None:
    print(f"\n\33[32m[convert torch to onnx]\33[0m")
    print(f"  \n\33[32m-> {onnx_model_path}\33[0m")
    print(f"  start: {display_now()}")
    try:
        torch.onnx.export(
            model,
            inputs_sample,
            onnx_model_path,
            opset_version=opset_version,
            export_params=True,
            do_constant_folding=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    except:
        print(f"\33[31m{traceback.format_exc()}\33[0m")
        raise
    print(f"  end: {display_now()}")


def convert_onnx_to_tf(onnx_model_path: str, tf_model_path: str) -> None:
    print(f"\n\33[32m[convert onnx to tf]\33[0m")
    print(f"  \n\33[32m-> {tf_model_path}\33[0m")
    print(f"  start: {display_now()}")
    try:
        onnx_model = onnx.load(onnx_model_path)
        tf_model = onnx_tf.backend.prepare(onnx_model)
        tf_model.export_graph(tf_model_path)
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
    print(f"\n\33[32m[convert tf to tflite]\33[0m")
    print(f"  \n\33[32m-> {tflite_model_path}\33[0m")
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


def construct_tflitebase_for_asl(
    tf_preprocessor_path: str,
    tf_encoder_path: str,
    tf_decoder_path: str,
    tf_linear_path: str,
    bos_embed: NDArray,
    max_phrase_length: int,
    eos_token_id: int,
    dim_input: int,
    tflite_basemodel_path: str,
):

    print(f"\n\33[32m[convert tf to tflitebase]\33[0m")
    print(f"  \n\33[32m-> {tflite_basemodel_path}\33[0m")
    print(f"  start: {display_now()}")

    class TFLiteModel(tf.Module):
        def __init__(
                self,
                preprocessor,
                encoder,
                decoder,
                linear,
                bos_embed: NDArray,
                max_phrase_length: int,
                eos_token_id: int,
            ) -> None:
            super(TFLiteModel, self).__init__()
            self.preprocessor = preprocessor.signatures["serving_default"]
            self.encoder = encoder.signatures["serving_default"]
            self.decoder = decoder.signatures["serving_default"]
            self.linear = linear.signatures["serving_default"]
            self.bos_embed: tf.Tensor = tf.constant(bos_embed, tf.float32)
            self.max_phrase_length: int = max_phrase_length
            self.eos_token_id: int = eos_token_id

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, dim_input], dtype=tf.float32, name='inputs')])
        def __call__(self, inputs, training=False):
            # Preprocess Data
            x: tf.Tensor = tf.cast(inputs, tf.float32)
            x = x[None]
            x = tf.cond(
                tf.shape(x)[1] == 0,
                lambda: tf.zeros((1, 1, dim_input)),
                lambda: tf.identity(x),
            )
            x = self.preprocessor(x)["outputs"]
            src: tf.Tensor = self.encoder(x)["encoder_hidden_state"]
            tgt: tf.Tensor = self.bos_embed

            stop: bool = False
            for _ in range(self.max_phrase_length):
                if stop:
                    continue
                decoder_hidden_state: tf.Tensor = self.decoder(
                    source=src, target=tgt)["outputs"]
                tgt = tf.concat([self.bos_embed, decoder_hidden_state], axis=1)
                stop = self._is_eos(decoder_hidden_state)
            outputs: tf.Tensor = self.linear(tgt[:, 1:, :])["outputs"]
            return {'outputs': outputs[0]}

        def _is_eos(self, decoder_hidden_state: tf.Tensor) -> bool:
            tail_id: int = tf.math.argmax(
                self.linear(decoder_hidden_state)["outputs"], axis=1,
            )[0, -1]
            return (tail_id  == self.eos_token_id)

    try:
        tflite_basemodel: TFLiteModel = TFLiteModel(
            tf.saved_model.load(tf_preprocessor_path),
            tf.saved_model.load(tf_encoder_path),
            tf.saved_model.load(tf_decoder_path),
            tf.saved_model.load(tf_linear_path),
            bos_embed,
            max_phrase_length,
            eos_token_id,
        )
        tf.saved_model.save(tflite_basemodel, tflite_basemodel_path)
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
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
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
    from config.asl_trial_v2 import config
    debug = False
    modelname = config["modelname"]

if __name__ == "__main__":

    # args
    args: argparse.Namespace = get_args()
    config: dict[str, Any] = importlib.import_module(f"config.{args.config}").config

    # const
    device: str = config["pred_device"]
    dim_input: int = config["model"]["Transformer"]["dim_model"]
    debug: bool = args.debug
    modelname: str = args.input or config["modelname"]

    # path
    dirpath_model: str = config["path"]["model_dir"]
    torch_model_path: str = f"{dirpath_model}/{modelname}.ckpt"
    onnx_model_paths: dict[str, str] = {
        "preprocessor": f"{dirpath_model}/preprocessor.onnx",
        "encoder": f"{dirpath_model}/encoder.onnx",
        "decoder": f"{dirpath_model}/decoder.onnx",
        "linear": f"{dirpath_model}/linear.onnx",
    }
    tf_model_paths: dict[str, str] = {
        "preprocessor": f"{dirpath_model}/preprocessor.tf",
        "encoder": f"{dirpath_model}/encoder.tf",
        "decoder": f"{dirpath_model}/decoder.tf",
        "linear": f"{dirpath_model}/linear.tf",
    }
    tflite_basemodel_path: str = f"{dirpath_model}/basemodel_for_tflite.tf"
    tflite_model_path: str = f"{dirpath_model}/model.tflite"
    inference_args_path: str = f"{dirpath_model}/inference_args.json"
    submission_path: str = f"{dirpath_model}/submission.zip"

    # preprocess
    for onnx_model_path in onnx_model_paths.values():
        if os.path.exists(onnx_model_path):
            os.remove(onnx_model_path)
    for tf_model_path in tf_model_paths.values():
        if os.path.exists(tf_model_path):
            shutil.rmtree(tf_model_path)
    if os.path.exists(tflite_basemodel_path):
        shutil.rmtree(tflite_basemodel_path)
    if os.path.exists(inference_args_path):
        os.remove(inference_args_path)
    if os.path.exists(submission_path):
        os.remove(submission_path)
    Model: type = import_classes(config)

    model: nn.Module = Model.load_from_checkpoint(
        torch_model_path, config=config["model"])

    # const
    bos_embed: torch.Tensor = model.transformer.embedding(
        torch.tensor([[config["special_token_ids"]["bos_token_id"]]]).cuda())
    max_phrase_length: int = config["max_phrase_length"]
    eos_token_id: int = config["special_token_ids"]["eos_token_id"]

    # models
    preprocessor: nn.Module = model.preprocessor
    encoder: nn.Module = model.transformer.encoder
    decoder: nn.Module = model.transformer.decoder
    linear: nn.Module = model.transformer.linear

    # onnx settings
    src_embed_sample: torch.Tensor = torch.randn(
        1, 10, config["model"]["Transformer"]["dim_model"]).cuda()
    tgt_embed_sample: torch.Tensor = torch.randn(
        1, 10, config["model"]["Transformer"]["dim_model"]).cuda()
    input_names: dict[str, list[str]] = {
        "preprocessor": ["inputs"],
        "encoder": ["inputs"],
        "decoder": ["source", "target"],
        "linear": ["inputs"],
    }
    output_names: dict[str, list[str]] = {
        "preprocessor": ["outputs"],
        "encoder": ["encoder_hidden_state"],
        "decoder": ["outputs"],
        "linear": ["outputs"],
    }
    dynamic_axes: dict[str, dict[str, dict[int, str]]] = {
        "preprocessor": {
            "inputs": {1: "length"},
            "outputs": {1: "length"},
        },
        "encoder": {
            "inputs": {1: "length"},
            "encoder_hidden_state": {1: "length"},
        },
        "decoder": {
            "source": {1: "length"},
            "target": {1: "length"},
            "outputs": {1: "length"},
        },
        "linear": {
            "inputs": {1: "length"},
            "outputs": {1: "length"},
        },
    }

    # convert pytorch_checkpoint to tflite
    convert_torch_to_onnx(
        preprocessor,
        src_embed_sample,
        input_names=input_names["preprocessor"],
        output_names=output_names["preprocessor"],
        dynamic_axes=dynamic_axes["preprocessor"],
        onnx_model_path=onnx_model_paths["preprocessor"],
    )
    convert_torch_to_onnx(
        encoder,
        src_embed_sample,
        input_names=input_names["encoder"],
        output_names=output_names["encoder"],
        dynamic_axes=dynamic_axes["encoder"],
        onnx_model_path=onnx_model_paths["encoder"],
    )
    convert_torch_to_onnx(
        decoder,
        (src_embed_sample, tgt_embed_sample),
        input_names=input_names["decoder"],
        output_names=output_names["decoder"],
        dynamic_axes=dynamic_axes["decoder"],
        onnx_model_path=onnx_model_paths["decoder"],
    )
    convert_torch_to_onnx(
        linear,
        tgt_embed_sample,
        input_names=input_names["linear"],
        output_names=output_names["linear"],
        dynamic_axes=dynamic_axes["linear"],
        onnx_model_path=onnx_model_paths["linear"],
    )

    # convert onnx to tensorflow
    convert_onnx_to_tf(onnx_model_paths["preprocessor"], tf_model_paths["preprocessor"])
    convert_onnx_to_tf(onnx_model_paths["encoder"], tf_model_paths["encoder"])
    convert_onnx_to_tf(onnx_model_paths["decoder"], tf_model_paths["decoder"])
    convert_onnx_to_tf(onnx_model_paths["linear"], tf_model_paths["linear"])

    # constract submission_model
    construct_tflitebase_for_asl(
        tf_model_paths["preprocessor"],
        tf_model_paths["encoder"],
        tf_model_paths["decoder"],
        tf_model_paths["linear"],
        bos_embed.detach().cpu().numpy(),
        max_phrase_length,
        eos_token_id,
        dim_input,
        tflite_basemodel_path,
    )

    # convert tensorflow to tflite
    convert_tf_to_tflite(tflite_basemodel_path, tflite_model_path)
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
        dummy_inputs_torch: torch.Tensor = torch.tensor(
            dummy_inputs, dtype=torch.float32).to(device)
        result_torch = model(dummy_inputs_torch[None])
        result_tflite = run_tflite(tflite_model_path, dummy_inputs)
        print("\n\33[32m[debug]\33[0m")
        print(f"  RMSE(torch, tflite): {rmse(result_torch.detach().cpu().numpy(), result_tflite):.3e}")
