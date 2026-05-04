#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a sklearn RandomForestClassifier and export it to ONNX."
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--n-trees", required=True, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    train_csv = Path(args.train_csv)
    model_output = Path(args.model_output)
    metrics_output = Path(args.metrics_output)

    data = np.loadtxt(train_csv, delimiter=",", skiprows=1, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError("training CSV must contain at least one feature column and one label column")

    x_train = data[:, :-1].astype(np.float32)
    y_train = data[:, -1].astype(np.int64)
    if x_train.shape[0] == 0:
        raise ValueError("training data is empty")

    start = time.perf_counter()
    model = RandomForestClassifier(
        n_estimators=args.n_trees,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(x_train, y_train)
    train_time_ms = int((time.perf_counter() - start) * 1000)

    train_predictions = model.predict(x_train)
    train_accuracy = float(np.mean(train_predictions == y_train))

    initial_types = [("float_input", FloatTensorType([None, x_train.shape[1]]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        target_opset=12,
        options={id(model): {"zipmap": False}},
    )

    model_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.write_bytes(onnx_model.SerializeToString())
    metrics_output.write_text(
        json.dumps(
            {
                "train_accuracy": train_accuracy,
                "train_time_ms": train_time_ms,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
