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
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    train_csv = Path(args.train_csv)
    model_output = Path(args.model_output)
    metrics_output = Path(args.metrics_output)

    # 使用 pandas 加载以便灵活处理空值
    try:
        import pandas as pd
        df = pd.read_csv(train_csv, dtype=np.float32)
        print(f"[Python] 加载前: {len(df)} 行")
        
        # 二次清理：删除任何包含 NaN 的行（防线机制）
        df = df.dropna()
        print(f"[Python] 加载后删除空值: {len(df)} 行")
        
        if len(df) == 0:
            raise ValueError("After dropna(), no data remaining in training CSV")
        
        x_train = df.iloc[:, :-1].values.astype(np.float32)
        y_train = df.iloc[:, -1].values.astype(np.int64)
    except ImportError:
        # 如果没有 pandas，回退到 numpy
        print("[Python] pandas 不可用，使用 numpy 加载")
        data = np.loadtxt(train_csv, delimiter=",", skiprows=1, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 2:
            raise ValueError("training CSV must contain at least one feature column and one label column")
        
        x_train = data[:, :-1].astype(np.float32)
        y_train = data[:, -1].astype(np.int64)
    
    if x_train.shape[0] == 0:
        raise ValueError("training data is empty")

    # 验证数据完整性
    if np.isnan(x_train).any() or np.isnan(y_train).any():
        raise ValueError("NaN 仍然存在于训练数据中，这表明 Rust 端的清洁出现问题")
    
    print(f"[Python] 训练数据: {x_train.shape[0]} 行 x {x_train.shape[1]} 列")
    
    start = time.perf_counter()
    rf_kwargs = dict(
        n_estimators=args.n_trees,
        n_jobs=-1,
        random_state=42,
    )
    if args.max_depth is not None:
        rf_kwargs["max_depth"] = args.max_depth
    if args.min_samples_leaf is not None:
        rf_kwargs["min_samples_leaf"] = args.min_samples_leaf
    model = RandomForestClassifier(**rf_kwargs)
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
