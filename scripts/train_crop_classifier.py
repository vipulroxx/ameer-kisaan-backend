"""Train the crop image classifier on the app's Agricultural-crops photo set.

    pip install -r requirements-train.txt
    python scripts/train_crop_classifier.py \
        --data ../Components/Agricultural-crops --out models

Transfer learning on MobileNetV2 (ImageNet weights), two phases: train a new
classification head, then fine-tune the top of the backbone. Produces:

    models/crop_classifier.tflite        model served by /api/v1/vision/identify-crop
    models/crop_classifier.labels.json   class names + version
    models/crop_classifier.metrics.json  held-out accuracy, per-class accuracy

Note: this dataset identifies *which crop* is in a photo (about 30 images per
class, so expect modest accuracy). Pest and disease detection needs a
labelled pest/disease photo set; the same script works on one, since each
sub-folder name becomes a class.
"""
import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMG = 224
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def pretty(name: str) -> str:
    name = re.sub(r"[-_]+", " ", name).strip()
    return name[:1].upper() + name[1:]


def collect(data_dir: Path, min_per_class: int):
    files, labels, skipped = [], [], 0
    classes = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    kept = []
    for c in classes:
        good = []
        for f in sorted((data_dir / c).iterdir()):
            if f.suffix.lower() not in EXTS:
                continue
            try:
                with Image.open(f) as im:
                    im.verify()
                good.append(str(f))
            except Exception:
                skipped += 1
        if len(good) >= min_per_class:
            kept.append(c)
            files += good
            labels += [len(kept) - 1] * len(good)
        else:
            print(f"skipping class {c!r}: only {len(good)} valid images")
    print(f"{len(files)} images in {len(kept)} classes ({skipped} unreadable files skipped)")
    return files, labels, kept


def split(files, labels, val_frac, seed):
    by = defaultdict(list)
    for f, label in zip(files, labels, strict=True):
        by[label].append(f)
    rng = random.Random(seed)
    tr, va = [], []
    for label, fs in by.items():
        rng.shuffle(fs)
        n_val = max(1, round(len(fs) * val_frac))
        va += [(f, label) for f in fs[:n_val]]
        tr += [(f, label) for f in fs[n_val:]]
    rng.shuffle(tr)
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--fine-tune-epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--min-per-class", type=int, default=8)
    ap.add_argument("--weights", default="imagenet", help="'imagenet' (needs internet once) or 'none'")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)

    files, labels, classes = collect(Path(args.data), args.min_per_class)
    train, val = split(files, labels, args.val_frac, args.seed)
    n_cls = len(classes)

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(tf.cast(img, tf.float32), (IMG, IMG))
        return img, label

    def ds(pairs, training):
        d = tf.data.Dataset.from_tensor_slices(([p for p, _ in pairs], [y for _, y in pairs]))
        if training:
            d = d.shuffle(len(pairs), seed=args.seed)
        return d.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch).prefetch(tf.data.AUTOTUNE)

    train_ds, val_ds = ds(train, True), ds(val, False)

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.15),
    ], name="augment")
    base = tf.keras.applications.MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False,
                                             weights=None if args.weights == "none" else args.weights)
    base.trainable = False

    inputs = tf.keras.Input((IMG, IMG, 3))  # raw 0-255 pixels; preprocessing is part of the model
    x = augment(inputs)
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_cls, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    counts = np.bincount([y for _, y in train], minlength=n_cls)
    class_weight = {i: len(train) / (n_cls * max(c, 1)) for i, c in enumerate(counts)}
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True)

    model.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, class_weight=class_weight, callbacks=[stop])

    if args.fine_tune_epochs:
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        model.compile(tf.keras.optimizers.Adam(1e-5), "sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, class_weight=class_weight,
                  callbacks=[stop])

    # Evaluate on the held-out split
    y_true, y_pred = [], []
    for xb, yb in val_ds:
        y_true += list(yb.numpy())
        y_pred += list(np.argmax(model.predict(xb, verbose=0), axis=1))
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    per_class = {pretty(classes[i]): round(float((y_pred[y_true == i] == i).mean()), 3)
                 for i in range(n_cls) if (y_true == i).any()}
    acc = float((y_true == y_pred).mean())
    print(f"held-out accuracy: {acc:.3f} on {len(y_true)} images")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tflite = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic-range quantisation: ~4x smaller
    (out / "crop_classifier.tflite").write_bytes(tflite.convert())
    version = time.strftime("%Y%m%d-%H%M%S")
    (out / "crop_classifier.labels.json").write_text(json.dumps(
        {"labels": [pretty(c) for c in classes], "image_size": IMG, "version": version}, indent=2))
    (out / "crop_classifier.metrics.json").write_text(json.dumps(
        {"version": version, "held_out_accuracy": round(acc, 4), "val_images": int(len(y_true)),
         "train_images": len(train), "classes": n_cls, "per_class_accuracy": per_class}, indent=2))
    print(f"wrote model to {out}/crop_classifier.tflite")


if __name__ == "__main__":
    main()
