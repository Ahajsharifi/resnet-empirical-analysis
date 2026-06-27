"""
train.py Main script 

usage : 
python train.py
python train.py --model resnet
python train.py --epochs 10
python train.py --model resnet --epochs 5 --lr 0.0005
"""

import argparse
import json
import os

import torch

from src.datasets.cifar10 import get_dataloaders
from src.models.deep_cnn import DeepCNN
from src.models.resnet import ResNet
from src.models.simple_cnn import SimpleCNN
from src.training.trainer import Trainer

#default configuration

DEFAULT_CONFIG = {
    "epochs" : 30,
    "lr" : 0.001,
    "weight_decay" : 1e-4,
    "patience" : 7,
    "batch_size" :64,
    "checkpoint_dir" : "outputs/checkpoints"
}

MODEL_REGISTRY ={
    "simple_cnn" : SimpleCNN,
    "deep_cnn" : DeepCNN,
    "resnet" : ResNet,
}


def train_one (name, config, train_loader, val_loader, test_loader):

    print(f"\n{'#'*55}")
    print(f"#  شروع آموزش: {name.upper()}")
    print(f"{'#'*55}")

    model = MODEL_REGISTRY[name](num_classes=10)

    cfg = {**config, "model_name": name}
    trainer = Trainer(model, train_loader, val_loader, cfg)

    history = trainer.train()

    trainer.load_best()

    test_loss, test_acc = trainer.evaluate(test_loader)

    history_path =os.path.join("outputs", f"{name}_history.json")
    os.makedirs("outputs", exist_ok=True)
    with open(history_path , "w") as f:
        json.dump(history, f,indent=2)

    return {
        "model": name,
        "best_val_acc": trainer.best_val_acc,
        "test_acc":test_acc,
        "test_loss":test_loss,
        "params":model.count_parameters()
    }

def main(args):
    # ---- DataLoader ----
    print("\nبارگذاری CIFAR-10...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size
    )
    print(f"  Train: {len(train_loader.dataset):,} نمونه")
    print(f"  Val  : {len(val_loader.dataset):,} نمونه")
    print(f"  Test : {len(test_loader.dataset):,} نمونه")
 
    # ---- Config ----
    config = {
        **DEFAULT_CONFIG,
        "epochs":     args.epochs,
        "lr":         args.lr,
        "batch_size": args.batch_size,
    }
 
    # ---- انتخاب مدل‌ها ----
    if args.model == "all":
        models_to_train = list(MODEL_REGISTRY.keys())
    else:
        if args.model not in MODEL_REGISTRY:
            raise ValueError(f"مدل '{args.model}' شناخته نشد. گزینه‌ها: {list(MODEL_REGISTRY.keys())}")
        models_to_train = [args.model]
 
    # ---- آموزش ----
    results = []
    for name in models_to_train:
        result = train_one(name, config, train_loader, val_loader, test_loader)
        results.append(result)
 
    # ---- جدول مقایسه نهایی ----
    print(f"\n{'='*55}")
    print("  نتایج نهایی")
    print(f"{'='*55}")
    print(f"  {'مدل':<15} {'پارامترها':>12} {'Val Acc':>10} {'Test Acc':>10}")
    print(f"  {'-'*50}")
    for r in results:
        print(
            f"  {r['model']:<15} "
            f"{r['params']:>12,} "
            f"{r['best_val_acc']:>9.2f}% "
            f"{r['test_acc']:>9.2f}%"
        )
    print(f"{'='*55}\n")
 
    # ذخیره جدول مقایسه
    summary_path = "outputs/summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  خلاصه نتایج ذخیره شد: {summary_path}")
    print("\n  گام بعدی: python plots.py")
 
 
# ----------------------------------------------------------------
# آرگومان‌های خط فرمان
# ----------------------------------------------------------------
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="آموزش مدل‌های CNN/ResNet روی CIFAR-10"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "simple_cnn", "deep_cnn", "resnet"],
        help="کدام مدل آموزش داده شود (پیش‌فرض: همه)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG["epochs"],
        help=f"تعداد epoch (پیش‌فرض: {DEFAULT_CONFIG['epochs']})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_CONFIG["lr"],
        help=f"Learning rate (پیش‌فرض: {DEFAULT_CONFIG['lr']})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
        help=f"Batch size (پیش‌فرض: {DEFAULT_CONFIG['batch_size']})",
    )
 
    args = parser.parse_args()
    main(args)
 