import json
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ----------------------------------------------------------------
# تنظیمات ظاهری
# ----------------------------------------------------------------

COLORS = {
    "simple_cnn": "#4C72B0",
    "deep_cnn":   "#DD8452",
    "resnet":     "#55A868",
}

LABELS = {
    "simple_cnn": "Simple CNN",
    "deep_cnn":   "Deep CNN",
    "resnet":     "ResNet",
}

plt.rcParams.update({
    "figure.dpi":      150,
    "font.size":       11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "legend.framealpha": 0.9,
})

RESULTS_DIR  = "outputs"
FIGURES_DIR  = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ----------------------------------------------------------------
# لود داده‌ها
# ----------------------------------------------------------------

def load_histories():
    """فایل‌های history.json را لود می‌کند."""
    histories = {}
    for name in ["simple_cnn", "deep_cnn", "resnet"]:
        path = os.path.join(RESULTS_DIR, f"{name}_history.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"فایل پیدا نشد: {path}\nاول train.py را اجرا کن.")
        with open(path) as f:
            histories[name] = json.load(f)
    return histories


def load_summary():
    """فایل summary.json را لود می‌کند."""
    path = os.path.join(RESULTS_DIR, "summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"فایل پیدا نشد: {path}")
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------
# نمودار ۱ — Loss Curves
# ----------------------------------------------------------------

def plot_loss_curves(histories, save=True):
    """Train و Validation Loss هر سه مدل را رسم می‌کند."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    fig.suptitle("Train vs Validation Loss", fontsize=14, fontweight="bold", y=1.02)

    for ax, (name, h) in zip(axes, histories.items()):
        epochs = range(1, len(h["train_loss"]) + 1)
        color  = COLORS[name]

        ax.plot(epochs, h["train_loss"], color=color,
                linestyle="--", linewidth=1.8, label="Train Loss", alpha=0.8)
        ax.plot(epochs, h["val_loss"],   color=color,
                linestyle="-",  linewidth=2.2, label="Val Loss")

        ax.set_title(LABELS[name], fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

        # نقطه minimum val loss
        min_idx = np.argmin(h["val_loss"])
        ax.axvline(min_idx + 1, color="red", linestyle=":", alpha=0.5, linewidth=1.2)
        ax.annotate(
            f"min={h['val_loss'][min_idx]:.3f}",
            xy=(min_idx + 1, h["val_loss"][min_idx]),
            xytext=(min_idx + 2, h["val_loss"][min_idx] + 0.05),
            fontsize=8, color="red",
        )

    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, "loss_curves.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"  ذخیره شد: {path}")
    plt.show()
    plt.close()


# ----------------------------------------------------------------
# نمودار ۲ — Accuracy Curves
# ----------------------------------------------------------------

def plot_accuracy_curves(histories, save=True):
    """Train و Validation Accuracy هر سه مدل را رسم می‌کند."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Validation Accuracy — مقایسه سه مدل",
                 fontsize=13, fontweight="bold")

    for name, h in histories.items():
        epochs = range(1, len(h["val_acc"]) + 1)
        color  = COLORS[name]

        ax.plot(epochs, h["train_acc"], color=color,
                linestyle="--", linewidth=1.5, alpha=0.5)
        ax.plot(epochs, h["val_acc"],   color=color,
                linestyle="-",  linewidth=2.2, label=LABELS[name])

        # بهترین val acc
        best = max(h["val_acc"])
        best_ep = h["val_acc"].index(best) + 1
        ax.scatter(best_ep, best, color=color, s=60, zorder=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(title="مدل")

    # خط راهنما
    ax.axhline(90, color="gray", linestyle=":", alpha=0.4, linewidth=1)
    ax.text(1, 90.3, "90%", fontsize=8, color="gray")

    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, "accuracy_curves.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"  ذخیره شد: {path}")
    plt.show()
    plt.close()


# ----------------------------------------------------------------
# نمودار ۳ — مقایسه نهایی
# ----------------------------------------------------------------

def plot_final_comparison(summary, save=True):
    """Bar chart مقایسه Test Accuracy و تعداد پارامترها."""
    names      = [LABELS[r["model"]] for r in summary]
    test_accs  = [r["test_acc"]      for r in summary]
    params     = [r["params"] / 1e6  for r in summary]   # به میلیون
    colors     = [COLORS[r["model"]] for r in summary]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("مقایسه نهایی مدل‌ها", fontsize=14, fontweight="bold")

    # ---- Test Accuracy ----
    bars = ax1.bar(names, test_accs, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
    ax1.set_title("Test Accuracy")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(70, 95)
    for bar, acc in zip(bars, test_accs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{acc:.2f}%",
            ha="center", va="bottom", fontweight="bold", fontsize=10,
        )

    # ---- تعداد پارامترها ----
    bars2 = ax2.bar(names, params, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
    ax2.set_title("تعداد پارامترها (میلیون)")
    ax2.set_ylabel("Parameters (M)")
    for bar, p in zip(bars2, params):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{p:.1f}M",
            ha="center", va="bottom", fontweight="bold", fontsize=10,
        )

    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, "final_comparison.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"  ذخیره شد: {path}")
    plt.show()
    plt.close()


# ----------------------------------------------------------------
# نمودار ۴ — Overview (همه در یک تصویر)
# ----------------------------------------------------------------

def plot_overview(histories, summary, save=True):
    """همه نمودارها را در یک تصویر ترکیب می‌کند — برای README."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Empirical Study of Residual Learning — CIFAR-10",
        fontsize=15, fontweight="bold", y=1.01,
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ---- ردیف اول: Val Accuracy هر سه مدل ----
    for col, (name, h) in enumerate(histories.items()):
        ax = fig.add_subplot(gs[0, col])
        epochs = range(1, len(h["val_acc"]) + 1)
        color  = COLORS[name]
        ax.plot(epochs, h["train_loss"], color=color, linestyle="--",
                linewidth=1.5, alpha=0.6, label="Train")
        ax.plot(epochs, h["val_loss"],   color=color, linestyle="-",
                linewidth=2,   label="Val")
        ax.set_title(f"{LABELS[name]} — Loss", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)

    # ---- ردیف دوم: مقایسه Accuracy + Bar chart ----
    ax_acc = fig.add_subplot(gs[1, :2])
    for name, h in histories.items():
        epochs = range(1, len(h["val_acc"]) + 1)
        ax_acc.plot(epochs, h["val_acc"], color=COLORS[name],
                    linewidth=2.2, label=LABELS[name])
    ax_acc.set_title("Validation Accuracy — مقایسه", fontweight="bold")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.legend()
    ax_acc.axhline(90, color="gray", linestyle=":", alpha=0.4)

    ax_bar = fig.add_subplot(gs[1, 2])
    names     = [LABELS[r["model"]] for r in summary]
    test_accs = [r["test_acc"]      for r in summary]
    colors    = [COLORS[r["model"]] for r in summary]
    bars = ax_bar.bar(names, test_accs, color=colors, width=0.5,
                      edgecolor="white", linewidth=1.5)
    ax_bar.set_title("Test Accuracy", fontweight="bold")
    ax_bar.set_ylabel("Accuracy (%)")
    ax_bar.set_ylim(70, 95)
    for bar, acc in zip(bars, test_accs):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{acc:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, "overview.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"  ذخیره شد: {path}")
    plt.show()
    plt.close()


# ----------------------------------------------------------------
# اجرای اصلی
# ----------------------------------------------------------------

if __name__ == "__main__":
    print("لود داده‌ها...")
    histories = load_histories()
    summary   = load_summary()

    print("\nرسم نمودارها...")
    plot_loss_curves(histories)
    plot_accuracy_curves(histories)
    plot_final_comparison(summary)
    plot_overview(histories, summary)

    print(f"\n✓ همه نمودارها در {FIGURES_DIR}/ ذخیره شدند.")
    print("  گام بعدی: python app/gradio_app.py")