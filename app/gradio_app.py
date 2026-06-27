"""
gradio_app.py
-------------
رابط کاربری Gradio برای مقایسه SimpleCNN، DeepCNN و ResNet.

کاربر یک تصویر آپلود می‌کند و هر سه مدل نتیجه پیش‌بینی
و confidence خود را نشان می‌دهند.

استفاده:
    python app/gradio_app.py
"""

import os
import sys

import gradio as gr
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from PIL import Image

# اضافه کردن root پروژه به path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.simple_cnn import SimpleCNN
from src.models.deep_cnn import DeepCNN
from src.models.resnet import ResNet

# ----------------------------------------------------------------
# تنظیمات
# ----------------------------------------------------------------

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

CHECKPOINT_DIR = "outputs/checkpoints"

MODELS_CONFIG = {
    "Simple CNN":  {"class": SimpleCNN, "file": "simple_cnn_best.pth"},
    "Deep CNN":    {"class": DeepCNN,   "file": "deep_cnn_best.pth"},
    "ResNet":      {"class": ResNet,    "file": "resnet_best.pth"},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------
# Transform — همان test_transform در cifar.py
# ----------------------------------------------------------------

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((32, 32)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
])

# ----------------------------------------------------------------
# لود مدل‌ها
# ----------------------------------------------------------------

def load_models():
    """هر سه مدل را از checkpoint لود می‌کند."""
    models = {}
    for name, cfg in MODELS_CONFIG.items():
        model = cfg["class"](num_classes=10).to(DEVICE)
        ckpt_path = os.path.join(CHECKPOINT_DIR, cfg["file"])

        if not os.path.exists(ckpt_path):
            print(f"⚠ Checkpoint پیدا نشد: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        models[name] = model
        print(f"✓ {name} لود شد")

    return models


# ----------------------------------------------------------------
# پیش‌بینی
# ----------------------------------------------------------------

def predict(image, models):
    """
    تصویر را با هر سه مدل پیش‌بینی می‌کند.

    Args:
        image  : PIL Image از Gradio
        models : dict مدل‌های لودشده

    Returns:
        نتایج هر مدل به صورت dict برای نمایش
    """
    if image is None:
        return {}, {}, {}

    # پیش‌پردازش
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)  # [1, 3, 32, 32]

    results = {}
    for name, model in models.items():
        with torch.no_grad():
            outputs = model(img_tensor)
            probs   = F.softmax(outputs, dim=1)[0]

        # top-3 پیش‌بینی
        top3_probs, top3_idx = probs.topk(3)
        results[name] = {
            cls: float(prob)
            for cls, prob in zip(
                [CLASSES[i] for i in top3_idx],
                top3_probs,
            )
        }

    return results


# ----------------------------------------------------------------
# رابط Gradio
# ----------------------------------------------------------------

def build_interface(models):

    def inference(image):
        if image is None:
            return "تصویری آپلود نشده.", "", ""

        results = predict(image, models)

        outputs = []
        for model_name, preds in results.items():
            top_class = list(preds.keys())[0]
            top_prob  = list(preds.values())[0]
            outputs.append({
                "label": f"{model_name}: {top_class} ({top_prob*100:.1f}%)",
                "confidences": {k: v for k, v in preds.items()}
            })

        # جدول متنی مقایسه
        table = "| مدل | پیش‌بینی | Confidence |\n|---|---|---|\n"
        for model_name, preds in results.items():
            top_class = list(preds.keys())[0]
            top_prob  = list(preds.values())[0]
            table += f"| {model_name} | {top_class} | {top_prob*100:.1f}% |\n"

        return (
            results.get("Simple CNN", {}),
            results.get("Deep CNN",   {}),
            results.get("ResNet",     {}),
            table,
        )

    with gr.Blocks(title="ResNet vs CNN — CIFAR-10") as demo:
        gr.Markdown("""
        # 🔬 Empirical Study of Residual Learning
        **مقایسه SimpleCNN، DeepCNN و ResNet روی CIFAR-10**

        تصویری آپلود کن و ببین هر سه مدل چه پیش‌بینی‌ای دارند.

        کلاس‌ها: ✈️ plane | 🚗 car | 🐦 bird | 🐱 cat | 🦌 deer
                  🐶 dog | 🐸 frog | 🐴 horse | 🚢 ship | 🚛 truck
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="تصویر ورودی")
                submit_btn  = gr.Button("پیش‌بینی", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    out_simple = gr.Label(num_top_classes=3, label="Simple CNN")
                    out_deep   = gr.Label(num_top_classes=3, label="Deep CNN")
                    out_resnet = gr.Label(num_top_classes=3, label="ResNet")

        comparison_table = gr.Markdown(label="مقایسه نتایج")

        submit_btn.click(
            fn=inference,
            inputs=[image_input],
            outputs=[out_simple, out_deep, out_resnet, comparison_table],
        )

        gr.Examples(
            examples=[],
            inputs=image_input,
        )

        gr.Markdown("""
        ---
        **نتایج آموزش:**
        | مدل | Test Accuracy | پارامترها |
        |---|---|---|
        | Simple CNN | 80.17% | 620K |
        | Deep CNN | 89.71% | 4.7M |
        | ResNet | 90.50% | 11.2M |
        """)

    return demo


# ----------------------------------------------------------------
# اجرا
# ----------------------------------------------------------------

if __name__ == "__main__":
    print("لود مدل‌ها...")
    models = load_models()

    if not models:
        print("❌ هیچ مدلی لود نشد. اول train.py را اجرا کن.")
        sys.exit(1)

    print(f"\n{len(models)} مدل لود شد. در حال راه‌اندازی Gradio...\n")
    demo = build_interface(models)
    demo.launch(share=False)