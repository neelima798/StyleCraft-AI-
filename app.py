from flask import Flask, render_template, request
from huggingface_hub import InferenceClient
import uuid

app = Flask(__name__)

HF_API_KEY = "YOUR_HF_TOKEN"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_API_KEY
)

STYLE_PRESETS = {
    "Traditional": "traditional ethnic wear with rich embroidery",
    "Modern": "modern fashion with clean cuts and minimal design",
    "Indo-Western": "indo-western fusion style",
    "Minimalist": "minimalist elegant fashion",
    "Vintage": "vintage classic fashion style"
}

@app.route("/", methods=["GET", "POST"])
def index():
    image_path = None

    if request.method == "POST":
        style_text = request.form.get("style")
        color = request.form.get("color")
        fabric = request.form.get("fabric")
        occasion = request.form.get("occasion")
        preset = request.form.get("preset")
        target = request.form.get("target")

        # 👤 Target logic
        if target == "Women":
            model_subject = "adult female fashion model"
        elif target == "Men":
            model_subject = "adult male fashion model"
        else:
            model_subject = "child fashion model"

        preset_description = STYLE_PRESETS.get(preset, "")

        # 🔥 AUTO-BUILT PROMPT
        final_prompt = f"""
        Single full-length {model_subject} wearing an elegant {color} {fabric} {style_text},
        {preset_description}, suitable for {occasion},
        ultra realistic, sharp focus, studio lighting,
        high detail fabric texture, professional fashion photography,
        clean background, one person only
        """

        negative_prompt = """
        blurry, low quality, duplicate person, multiple bodies,
        extra arms, extra legs, ghosting, double exposure,
        deformed face, distorted anatomy, cropped, out of frame
        """

        image = client.text_to_image(
            prompt=final_prompt,
            model=MODEL_ID,
            negative_prompt=negative_prompt,
            guidance_scale=8.5,
            num_inference_steps=35
        )

        filename = f"{uuid.uuid4().hex}.png"
        image_path = f"static/{filename}"
        image.save(image_path)

    return render_template("index.html", image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
