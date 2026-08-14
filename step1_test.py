"""
STEP 1 TEST SCRIPT
-------------------
Goal: generate ONE fake aged-paper background + render Devanagari text on it.
Run with:  python step1_test.py
Output:    test_output.png  (open it and look!)
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

# ---------- CONFIG ----------
WIDTH, HEIGHT = 1000, 700
MARGIN = 60                     # keep text inside this margin
FONT_PATH = "fonts/NotoSansDevanagari-Regular.ttf"
FONT_SIZE = 34
OUTPUT_PATH = "test_output.png"

SAMPLE_TEXT = (
    "लक्षण। अपूर्व असे परियेसा ।८। ऋषि म्हणे रायासी। पुत्रभविष्य "
    "पुससी। ऐकोनि दुःख पावसी। कवणेपरी सांगावे ।९। राव विनवी तये "
    "वेळी। निरोपावे सकळी। उपाय करिसी तात्काळी। दुःखावेगळा तूचि "
    "करिसी ।१०।"
)


def make_paper_background(width, height):
    base_color = np.array([222, 203, 168], dtype=np.float32)
    img = np.ones((height, width, 3), dtype=np.float32) * base_color

    noise = np.random.normal(0, 8, (height, width, 1))
    img += noise

    stain_layer = np.zeros((height, width), dtype=np.float32)
    num_stains = random.randint(4, 8)
    for _ in range(num_stains):
        cx, cy = random.randint(0, width), random.randint(0, height)
        radius = random.randint(60, 220)
        yy, xx = np.ogrid[:height, :width]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        blob = np.clip(1 - dist / radius, 0, 1) ** 2
        stain_layer += blob * random.uniform(15, 35)

    for c in range(3):
        img[:, :, c] -= stain_layer

    yy, xx = np.ogrid[:height, :width]
    cx, cy = width / 2, height / 2
    dist_from_center = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    vignette = 1 - (dist_from_center / max_dist) * 0.35
    for c in range(3):
        img[:, :, c] *= vignette

    img = np.clip(img, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img, mode="RGB")
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.6))

    return pil_img


def wrap_text_to_width(draw, text, font, max_width):
    words = text.split(" ")
    lines = []
    current_line = ""

    for word in words:
        test_line = (current_line + " " + word).strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]
        if line_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    return lines


def render_text_on_image(img, text, font_path, font_size, margin):
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)

    max_text_width = img.width - (2 * margin)
    lines = wrap_text_to_width(draw, text, font, max_text_width)

    y = margin
    line_spacing = int(font_size * 1.5)

    for line in lines:
        if y + line_spacing > img.height - margin:
            break

        x_offset = margin + random.randint(-3, 3)
        ink_r = random.randint(30, 55)
        ink_g = random.randint(20, 40)
        ink_b = random.randint(15, 30)
        ink_color = (ink_r, ink_g, ink_b)

        draw.text((x_offset, y), line, font=font, fill=ink_color)
        y += line_spacing + random.randint(-2, 4)

    return img


def main():
    print("Generating paper background...")
    bg = make_paper_background(WIDTH, HEIGHT)

    print("Rendering text...")
    final_img = render_text_on_image(bg, SAMPLE_TEXT, FONT_PATH, FONT_SIZE, MARGIN)

    final_img.save(OUTPUT_PATH)
    print(f"Done! Saved to {OUTPUT_PATH} -- open it and take a look.")


if __name__ == "__main__":
    main()