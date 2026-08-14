import os
import random
import warnings
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageFilter

from hb_render import shape_and_render_word

warnings.filterwarnings("ignore", category=UserWarning, module="PIL.ImageFont")

# ---------- CONFIGURATION ----------
WIDTH, HEIGHT = 1000, 480
MARGIN_LEFT = 90
MARGIN_RIGHT = 190          # wide right margin reserved for side annotations
BODY_RIGHT_EDGE = WIDTH - MARGIN_RIGHT

SCRIPTS = {
    "devanagari": {
        "font": "fonts/Kalam-Regular.ttf",
        "corpus": "data/devanagari_md.md",
        "font_size": 26,
    },
    "modi": {
        # MarathiCursiveT — genuine cursive Modi font (M+ Fonts license),
        # replaces the print font. Uses real Modi Unicode codepoints.
        "font": "fonts/MarathiCursiveT.ttf",
        "corpus": "data/Modi_md.md",
        "font_size": 24,
    },
    "sharada": {
        "font": "fonts/NotoSansSharada-Regular.ttf",
        "corpus": "data/sharada_md.md",
        "font_size": 24,
    },
}

BLACK_INK = (35, 30, 28)
RED_INK = (178, 45, 38)


def apply_deckled_edges(img, edge_color=(90, 70, 45)):
    """
    Adds an irregular, torn/deckled page boundary. Must run LAST in the
    background pipeline, after apply_page_deformation — deformation remaps
    pixels via cv2.remap, and if the torn edge already exists when that runs,
    the remap smears/stretches that texture into a blurry band. Cutting the
    edge after deformation keeps it crisp.
    """
    w, h = img.size
    arr = np.array(img).astype(np.float32)

    noise = np.random.RandomState(random.randint(0, 99999)).rand(h, w).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=6)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)

    depth = random.randint(4, 10)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.minimum.reduce([xx, w - 1 - xx, yy, h - 1 - yy])
    ragged = dist - (noise * depth * 2 - depth)
    torn_mask = np.clip(ragged / 3.0, 0, 1)

    for c in range(3):
        arr[..., c] = arr[..., c] * torn_mask + edge_color[c] * (1 - torn_mask) * 0.3
    worn = 210 * (1 - torn_mask) * 0.6
    arr += worn[..., None] * 0.4

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_aging_overlay(img, aging_intensity=1.0):
    """
    Universal aging pass — applied to EVERY background material (paper,
    palmleaf, vellum) at the same point in the pipeline, so aging_intensity
    scales consistently regardless of which material got picked. Previously
    aging lived inside individual material functions and the scanned-paper
    path had none at all, so a large share of images (paper is the most
    common pick) looked equally aged no matter the intensity value.
    """
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]

    # warm yellow/brown tint, deepens with intensity
    tint_strength = np.clip(0.05 * aging_intensity, 0, 0.4)
    aging_tint = np.array([14, 4, -18])
    arr += aging_tint * tint_strength * 10

    # blotchy stains
    stain_layer = np.zeros((h, w), dtype=np.uint8)
    n_stains = random.randint(2, max(2, int(6 * aging_intensity)))
    for _ in range(n_stains):
        cx, cy = random.randint(0, w), random.randint(0, h)
        r = random.randint(30, 90)
        cv2.circle(stain_layer, (cx, cy), r, 255, -1)
    stain_layer = cv2.GaussianBlur(stain_layer, (0, 0), sigmaX=22)
    stain_strength = (stain_layer.astype(np.float32) / 255.0) * random.uniform(12, 26) * aging_intensity
    arr -= stain_strength[..., None]

    # foxing speckles
    n_specks = int(random.randint(15, 50) * aging_intensity)
    speck_layer = np.zeros((h, w), dtype=np.uint8)
    for _ in range(n_specks):
        cx, cy = random.randint(0, w), random.randint(0, h)
        r = random.randint(1, 3)
        cv2.circle(speck_layer, (cx, cy), r, 255, -1)
    speck_layer = cv2.GaussianBlur(speck_layer, (0, 0), sigmaX=0.8)
    speck_strength = (speck_layer.astype(np.float32) / 255.0) * random.uniform(18, 36)
    arr -= speck_strength[..., None]

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_margin_rule_lines(img):
    """Adds the two faint vertical red rule lines marking the writing margin."""
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    rule_color = np.array([150, 60, 55])
    for frac in [MARGIN_LEFT / w - 0.012, BODY_RIGHT_EDGE / w + 0.012]:
        x = int(frac * w)
        if 0 <= x < w:
            jitter = np.random.normal(0, 0.6, h).cumsum()
            jitter -= jitter.mean()
            for y in range(h):
                xx = int(np.clip(x + jitter[y] * 0.3, 1, w - 2))
                alpha = random.uniform(0.35, 0.55)
                arr[y, xx - 1:xx + 1] = arr[y, xx - 1:xx + 1] * (1 - alpha) + rule_color * alpha
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ---------- 1. BACKGROUND ----------
def generate_vellum_background(width, height, aging_intensity=1.0):
    """
    Material-specific texture only (base tone + weave). Aging (tint/stains/
    speckles) and the deckled edge are now applied centrally in
    generate_dataset, after deformation — see apply_aging_overlay and
    apply_deckled_edges.
    """
    base_tone = np.array(random.choice([
        [198, 184, 152], [206, 194, 168], [188, 172, 140], [212, 200, 178],
    ]), dtype=np.float32)

    arr = np.tile(base_tone, (height, width, 1))

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    tone_drift = 5 * np.sin(yy / random.uniform(80, 140) + random.uniform(0, 6))
    tone_drift += 4 * np.sin(xx / random.uniform(180, 320) + random.uniform(0, 6))
    arr += tone_drift[..., None]

    weave = np.zeros((height, width), dtype=np.float32)
    step = random.randint(3, 5)
    weave[::step, :] -= random.uniform(2, 5)
    weave[:, ::step] -= random.uniform(2, 5)
    weave = cv2.GaussianBlur(weave, (0, 0), sigmaX=0.6)
    arr += weave[..., None]

    arr += np.random.normal(0, 2.5, arr.shape)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_palmleaf_background(width, height, aging_intensity=1.0):
    """
    Material-specific texture only (base tone + fiber grain + binding hole).
    Aging and deckled edge are applied centrally — see apply_aging_overlay
    and apply_deckled_edges.
    """
    base_tone = np.array(random.choice([
        [150, 122, 70], [168, 138, 82], [140, 108, 60], [176, 148, 96],
    ]), dtype=np.float32)

    arr = np.tile(base_tone, (height, width, 1))

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    tone_drift = 8 * np.sin(yy / random.uniform(60, 120) + random.uniform(0, 6))
    tone_drift += 6 * np.sin(xx / random.uniform(150, 300) + random.uniform(0, 6))
    arr += tone_drift[..., None]

    grain_layer = np.zeros((height, width), dtype=np.float32)
    num_fibers = random.randint(35, 55)
    for _ in range(num_fibers):
        y_center = random.uniform(0, height)
        wobble_amp = random.uniform(1.5, 5)
        wobble_freq = random.uniform(0.01, 0.03)
        phase = random.uniform(0, 6.28)
        darkness = random.uniform(4, 16)
        thickness = random.choice([1, 1, 1, 2])
        n_segments = random.randint(2, 4)
        seg_bounds = sorted(random.sample(range(0, width), min(width, n_segments * 2)))
        for si in range(0, len(seg_bounds) - 1, 2):
            x0, x1 = seg_bounds[si], seg_bounds[si + 1]
            xs = np.arange(x0, x1)
            if len(xs) == 0:
                continue
            ys = (y_center + wobble_amp * np.sin(xs * wobble_freq + phase)).astype(int)
            ys = np.clip(ys, 0, height - 1)
            local_darkness = darkness * random.uniform(0.6, 1.0)
            for dt in range(thickness):
                yv = np.clip(ys + dt, 0, height - 1)
                grain_layer[yv, xs] -= local_darkness
    arr += grain_layer[..., None]

    arr += np.random.normal(0, 3.0, arr.shape)

    if random.random() < 0.5:
        hx = random.choice([
            random.randint(15, max(16, MARGIN_LEFT - 25)),
            random.randint(min(width - 40, BODY_RIGHT_EDGE + 40), width - 15),
        ])
        hy = height // 2 + random.randint(-10, 10)
        cv2.circle(arr, (hx, hy), 6, (90, 65, 35), -1)
        cv2.circle(arr, (hx, hy), 10, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def get_master_background(width, height, material=None):
    """
    Returns raw material texture only — no aging, no deckled edge. Both are
    applied centrally in generate_dataset (apply_aging_overlay, then
    apply_deckled_edges after deformation) so every material responds
    consistently to aging_intensity.
    """
    if material is None:
        material = random.choices(["paper", "palmleaf", "vellum"], weights=[0.45, 0.30, 0.25])[0]

    if material == "palmleaf":
        return generate_palmleaf_background(width, height)

    if material == "vellum":
        return generate_vellum_background(width, height)

    bg_cache_path = "data/bg_paper.png"
    sample_path = "data/sample_image.png"
    if not os.path.exists(sample_path):
        sample_path = "data/sample_image.jpg"

    if os.path.exists(bg_cache_path):
        bg_img = Image.open(bg_cache_path).convert("RGB")
        return bg_img.resize((width, height), Image.Resampling.LANCZOS)

    if os.path.exists(sample_path):
        sample_img = cv2.imread(sample_path)
        hsv = cv2.cvtColor(sample_img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)

        _, dark_mask = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY_INV)
        lower_red1, upper_red1 = np.array([0, 40, 40]), np.array([18, 255, 255])
        lower_red2, upper_red2 = np.array([160, 40, 40]), np.array([180, 255, 255])
        red_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))

        full_mask = cv2.bitwise_or(dark_mask, red_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_mask = cv2.dilate(full_mask, kernel, iterations=1)

        clean_paper = cv2.inpaint(sample_img, dilated_mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)

        ring = cv2.dilate(dilated_mask, kernel, iterations=3)
        blurred = cv2.GaussianBlur(clean_paper, (0, 0), sigmaX=1.5)
        ring_f = (cv2.GaussianBlur(ring, (0, 0), sigmaX=4).astype(np.float32) / 255.0)[..., None]
        clean_paper = (clean_paper.astype(np.float32) * (1 - ring_f) + blurred.astype(np.float32) * ring_f).astype(np.uint8)

        cv2.imwrite(bg_cache_path, clean_paper)
        result = Image.fromarray(cv2.cvtColor(clean_paper, cv2.COLOR_BGR2RGB)).resize((width, height), Image.Resampling.LANCZOS)
        return result

    base = np.full((height, width, 3), [228, 208, 165], dtype=np.uint8)
    return Image.fromarray(base)


def randomize_background(bg):
    src = np.array(bg)
    h, w = src.shape[:2]

    scale = random.uniform(1.0, 1.12)
    ch, cw = int(h / scale), int(w / scale)
    ch, cw = max(20, ch), max(20, cw)
    top = random.randint(0, max(0, h - ch))
    left = random.randint(0, max(0, w - cw))
    crop = src[top:top + ch, left:left + cw]
    crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
    if random.random() < 0.5:
        crop = cv2.flip(crop, 1)
    if random.random() < 0.3:
        crop = cv2.flip(crop, 0)

    arr = crop.astype(np.float32)

    tone = np.array([
        random.uniform(-14, 14),
        random.uniform(-10, 10),
        random.uniform(-16, 16),
    ])
    arr += tone
    arr *= random.uniform(0.95, 1.05)

    grain = np.random.normal(0, random.uniform(4.0, 7.5), arr.shape)
    arr += grain

    stain_layer = np.zeros(arr.shape[:2], dtype=np.uint8)
    for _ in range(random.randint(0, 3)):
        cx, cy = random.randint(0, w), random.randint(0, h)
        r = random.randint(20, 55)
        cv2.circle(stain_layer, (cx, cy), r, 255, -1)
    stain_layer = cv2.GaussianBlur(stain_layer, (0, 0), sigmaX=random.uniform(12, 20))
    stain_strength = (stain_layer.astype(np.float32) / 255.0) * random.uniform(5, 12)
    arr -= stain_strength[..., None]

    blur = cv2.GaussianBlur(arr, (0, 0), sigmaX=1.2)
    arr = cv2.addWeighted(arr, 1.35, blur, -0.35, 0)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------- 2. PAGE DEFORMATION (curl / fold / warp) ----------
def apply_page_deformation(img):
    """
    Simulates page curling, folds and surface warping via a displacement field.
    Must run BEFORE text is composited AND before apply_deckled_edges (see
    apply_edge_shading for the safe post-text step: shading only, no remap).
    """
    arr = np.array(img)
    h, w = arr.shape[:2]

    map_x, map_y = np.meshgrid(np.arange(w).astype(np.float32), np.arange(h).astype(np.float32))

    curl_amp = random.uniform(5, 10)
    curl_freq = random.uniform(0.8, 1.6)
    map_y += curl_amp * np.sin((map_x / w) * math.pi * curl_freq)

    curl_amp2 = random.uniform(2, 5)
    map_x += curl_amp2 * np.sin((map_y / h) * math.pi * random.uniform(0.8, 1.5))

    cx, cy = w / 2.0, h / 2.0
    norm_x = (map_x - cx) / cx
    norm_y = (map_y - cy) / cy
    corner_strength = np.clip((norm_x ** 2 + norm_y ** 2), 0, 1.0)
    corner_amp = random.uniform(2, 4)
    map_y += corner_amp * corner_strength * np.sign(norm_y + 1e-6)
    map_x += corner_amp * 0.4 * corner_strength * np.sign(norm_x + 1e-6)

    n_folds = random.randint(2, 3)
    for _ in range(n_folds):
        fold_x = random.uniform(0.15, 0.85) * w
        fold_width = random.uniform(60, 130)
        fold_amp = random.uniform(2, 5)
        map_x += fold_amp * np.exp(-((map_x - fold_x) ** 2) / (2 * fold_width ** 2)) * np.sign(map_y - h / 2)

    if random.random() < 0.4:
        fold_y = random.uniform(0.2, 0.8) * h
        fold_height = random.uniform(60, 110)
        fold_amp_y = random.uniform(1.5, 3.5)
        map_y += fold_amp_y * np.exp(-((map_y - fold_y) ** 2) / (2 * fold_height ** 2)) * np.sign(map_x - w / 2)

    warp_amp = random.uniform(0.5, 1.0)
    map_x += warp_amp * np.sin(map_y / 18.0 + random.uniform(0, 6))
    map_y += warp_amp * np.cos(map_x / 22.0 + random.uniform(0, 6))

    warped = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped)


def apply_edge_shading(img):
    """Safe to run after text compositing: darkens edges without remapping any
    pixels, so glyphs stay crisp and match ground truth."""
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    edge_dist = np.minimum.reduce([xx, w - 1 - xx, yy, h - 1 - yy]).astype(np.float32)
    edge_dist = edge_dist / max(1, edge_dist.max())
    shade = 0.88 + 0.12 * np.clip(edge_dist * 3, 0, 1)
    arr = (arr * shade[..., None]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------- 3. HANDWRITING RENDERER (per-glyph jitter) ----------
def draw_wavy_line(draw, words, font, start_x, start_y, max_width, font_path, font_size, ink_variation=True):
    space_bbox = draw.textbbox((0, 0), " ", font=font)
    space_w = space_bbox[2] - space_bbox[0]

    shaped_words = []
    for word in words:
        is_marker = any(m in word for m in ["।", "॥", "𑇆"]) or any(ch.isdigit() for ch in word)
        highlight = (not is_marker) and random.random() < 0.10
        color = RED_INK if (is_marker or highlight) else BLACK_INK

        alpha = 255
        darkness_jitter = 1.0
        if ink_variation and random.random() < 0.02:
            alpha = random.randint(215, 240)
            darkness_jitter = random.uniform(0.93, 0.99)
        render_color = tuple(int(c * darkness_jitter) for c in color)

        shaped_img, ascent_px = shape_and_render_word(word, font_path, font_size, color=render_color, alpha=alpha)
        w, h = shaped_img.size
        if w <= 1:
            continue
        shaped_words.append([word, shaped_img, ascent_px, w, h])

    word_widths = [sw[3] for sw in shaped_words]
    total_w = sum(word_widths) + space_w * max(0, len(shaped_words) - 1)

    extra_gap = 0
    if len(shaped_words) > 1 and total_w < max_width:
        extra_gap = (max_width - total_w) / (len(shaped_words) - 1)
        extra_gap = min(extra_gap, space_w * 2.2)

    baseline_phase = random.uniform(0, 6.28)
    baseline_freq = random.uniform(0.006, 0.012)
    baseline_amp = random.uniform(1.5, 3.0)
    global_slant = random.uniform(-0.6, 0.6)

    cursor_x = start_x
    gt_words = []
    pad = 10

    for word, shaped_word_img, word_ascent, ww_shaped, wh_shaped in shaped_words:
        rot = random.uniform(-0.6, 0.6)

        available_w = max_width - (cursor_x - start_x)
        if ww_shaped > available_w > 20:
            scale = available_w / ww_shaped
            new_w = max(1, int(ww_shaped * scale))
            new_h = max(1, int(wh_shaped * scale))
            shaped_word_img = shaped_word_img.resize((new_w, new_h), Image.LANCZOS)
            word_ascent = word_ascent * scale
            ww_shaped, wh_shaped = new_w, new_h

        wc = Image.new("RGBA", (ww_shaped + pad * 2, wh_shaped + pad * 2), (0, 0, 0, 0))
        wc.paste(shaped_word_img, (pad, pad), shaped_word_img)
        if abs(rot) > 0.05:
            wc = wc.rotate(rot, resample=Image.BICUBIC, center=(pad, pad + int(word_ascent)), expand=False)

        wave_y = baseline_amp * math.sin(cursor_x * baseline_freq + baseline_phase)
        micro_jitter = random.uniform(-0.5, 0.5)
        slant_y = (cursor_x - start_x) * math.tan(math.radians(global_slant))
        y = start_y + wave_y + slant_y + micro_jitter - word_ascent

        draw._image.paste(wc, (int(cursor_x - pad), int(y - pad)), wc)

        cursor_x += ww_shaped + space_w + extra_gap
        gt_words.append(word)

    return " ".join(gt_words)


def add_ink_bleed_and_smudges(folio_img, text_mask_img):
    arr = np.array(folio_img).astype(np.float32)
    mask = np.array(text_mask_img.convert("L"))

    bleed = cv2.GaussianBlur(mask, (0, 0), sigmaX=0.6)
    bleed_norm = (bleed.astype(np.float32) / 255.0) * 0.12
    for c in range(3):
        arr[..., c] -= bleed_norm * (255 - BLACK_INK[c]) * 0.25

    smudge_layer = np.zeros(arr.shape[:2], dtype=np.uint8)
    n_smudges = random.choices([0, 1, 2], weights=[0.55, 0.35, 0.10])[0]
    for _ in range(n_smudges):
        cx = random.randint(MARGIN_LEFT, BODY_RIGHT_EDGE)
        cy = random.randint(40, HEIGHT - 40)
        axes = (random.randint(6, 14), random.randint(2, 5))
        angle = random.uniform(0, 180)
        cv2.ellipse(smudge_layer, (cx, cy), axes, angle, 0, 360, 255, -1)
    smudge_layer = cv2.GaussianBlur(smudge_layer, (0, 0), sigmaX=4)
    smudge_strength = (smudge_layer.astype(np.float32) / 255.0) * random.uniform(8, 16)
    arr -= smudge_strength[..., None] * 0.5

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def render_manuscript_text(bg, text_lines, font_path, font_size=24):
    width, height = bg.size
    text_canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_canvas)
    draw._image = text_canvas

    try:
        font = ImageFont.truetype(font_path, font_size)
        margin_font = ImageFont.truetype(font_path, max(12, int(font_size * 0.6)))
    except Exception:
        font = ImageFont.load_default()
        margin_font = font

    full_text = " ".join(text_lines).replace("\xa0", " ").replace("\u200b", "").strip()
    raw_words = full_text.split()

    max_text_width = BODY_RIGHT_EDGE - MARGIN_LEFT - 20
    line_spacing = int(font_size * 1.65)

    _shaped_width_cache = {}

    def shaped_word_width(w):
        if w not in _shaped_width_cache:
            img, _ = shape_and_render_word(w, font_path, font_size)
            _shaped_width_cache[w] = img.size[0]
        return _shaped_width_cache[w]

    space_w_est = draw.textbbox((0, 0), " ", font=font)[2]

    formatted_lines, current_line, current_w = [], [], 0
    for word in raw_words:
        ww = shaped_word_width(word)
        projected_w = current_w + (space_w_est if current_line else 0) + ww
        if projected_w > max_text_width and current_line:
            formatted_lines.append(current_line)
            current_line, current_w = [word], ww
        else:
            current_line.append(word)
            current_w = projected_w
    if current_line:
        formatted_lines.append(current_line)

    lines_to_draw = formatted_lines[:6]
    top_margin, bottom_margin = 55, 55
    usable_height = height - top_margin - bottom_margin
    block_height = max(0, (len(lines_to_draw) - 1) * line_spacing + font_size)
    y = top_margin + max(0, (usable_height - block_height) // 2)

    ground_truth_lines = []
    for word_list in lines_to_draw:
        if y + line_spacing > height - bottom_margin:
            break
        line_gt = draw_wavy_line(draw, word_list, font, MARGIN_LEFT, y, max_text_width, font_path, font_size)
        ground_truth_lines.append(line_gt)
        y += line_spacing

    margin_note = ""
    if random.random() < 0.7 and ground_truth_lines:
        source_words = " ".join(ground_truth_lines).split()
        note_line_count = random.choice([1, 1, 2])
        note_lines = []
        cursor = random.randint(0, max(0, len(source_words) - 4))
        for _ in range(note_line_count):
            picked = source_words[cursor:cursor + random.randint(1, 2)]
            cursor += len(picked)
            if picked:
                note_lines.append(" ".join(picked))
        margin_note = " / ".join(note_lines)

        mx = min(BODY_RIGHT_EDGE + 30, width - 130)
        my_start = random.randint(50, max(60, height - 140))
        note_canvas = Image.new("RGBA", (150, 90), (0, 0, 0, 0))
        ndraw = ImageDraw.Draw(note_canvas)
        ny = 4
        for nl in note_lines:
            ndraw.text((3, ny), nl, font=margin_font, fill=(*RED_INK, 210))
            ny += int(margin_font.size * 1.5)
        ndraw.line([(0, ny + 2), (min(140, ndraw.textlength(note_lines[0], font=margin_font) + 6), ny + 2)],
                    fill=(*RED_INK, 140), width=1)
        note_canvas = note_canvas.rotate(random.uniform(-3, 3), resample=Image.BICUBIC, expand=False)
        text_canvas.paste(note_canvas, (mx, my_start), note_canvas)

    folio = bg.convert("RGB")
    folio.paste(text_canvas, (0, 0), text_canvas)

    folio = add_ink_bleed_and_smudges(folio, text_canvas)

    gt = "\n".join(ground_truth_lines)
    if margin_note:
        gt += f"\n[margin: {margin_note}]"
    return folio, gt


# ---------- 4. DATASET GENERATION ----------
def load_sample_text(corpus_path, num_lines=6):
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    words = text.split()
    if len(words) < 20:
        return [" ".join(words)]
    chunk = random.randint(35, 55)
    start = random.randint(0, max(0, len(words) - chunk))
    return [" ".join(words[start:start + chunk])]


def generate_dataset():
    base_output_dir = "dataset_output"

    for script_name, cfg in SCRIPTS.items():
        if not os.path.exists(cfg["corpus"]):
            print(f"⚠️  Skipping {script_name}: corpus not found at {cfg['corpus']}")
            continue
        print(f"📦 Generating 100 manuscript samples for: {script_name}...")
        splits = {"train": 85, "validation": 10, "test": 5}

        for split_name, count in splits.items():
            split_dir = os.path.join(base_output_dir, script_name, split_name)
            os.makedirs(split_dir, exist_ok=True)

            for i in range(1, count + 1):
                sample_lines = load_sample_text(cfg["corpus"], num_lines=6)
                aging_intensity = random.uniform(1.2, 3.2)

                # --- background pipeline order matters ---
                # 1. raw material texture (no aging, no edge)
                # 2. universal aging overlay (tint/stains/speckles) — applies
                #    the SAME way regardless of which material was picked
                # 3. randomize (crop/flip/tone jitter)
                # 4. margin rule lines
                # 5. page deformation (curl/fold/warp) — remaps pixels
                # 6. deckled edge — LAST, so deformation can't smear/blur it
                bg = get_master_background(WIDTH, HEIGHT)
                bg = apply_aging_overlay(bg, aging_intensity=aging_intensity)
                bg = randomize_background(bg)
                bg = add_margin_rule_lines(bg)
                bg = apply_page_deformation(bg)
                bg = apply_deckled_edges(bg, edge_color=(90, 65, 40))

                final_img, actual_gt_text = render_manuscript_text(
                    bg, sample_lines, cfg["font"], cfg["font_size"]
                )
                final_img = apply_edge_shading(final_img)

                img_path = os.path.join(split_dir, f"Image_{i}.png")
                md_path = os.path.join(split_dir, f"Image_{i}.md")

                final_img.save(img_path)
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(actual_gt_text)

        print(f"✅ Finished {script_name}!")


def main():
    generate_dataset()
    print("\n🚀 All manuscripts generated.")


if __name__ == "__main__":
    main()