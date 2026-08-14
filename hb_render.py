"""
Proper Indic-script text shaping using HarfBuzz + FreeType, bypassing Pillow's
text layout entirely (which is what was producing wrong/missing conjuncts and
matras without raqm). This shapes the word correctly (reordering, conjunct
formation) then rasterizes each positioned glyph and pastes it onto an RGBA
canvas — same visual result you'd get from Pillow+raqm, but independent of
whether raqm/fribidi is installed on the system.

Usage: shape_and_render_word(word, font_path, font_size, color, alpha) -> PIL Image (RGBA), tightly cropped to the glyphs, plus the pen-origin baseline offset so callers can position it correctly.
"""

import uharfbuzz as hb
import freetype
import numpy as np
from PIL import Image

_face_cache = {}
_hb_font_cache = {}


def _get_freetype_face(font_path, font_size):
    key = (font_path, font_size)
    if key not in _face_cache:
        face = freetype.Face(font_path)
        face.set_char_size(font_size * 64)  # 26.6 fixed point
        _face_cache[key] = face
    return _face_cache[key]


def _get_hb_font(font_path, font_size):
    key = (font_path, font_size)
    if key not in _hb_font_cache:
        with open(font_path, "rb") as f:
            font_data = f.read()
        hb_face = hb.Face(font_data)
        hb_font = hb.Font(hb_face)
        # HarfBuzz uses 26.6-ish upem scaling; match FreeType's 64x scale
        upem = hb_face.upem
        scale = font_size * 64
        hb_font.scale = (scale, scale)
        _hb_font_cache[key] = hb_font
    return _hb_font_cache[key]


def shape_word(word, font_path, font_size):
    """Returns (glyph_infos, glyph_positions) after proper Indic shaping."""
    hb_font = _get_hb_font(font_path, font_size)
    buf = hb.Buffer()
    buf.add_str(word)
    buf.guess_segment_properties()  # auto-detects script/language/direction
    hb.shape(hb_font, buf)
    return buf.glyph_infos, buf.glyph_positions


def shape_and_render_word(word, font_path, font_size, color=(35, 30, 28), alpha=255):
    """
    Shapes `word` correctly and rasterizes it to a tightly-cropped RGBA image.
    Returns (PIL.Image RGBA, ascent_px) where ascent_px lets the caller align
    multiple words on a shared baseline.
    """
    if not word:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0)), 0

    face = _get_freetype_face(font_path, font_size)
    infos, positions = shape_word(word, font_path, font_size)

    # First pass: compute overall bounding box in 26.6 pixel space
    pen_x, pen_y = 0, 0
    glyph_bitmaps = []  # (bitmap, left, top, advance_pen_x, advance_pen_y)
    min_x, min_y, max_x, max_y = 1e9, 1e9, -1e9, -1e9

    for info, pos in zip(infos, positions):
        gid = info.codepoint
        face.load_glyph(gid, flags=freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
        bitmap = face.glyph.bitmap
        left = face.glyph.bitmap_left
        top = face.glyph.bitmap_top

        x_offset = pos.x_offset / 64.0
        y_offset = pos.y_offset / 64.0

        gx = pen_x + x_offset + left
        gy = pen_y - y_offset - top  # FreeType y-up -> image y-down

        if bitmap.width > 0 and bitmap.rows > 0:
            buf = np.array(bitmap.buffer, dtype=np.uint8).reshape(bitmap.rows, bitmap.width)
            glyph_bitmaps.append((buf, gx, gy))
            min_x = min(min_x, gx)
            min_y = min(min_y, gy)
            max_x = max(max_x, gx + bitmap.width)
            max_y = max(max_y, gy + bitmap.rows)

        pen_x += pos.x_advance / 64.0
        pen_y += pos.y_advance / 64.0

    if not glyph_bitmaps:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0)), 0

    canvas_w = int(np.ceil(max_x - min_x)) + 2
    canvas_h = int(np.ceil(max_y - min_y)) + 2
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

    for buf, gx, gy in glyph_bitmaps:
        ox = int(round(gx - min_x))
        oy = int(round(gy - min_y))
        h, w = buf.shape
        # Boost antialiased coverage before using it as ink opacity — thin
        # strokes at small font sizes rarely hit full 255 coverage even at
        # their center, which was making all text look pale/grey. Real ink
        # doesn't feather at stroke edges the way font antialiasing does, so
        # push mid-coverage pixels much closer to fully opaque (gamma < 1).
        coverage = buf.astype(np.float32) / 255.0
        boosted = np.power(coverage, 0.45)  # gamma boost: 0.5 coverage -> ~0.73 opacity
        for c in range(3):
            canvas[oy:oy + h, ox:ox + w, c] = color[c]
        canvas[oy:oy + h, ox:ox + w, 3] = np.maximum(
            canvas[oy:oy + h, ox:ox + w, 3],
            (boosted * 255.0 * (alpha / 255.0)).astype(np.uint8)
        )

    ascent_px = -min_y  # distance from top of canvas down to the baseline
    return Image.fromarray(canvas, mode="RGBA"), ascent_px


if __name__ == "__main__":
    # Quick smoke test — replace with a real font path + a word containing a
    # conjunct (e.g. क्ष, ज्ञ) to visually confirm correct shaping.
    import sys
    font_path = sys.argv[1] if len(sys.argv) > 1 else "fonts/NotoSansDevanagari-Regular.ttf"
    test_word = sys.argv[2] if len(sys.argv) > 2 else "क्षत्रिय"
    img, ascent = shape_and_render_word(test_word, font_path, 48)
    img.save("hb_test_output.png")
    print(f"Rendered '{test_word}' -> hb_test_output.png (size={img.size}, ascent={ascent:.1f})")