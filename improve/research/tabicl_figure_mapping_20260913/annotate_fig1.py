"""Mark the current TabICLv2 replacement scope on the published Figure 1."""
from pathlib import Path
import hashlib
import json

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
RED = '#C43C35'
INK = '#172B3A'
FONT_ROOT = Path('/usr/share/fonts/opentype/noto')


def font(size, bold=False):
    name = 'NotoSansCJK-Bold.ttc' if bold else 'NotoSansCJK-Regular.ttc'
    return ImageFont.truetype(str(FONT_ROOT / name), size, index=2)


def render(original, crop_top, name):
    header, footer = 118, 210
    panel = original.crop((0, crop_top, original.width, original.height))
    canvas = Image.new('RGB', (panel.width, panel.height + header + footer), 'white')
    canvas.paste(panel, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 12), 'TabICLv2 替换范围：图 1(b) 的图编码器 + 边解码器',
              fill=INK, font=font(34, True))
    draw.text((30, 67), '① 图编码器（右侧为同一模块的展开）；② 边解码器（concat + MLP）',
              fill=RED, font=font(25))
    offset = header - crop_top

    # Coordinates refer to the unchanged 1600 x 1606 source image.
    for box in [(57, 1207, 586, 1285), (57, 1317, 478, 1603),
                (630, 763, 1596, 1603)]:
        x0, y0, x1, y1 = box
        draw.rectangle((x0, y0 + offset, x1, y1 + offset), outline=RED, width=5)

    for x, y, label in [(30, 1244, '1'), (30, 1434, '2'), (657, 786, '1')]:
        y += offset
        draw.ellipse((x - 21, y - 21, x + 21, y + 21), fill=RED)
        draw.text((x, y - 2), label, font=font(26, True), fill='white', anchor='mm')

    bottom = header + panel.height
    draw.line((25, bottom + 10, panel.width - 25, bottom + 10), fill='#D8E0E5', width=2)
    draw.text((30, bottom + 24),
              '原图 / 节点数据 → 36 维图统计   ＋   RNA–基因 / 药物–基因 / 通路 → 26 维',
              fill=INK, font=font(25))
    draw.text((30, bottom + 70), '62 维配对特征 → TabICLv2（权重冻结）→ 关联分数',
              fill=INK, font=font(30, True))
    draw.text((30, bottom + 122),
              '红框和中文说明为本次标注；原图 (a) 的数据与图关系继续作为输入特征的来源。',
              fill='#4B5964', font=font(21))
    draw.text((30, bottom + 162),
              '原图：Wei et al., Bioinformatics (2026), Fig. 1 · doi:10.1093/bioinformatics/btag029 · CC BY 4.0',
              fill='#4B5964', font=font(20))
    target = ROOT / name
    canvas.save(target)
    return target


if __name__ == '__main__':
    original = Image.open(ROOT / 'fig1_original.webp').convert('RGB')
    if original.size != (1600, 1606):
        raise ValueError('Unexpected source size; annotation coordinates must be reviewed')
    original.save(ROOT / 'fig1_original.png')
    paths = [render(original, 760, 'fig1b_replacement_annotated.png'),
             render(original, 0, 'fig1_replacement_annotated.png')]
    manifest_path = ROOT / 'SOURCE.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['annotation_note'] = (
        'Red boxes and Chinese explanatory text were added to the original figure. '
        'The current pipeline uses fixed graph/evidence pair features and frozen TabICLv2 '
        'instead of the published GeneralConv encoder and concat/MLP edge decoder. '
        'The right-hand block of Figure 1(b) is an expansion of the same encoder.')
    manifest['annotation_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['derived_files'] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    for path in paths:
        print(path)
