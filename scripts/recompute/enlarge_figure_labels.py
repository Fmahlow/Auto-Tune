"""Enlarge the labels in the image mosaics used as Figures 1, 2, and 4.

The source photos are extracted from the current PDFs, so rebuilding does not
require the original image-generation outputs. Figure 4 uses its saved original
composite JPEG to avoid recompression across repeat runs.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess

import pdfplumber
from PIL import Image
from reportlab import rl_config
from reportlab.pdfgen import canvas


PAPER = Path(__file__).resolve().parents[2] / "paper_tex"
rl_config.useA85 = 0


def extract_images(pdf: Path, prefix: Path, *options: str) -> list[Path]:
    subprocess.run(["pdfimages", *options, str(pdf), str(prefix)], check=True)
    return sorted(prefix.parent.glob(prefix.name + "-*"))


def image_mosaic(number: int, work: Path) -> None:
    pdf = PAPER / f"fig_{number}_paper.pdf"
    with pdfplumber.open(pdf) as document:
        page = document.pages[0]
        width, height = page.width, page.height
        boxes = page.images
    images = extract_images(pdf, work / f"fig{number}", "-png")
    if len(images) != 8 or len(boxes) != 8:
        raise ValueError(f"Figure {number}: expected eight image tiles")

    output = work / f"fig_{number}_paper.pdf"
    drawing = canvas.Canvas(str(output), pagesize=(width, height))
    for image, box in zip(images, boxes):
        drawing.drawImage(
            str(image), box["x0"], height - box["top"] - box["height"],
            width=box["width"], height=box["height"],
        )
    drawing.setFont("Helvetica-Bold", 18)
    for index, box in enumerate(boxes):
        label = f"({chr(ord('a') + index)})"
        x = box["x0"] + box["width"] / 2
        y = height - 22 if index < 4 else 8
        drawing.drawCentredString(x, y, label)
    drawing.save()
    output.replace(pdf)


def comparison_mosaic(work: Path) -> None:
    pdf = PAPER / "fig_4_paper.pdf"
    source = PAPER / "fig_4_source.jpg"
    with pdfplumber.open(pdf) as document:
        page = document.pages[0]
        height = page.height
    if not source.exists():
        raise FileNotFoundError("Figure 4: missing original fig_4_source.jpg")
    with Image.open(source) as original:
        original_width = height * original.width / original.height

    # The original row labels occupy a narrow white strip. Widen that strip
    # without shrinking the photographs, then replace all rasterized labels.
    left_extension = 55
    margin_width = 122
    output = work / "fig_4_paper.pdf"
    drawing = canvas.Canvas(str(output), pagesize=(original_width + left_extension, height))
    drawing.drawImage(
        str(source), left_extension, 0,
        width=original_width, height=height,
    )
    drawing.setFillColorRGB(1, 1, 1)
    drawing.rect(0, 0, margin_width, height, fill=1, stroke=0)
    drawing.rect(left_extension, height - 19.5, original_width, 19.5, fill=1, stroke=0)
    drawing.setFillColorRGB(0, 0, 0)

    drawing.setFont("Helvetica-Bold", 17)
    for title, x in zip(("Chamanto", "Cuscuz", "Jian", "Lokum"),
                        (139.3, 290.1, 441.0, 592.0)):
        drawing.drawCentredString(left_extension + x, height - 15.5, title)

    drawing.setFont("Helvetica-Bold", 16)
    for lines, center in ((('DreamBooth', '+LoRA'), 261),
                          (('Textual', 'Inversion'), 90)):
        drawing.drawCentredString(margin_width / 2, center + 4, lines[0])
        drawing.drawCentredString(margin_width / 2, center - 15, lines[1])
    drawing.save()
    output.replace(pdf)


def main() -> None:
    with TemporaryDirectory(prefix="paper-figure-labels-") as directory:
        work = Path(directory)
        for number in (1, 2):
            image_mosaic(number, work)
        comparison_mosaic(work)


if __name__ == "__main__":
    main()
