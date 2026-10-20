# ShashiOCR: Accurate OCR for Telugu Documents with Fine-Tuned Vision-Language Models

Official code and model release for the paper
**"Towards Accurate OCR for Telugu Documents Using Fine-Tuned Vision-Language Models"**
(CVIP 2026).

ShashiOCR is a compact **1B-parameter** vision-language model that performs
**unified** Telugu document OCR: in a single decoding pass it predicts text
regions, their bounding boxes, and the reading order, with no separate layout
detection, recognition, or reading-order modules.

It is trained **without any manual page transcription**, using a combination of
teacher-generated pseudo-labels on real archival pages and synthetic pages with
exact ground truth. On a human-verified 452-page benchmark it reaches
**95.73% CRR / 79.67% WRR**, outperforming its own 4B teacher and every
evaluated open, locally deployable OCR system.

---

## Table of contents

- [Highlights](#highlights)
- [Results](#results)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Output format](#output-format)
- [Evaluation](#evaluation)
- [Training](#training)
- [Repository structure](#repository-structure)
- [Datasets](#datasets)
- [Limitations](#limitations)
- [Citation](#citation)
- [Acknowledgment](#acknowledgment)
- [License](#license)

---

## Highlights

- **Unified task formulation.** A page image is mapped directly to a sequence of
  regions `(box, text)`, with boxes on a normalized `1000 x 1000` grid emitted
  in reading order. Localization, reading order, and transcription are learned
  jointly, so errors cannot propagate between pipeline stages.
- **Structured page-level distillation.** A 4B multilingual teacher
  (Chandra OCR) produces pseudo-labels for real Telugu pages; a 1B student
  (LightOnOCR-2-1B) of a different architecture is distilled on them. The
  student ends up **better than the teacher**.
- **Synthetic supervision that fixes teacher bias.** 100K synthetic Telugu pages
  with exact ground truth counter the teacher's *systematic* errors, which
  pseudo-labels alone cannot correct.
- **One commodity GPU.** LoRA (38.3M trainable parameters, 3.7% of the model),
  bfloat16, gradient checkpointing and chunk-based continual training keep the
  whole pipeline inside a single 48 GB GPU.
- **Script-agnostic recipe.** Only the fonts, the synthetic generator's script
  coverage, and the teacher model are Telugu-specific; everything else transfers
  to other low-resource Indic scripts.

---

## Results

Performance on the 452-page human-verified Telugu benchmark (Bhashini test set).
CRR and WRR are character and word recognition rates; micro scores are computed
over the whole corpus, macro scores are averaged equally across pages.

| Model type       | License | System              | Size  | Micro CRR | Micro WRR | Macro CRR | Macro WRR |
|------------------|---------|---------------------|-------|-----------|-----------|-----------|-----------|
| Generic VLM      | Closed  | Sarvam              | 3B    | 95.31     | 90.62     | 95.08     | 89.79     |
| Generic VLM      | Closed  | Google Cloud Vision | --    | 96.33     | 90.63     | 96.11     | 89.85     |
| Generic VLM      | Closed  | Gemini (base)       | --    | 70.09     | 35.82     | 70.89     | 36.12     |
| Generic VLM      | Open    | Qwen2.5-VL-7B       | 7B    | 34.97     | 1.82      | 35.99     | 2.47      |
| VLM-based OCR    | Open    | Surya1              | 9B    | 84.16     | 64.79     | 82.89     | 62.04     |
| VLM-based OCR    | Open    | Surya2              | 0.65B | 89.88     | 67.16     | 89.38     | 65.07     |
| VLM-based OCR    | Open    | LightOnOCR-2-1B     | 1B    | 25.58     | 2.42      | 26.53     | 2.72      |
| VLM-based OCR    | Open    | PaddleOCR-VL        | 0.9B  | 55.14     | 12.89     | 54.72     | 11.44     |
| VLM-based OCR    | Open    | GOT-OCR2.0          | 0.6B  | 41.32     | 20.81     | 37.60     | 40.22     |
| VLM-based OCR    | Open    | DeepSeek-OCR        | 3B    | 42.84     | 5.79      | 42.11     | 5.20      |
| VLM-based OCR    | Open    | dots.ocr            | 1.7B  | 63.92     | 49.95     | 62.17     | 48.12     |
| VLM-based OCR    | Open    | ChandraOCR (teacher)| 4B    | 90.91     | 69.72     | 89.88     | 68.32     |
| **VLM-based OCR**| **Open**| **ShashiOCR (ours)**| **1B**| **95.73** | **79.67** | **95.46** | **78.09** |
| Traditional OCR  | Open    | Tesseract V5        | --    | 93.25     | 71.13     | 92.26     | 67.62     |
| Traditional OCR  | Open    | EasyOCR             | --    | 87.64     | 57.61     | 86.75     | 55.31     |

**Takeaways**

- **+70.15 CRR / +77.25 WRR** over the unadapted LightOnOCR-2-1B backbone we
  start from: almost all of the accuracy comes from our adaptation, not from the
  pretrained model.
- **+4.82 CRR / +9.95 WRR** over the 4B teacher that generated the training
  supervision: the student is not capped by its teacher.
- The 7B Qwen2.5-VL reaches only 34.97% CRR, so scale alone does not solve
  Telugu document OCR.

### Performance by document collection

| Collection                           | Micro CRR | Micro WRR |
|--------------------------------------|-----------|-----------|
| Dense prose                          | 97.49     | 85.21     |
| Literary-criticism articles          | 97.80     | 84.12     |
| Devotional verse + large display text| 89.75     | 63.78     |

Errors are concentrated rather than spread: in the hardest collection, 12 of 14
pages fall below 80% CRR, and five of the six near-complete failures occur
there.

---

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/SKS200/telugu-document-ocr.git
cd telugu-document-ocr

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Install PyTorch matching your CUDA version first if the default wheel does not
suit your driver, for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Hardware.** Inference needs roughly 6 GB of VRAM in bfloat16 for the 1B model.
Training was done on a single NVIDIA RTX 6000 Ada (48 GB).

---

## Quick start

### Command line

```bash
python src/infer.py \
  --model SKS200/shashiocr-1b \
  --input samples/page_001.jpg \
  --output out/page_001.json \
  --visualize out/page_001_boxes.jpg
```

Run on a folder of images or on a PDF:

```bash
python src/infer.py --model SKS200/shashiocr-1b --input data/pages/ --output out/
python src/infer.py --model SKS200/shashiocr-1b --input scan.pdf  --output out/ --dpi 300
```

> **TODO:** replace `SKS200/shashiocr-1b` with the actual Hugging Face model ID
> (or a local checkpoint path) once the weights are published.

### Python

```python
from src.infer import ShashiOCR

ocr = ShashiOCR("SKS200/shashiocr-1b", device="cuda", dtype="bfloat16")
page = ocr.read("samples/page_001.jpg")

print(page.text)              # full page text in reading order
for region in page.regions:
    print(region.box, region.text[:40])
```

---

## Output format

Each page is returned as a list of regions **already in reading order**.
Coordinates are `[x1, y1, x2, y2]` on a normalized `1000 x 1000` grid, so they
are resolution independent; multiply by the page width and height to get pixels.

```json
{
  "image": "page_001.jpg",
  "width": 2480,
  "height": 3508,
  "regions": [
    {"id": 0, "box": [112,  86, 889, 141], "text": "..."},
    {"id": 1, "box": [ 96, 168, 486, 742], "text": "..."},
    {"id": 2, "box": [514, 168, 904, 742], "text": "..."}
  ]
}
```

The `text` fields contain Telugu in Unicode NFC. Reading order for multi-column
pages follows the column-aware rule used in training: regions with more than 40%
horizontal overlap are grouped into a column, columns are ordered left to right,
and regions within a column top to bottom.

---

## Evaluation

`src/evaluate.py` computes CRR and WRR exactly as reported in the paper, both
micro (corpus-level) and macro (per-page averaged).

```bash
python src/evaluate.py \
  --pred  out/ \
  --gt    data/bhashini_test/ground_truth/ \
  --report out/metrics.json
```

The script expects one prediction file and one ground-truth file per page,
matched by filename stem. It accepts `.json` (the format above) or plain `.txt`.
Text is Unicode-NFC normalized and whitespace-collapsed before scoring; pass
`--no-normalize` to disable this.

```
CRR = (1 - edit_distance(pred_chars, gt_chars) / len(gt_chars)) * 100
WRR = (1 - edit_distance(pred_words, gt_words) / len(gt_words)) * 100
```

---

## Training

Training has two stages that share the same configuration and run over data in
chunks of 5,000 pages, with a 90/10 train/validation split per chunk, up to 10
epochs with early stopping, and the lowest-validation-loss checkpoint kept. The
adapter learned on one chunk initializes the next, so the model accumulates
knowledge across the corpus while giving frequent checkpoints.

**Stage 1 - synthetic adaptation** (100K synthetic pages with exact ground truth):

```bash
python src/train.py \
  --stage synthetic \
  --base-model lightonai/LightOnOCR-2-1B-bbox \
  --data data/synthetic_100k \
  --output checkpoints/stage1
```

**Stage 2 - real pseudo-labeled fine-tuning** (750K stratified real pages):

```bash
python src/train.py \
  --stage real \
  --base-model lightonai/LightOnOCR-2-1B-bbox \
  --init-adapter checkpoints/stage1 \
  --data data/real_pseudo_750k \
  --output checkpoints/shashiocr
```

> **TODO:** adjust the script name and flags to match the training entry point
> actually in this repository.

### Hyperparameters

| Setting | Value |
|---|---|
| Base model | LightOnOCR-2-1B-bbox (Pixtral vision encoder, 24 layers; Qwen3 decoder, 28 layers) |
| Adaptation | LoRA, rank 32, alpha 64, dropout 0.05, base frozen |
| LoRA targets | attention q/k/v/o and FFN gate/up/down, in **both** encoder and decoder |
| Trainable parameters | 38.3M (3.7% of the model) |
| Optimizer | AdamW, beta = (0.9, 0.999), no weight decay |
| Learning rate | 1e-4, cosine schedule, 3% warmup |
| Effective batch size | 32 (2 pages per device x 16 accumulation steps) |
| Precision | bfloat16 with gradient checkpointing |
| Coordinate-token loss weight | w = 6 |
| Chunk size | 5,000 pages, 90/10 train/val, up to 10 epochs, early stopping |
| Hardware | 1 x NVIDIA RTX 6000 Ada (48 GB) |

Coordinate tokens (vocabulary entries consisting only of digits and commas) are
up-weighted by a factor of six so that text prediction does not dominate spatial
prediction:

```
L = - sum_n  w_n * log p(s_n | s_<n, I),    w_n = 6 for coordinate tokens, else 1
```

### Training-time monitoring

Token-level loss does not reflect structural quality, so predictions on a fixed
validation subset are scored after every epoch for: character error rate, box
precision and recall at IoU >= 0.5, mean IoU, region-count agreement, and the
median predicted box width and horizontal spread. The last two flag degenerate
outputs in which boxes collapse toward a page margin.

---

## Repository structure

> **TODO:** update to match the real layout.

```
telugu-document-ocr/
  src/
    infer.py           inference, single image / folder / PDF
    evaluate.py        CRR and WRR, micro and macro
    train.py           two-stage LoRA fine-tuning
    pseudo_label.py    teacher inference + filtering + serialization
    synth/             synthetic Telugu page generator
  configs/
  samples/
  requirements.txt
  README.md
```

---

## Datasets

| Split | Pages | Source |
|---|---|---|
| Synthetic pretraining | 100K | Generated with a mesh-candidate layout pipeline over 18 validated Telugu font families; exact region boxes and text |
| Real pseudo-labeled | ~750K | Stratified sample of a ~7.5M-page community-digitized archive of Telugu newspapers, periodicals and books; labeled by the Chandra-4B teacher |
| Benchmark (test) | 452 | Bhashini real test set, human-verified ground truth, disjoint from training |

**Stratified sampling.** Page counts per publication are long-tailed, so uniform
sampling would over-represent a few large newspaper collections. Publications
with fewer than 100 pages contribute 10% (at least one page); 100-500 pages
contribute about 10%; more than 500 pages contribute at most 100 pages. The
procedure is deterministic and reproducible.

**Pseudo-label filtering.** Valid text-bearing regions are kept; figures,
photographs, separators and table boundaries are discarded; pages with
unreadable images, invalid predictions, or no valid text regions are dropped
entirely.

> **TODO:** add download links or a request form for the released data.

---

## Limitations

- Residual errors are mostly **sub-character**: vowel-sign (matra)
  substitutions, deletions and insertions account for roughly 40% of all edits,
  followed by visually similar consonant confusions and partially lost
  conjuncts. Whole-word substitution is not a dominant failure mode.
- Severely degraded pages (heavy show-through, ink bleed, foxing, overlaid
  stamps, broken or filled-in glyphs) account for most catastrophic failures.
- Word-level recognition still trails the strongest commercial services
  (Google Cloud Vision, Sarvam), which are not locally deployable.
- The model is trained for **printed** Telugu; handwriting is out of scope.

---

## Citation

If you use this code, the model, or the benchmark, please cite:

```bibtex
@inproceedings{sahoo2026shashiocr,
  title     = {Towards Accurate {OCR} for {T}elugu Documents Using Fine-Tuned
               Vision-Language Models},
  author    = {Sahoo, Shashikanta and Mondal, Ajoy and
               Yarlagadda, Kiran Chandra and Jawahar, C. V.},
  booktitle = {Proceedings of the International Conference on Computer Vision
               and Image Processing (CVIP)},
  year      = {2026}
}
```

> **TODO:** add `pages`, `publisher`, `series` and `doi` once the proceedings are
> published.

---

## Acknowledgment

This work is supported by MeitY, Government of India, through the NLTM-Bhashini
project.

We build on [LightOnOCR-2-1B](https://huggingface.co/lightonai) as the student
backbone and [Chandra](https://github.com/datalab-to/chandra) as the teacher
model, and we thank their authors for releasing them openly.

---

## License

> **TODO:** choose and add a license (for example Apache-2.0 or MIT) in a
> `LICENSE` file, and note separately that the base model, teacher model, and
> archival page images carry their own licenses and terms.

---

## Contact

Shashikanta Sahoo - `shashikanta.sahoo@students.iiit.ac.in`
CVIT, International Institute of Information Technology, Hyderabad, India
