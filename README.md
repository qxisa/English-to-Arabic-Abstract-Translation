# English to Arabic Scientific Abstract Translation

This project develops and evaluates machine translation systems for translating English scientific abstracts into Arabic, using the ASCAT dataset as both training data and benchmark.

## Dataset

ASCAT (Arabic Scientific Corpus and Benchmark for Advanced Translation Evaluation) contains 536 English-Arabic parallel scientific abstract pairs spanning five domains: Physics, Mathematics, Computer Science, Quantum Mechanics, and AI. Each abstract averages roughly 125 English words and 112 Arabic words.

Dataset: https://www.google.com/url?q=https%3A%2F%2Fhuggingface.co%2Fdatasets%2FNAMAA-Space%2FASCAT-Arabic-Scientific-Translation

## Notebooks

### preprocessing and phase 2.ipynb

Covers data loading, cleaning, and preparation for model training. Steps include removing duplicates, normalizing English and Arabic text (whitespace, encoding artifacts, diacritics, Alef/Taa/Kashida normalization), filtering short abstracts, and producing stratified train/validation/test splits at a 70/15/15 ratio. A random oversampling step balances minority scientific domains in the training set. The notebook also includes a literature review, a description of the end-to-end pipeline, and an outline of the evaluation methodology.

### opus_mt_refined.ipynb

Runs Helsinki-NLP/opus-mt-en-ar (MarianMT) in zero-shot mode as a baseline. Loads the preprocessed test set, generates translations, and computes BLEU and BERTScore. Charts are produced to visualize per-domain performance.

### mbart_m2m_revised.ipynb

Compares two models:

- mBART-50 (facebook/mbart-large-50-many-to-many-mmt): used zero-shot as a second baseline, no fine-tuning applied.
- M2M100 / small100 (alirezamsh/small100): a 300M-parameter community adaptation of facebook/m2m100_418M, fine-tuned on the balanced ASCAT training split using LoRA (rank 16, applied to query and value projections). The smaller variant was chosen because the full 418M model exceeded GPU memory limits on the available hardware.

After training, both models are evaluated on the test set using BLEU and BERTScore. Fine-tuning produced BERTScore F1 gains over the zero-shot mBART-50 baseline in every domain, ranging from 0.87 percentage points (Physics) to 5.91 percentage points (Other), with Computer Science gaining 4.90 pp and Mathematics gaining 3.81 pp.

## Models

| Model | Type | Parameters |
|---|---|---|
| Helsinki-NLP/opus-mt-en-ar | Zero-shot baseline | ~74M |
| facebook/mbart-large-50-many-to-many-mmt | Zero-shot baseline | ~600M |
| alirezamsh/small100 + LoRA | Fine-tuned | ~300M |

## Evaluation Metrics

BLEU (sacrebleu) and BERTScore are used as primary metrics. BERTScore captures semantic similarity and is particularly informative for Arabic, where surface-level n-gram overlap can underestimate translation quality.

## Results

Fine-tuning the smaller M2M100 variant on domain-specific ASCAT data matched or exceeded the zero-shot performance of the larger mBART-50 model across all five scientific domains. The largest gains were observed in the most diverse and underrepresented domains.
