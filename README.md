# SHL Hiring Assessment — Grammar Scoring Engine for Spoken Audio

Predict a continuous **grammar score (0–5)** for 45–60 s English speech recordings (Kaggle "SHL Hiring Assessment").
Leaderboard metrics: **Pearson correlation** and **RMSE**.

> **Status:** all notebooks were run on Kaggle (GPU T4). Scores below are the real public-leaderboard and cross-validation numbers.
> Notebook outputs are deliberately **cleared** in this repo because they print transcript text derived from the competition data, which the rules forbid redistributing.
> The executed notebooks (with all outputs, plots and the training RMSE) are the committed versions on Kaggle: **<https://www.kaggle.com/code/nandhanamnambiar222/notebookc0c01e1016/output?scriptVersionId=356755638>**.

## Approach (simple and explainable)

```
.wav → preprocess (16 kHz mono, trim edges, loudness) → Whisper (local, once) → transcript + word timestamps
      → text statistics · GPT-2 perplexity · (LanguageTool) · fluency/pauses · (LLM rubric)   [engineered features]
      → sentence-encoder / RoBERTa transcript embeddings · WavLM audio embedding             [embeddings]
      → Ridge / SVR (embeddings), Ridge / LightGBM (engineered)  → repeated stratified 5-fold CV + ablation
      → top-3 models → non-negative linear stack (weights from out-of-fold predictions) → clip [0, 5] → submission.csv
```
`(…)` = optional, auto-skipped if unavailable.

## Repository layout

| path | purpose |
|---|---|
| `00_baseline.ipynb` | fastest path: Whisper → MiniLM embedding → Ridge → valid `submission.csv` |
| `01_extract_features.ipynb` | run **once**: all features and embeddings cached to disk |
| `final.ipynb` | loads caches only (never runs Whisper): EDA, CV, ablation, ensemble, final fit, REPORT, interview prep |
| `src/` | documented modules (`config`, `data`, `audio`, `transcribe`, `text_features`, `fluency`, `lm_features`, `grammar_tool`, `embeddings`, `llm_rubric`, `models`, `metrics`, `plots`, `submission`, `utils`) |
| `requirements.txt`, `.gitignore` | dependencies; data/caches/CSVs are excluded so the dataset is not redistributed |

## How to run on Kaggle (GPU notebook)

1. Add this repo as a **private Kaggle Dataset** (or clone it into `/kaggle/working/shl-grammar-scoring`). The notebooks find `src/` automatically.
2. Attach the competition data (mounted read-only at `/kaggle/input/<competition-folder>/`). Nothing is uploaded anywhere.
3. Internet **ON** for the first run (downloads Whisper / encoder weights), or **OFF** with weights supplied from Kaggle Models / a Dataset — then set `OFFLINE=True` and put local paths in `CONFIG` (`WHISPER_MODEL` needs a *CTranslate2* Whisper folder, e.g. `Systran/faster-whisper-small`).
4. Edit the single `CONFIG` cell at the top of each notebook (the same dict in all three): `DATA_DIR`, optional path/column overrides, model names, switches.
5. Run `00_baseline.ipynb` → submit `submission.csv` (≈ within an hour).
6. Run `01_extract_features.ipynb` (reuses the Whisper cache; computes everything else once).
7. Run `final.ipynb`. In a *new* session, save notebook 01's output as a **private** Dataset, attach it and point `CACHE_DIR` at it.

Seeds are fixed (`CONFIG["SEED"]`, CV seeds = SEED + repeat index). GPU kernels can still introduce tiny nondeterminism in embeddings.

## Results

| experiment | CV RMSE | CV Pearson | notes |
|---|---|---|---|
| constant mean (label std) | _run to fill_ | – | "learned nothing" reference |
| 00 baseline (MiniLM + Ridge) | _run to fill_ | _run to fill_ | |
| best single model (final) | _run to fill_ | _run to fill_ | |
| final ensemble | _run to fill_ | _run to fill_ | training RMSE printed in `final.ipynb` §8 |

`final.ipynb` writes `ablation_results.csv` and `results_log.csv`; copy the real numbers here after running.

## Rules compliance

* No external datasets. Pretrained models are used only as local feature extractors.
* Everything runs locally; LanguageTool is started as a **local** server (never a remote one); no API calls.
* Test labels are random placeholders and are never used (`src/data.py` discards them and sets test labels to NaN).
* All imputing, scaling and ridge-penalty selection happen inside CV folds (sklearn `Pipeline`); no feature selection, no early stopping.
* Libraries are open source; licences recalled from memory, **please verify** (faster-whisper MIT, librosa ISC, transformers / sentence-transformers Apache-2.0, LightGBM MIT, torch BSD, scikit-learn BSD, matplotlib PSF-style, language-tool-python GPL-3.0). Model weights have their own licences — check them for the models you choose.

## What is verified / unverified

**Smoke-tested in a sandbox on synthetic data** (no GPU, no internet, no audio): config, data/column/path discovery, text features, fluency features, stratified repeated CV with per-fold printing, constant baseline, Ridge/SVR (LightGBM was *not* installed there, so the sklearn fallback ran), stacking and second-level CV, submission writer + validator, all plots, the Whisper cache/resume logic (with a mock model), and the full code path of `final.ipynb` on a fake feature cache.

**NOT verified — expect to debug these:**
* Actual Whisper transcription (`faster-whisper` API usage, word timestamps, GPU/compute type, offline model folder).
* Audio loading/preprocessing (`librosa`, `soundfile`) and `scan_audio_info`.
* SentenceTransformer, RoBERTa mean-pool, GPT-2 perplexity, WavLM embedding code on the GPU; memory use for 60 s clips.
* LanguageTool (Java, download, match attributes differ across versions) and the optional LLM rubric scorer (chat template, single-token digits).
* Kaggle specifics: folder layout auto-discovery, `src/` discovery, pip availability, runtime limits.
* The real LightGBM code path (`make_lgbm`).
* `00_baseline.ipynb` and `01_extract_features.ipynb` were only syntax-checked (not executed).
* Any claim about which features/models help — that is exactly what the ablation will tell you.

## Assumptions about the data (fix in CONFIG if wrong)

1. `train.csv`, `test.csv`, `sample_submission.csv` exist somewhere under `DATA_DIR`, with those exact names.
2. `train.csv` has one file-name column (values ending in `.wav`, with or without extension) and one numeric label column; auto-detected, override with `ID_COL` / `LABEL_COL`.
3. `test.csv` has the same file-name column name (else its first column is used); its label column is ignored.
4. **Submission template is `test.csv`, not `sample_submission.csv`.** The provided sample has 204 rows with different file names, but the grader rejects anything except 216 rows ("Submission has 204 rows but 216 were expected"). The notebooks therefore write one row per `test.csv` row, in that order. Train and test folders also reuse file names (e.g. `audio_5.wav` in both, different audio); every cache here is keyed by split + name, so they never mix.
5. Audio lives in folders whose names contain `train` / `test`, files are `.wav`, and file stems are unique within a split; otherwise a global search under `DATA_DIR` is used.
6. Speech is English; labels lie in [0, 5].
7. Train and test file names do not need to be distinct (cache keys include the split).


## 02_submit.ipynb (final model + submission file)
Loads the cached features from `01_extract_features` (attach the COMMITTED 01 notebook as an input), trains small LightGBM models and writes a validated `submission.csv` (one row per `test.csv` row). Set `VARIANT` in the first cell:
`A` = all features, all clips; `B` = length-dependent features removed and the label-0 batch (file IDs >= 5000, absent from the test names) removed; `C` = length-dependent features removed, all clips.

## 03_audio_layers.ipynb (optional audio improvement)
Extracts mean+std pooled embeddings from every layer of the audio model, scores each layer with ridge + repeated stratified CV, picks the best combination (best layer / mean of best 3 / mean of all / concatenation) and writes it as `audio_wavlm.npy` into a copy of the feature cache (original kept as `audio_wavlm_last.npy`). Attach the COMMITTED 03 notebook (instead of 01) as the input of `final.ipynb`.


## Results (Kaggle public leaderboard, lower RMSE is better)

| Step | Model | Public score |
|---|---|---|
| Constant guess | training mean for every clip | 1.0050 |
| `00_baseline` | Whisper small → MiniLM embedding → Ridge | 0.7262 |
| `02_submit` variant C | LightGBM, engineered features without length features, all clips | 0.4635 |
| `02_submit` variant B | same, label-0 batch removed | 0.4684 |
| `02_submit` variant A | LightGBM, all engineered features, all clips | 0.4456 |
| **`final.ipynb`** | **stack of 3 ridge models on embeddings + engineered features** | **0.4081** |

Cross-validation (repeated stratified 5-fold, 769 training clips): constant 1.238 RMSE; best single model `ridge_everything` 0.556 RMSE / 0.895 Pearson; final stack 0.528 RMSE / 0.905 Pearson.
Training RMSE of the final model: **<COPY THE "TRAINING RMSE" LINE FROM THE final.ipynb LOG HERE>**.

What the ablation showed (CV RMSE on all training clips): the WavLM audio embedding alone gives 0.580, fluency / ASR-confidence features alone 0.753, transcript-only embeddings 0.83 to 0.99, GPT-2 perplexity 1.14 and LanguageTool counts 1.20. Most of the signal comes from the audio and from how confidently Whisper recognises the speech, not from text-level grammar checks.

Data observations that shaped the pipeline:
- The 37 training clips labelled 0 all have file IDs of 5000 or higher, and no test file name does. Removing them did not help the leaderboard (variant B), so they are kept.
- Test clips are shorter than training clips (median about 45 s vs 60 s). Dropping length-dependent features (variant C) scored slightly worse than keeping them (variant A).
- `sample_submission.csv` has 204 rows with different names; the grader expects 216 rows, one per `test.csv` row, so submissions follow `test.csv`.

## Limitations
About 770 labelled clips, so fold-to-fold RMSE varies by several hundredths. Choosing among a handful of variants on the public leaderboard is slightly optimistic. Whisper may tidy some grammar errors in the transcript.
