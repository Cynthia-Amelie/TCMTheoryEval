# TCMTheoryEval: a benchmark dataset for evaluating large language models on six-meridian syndrome differentiation

TCMTheoryEval is a Chinese benchmark for evaluating large language models on four sequential tasks related to Traditional Chinese Medicine Six-Meridian theory:

1. question-keyword extraction;
2. Six-Meridian multi-label classification;
3. single-answer selection from ten options; and
4. explanation generation.

The release contains 353 expert-reviewed questions in fixed training, validation, and test splits. The split files are UTF-8 encoded JSON arrays.

## Public-release status

- Source code is released under the [MIT License](LICENSE).
- The data files under `data/` are released under [CC BY 4.0](DATA_LICENSE.md).

## Repository structure

```text
TCMTheoryEval/
├── data/
│   ├── train.json
│   ├── validation.json
│   └── test.json
├── code/
│   ├── run_evaluation.py
│   └── score_predictions.py
├── models/
│   └── bert-base-chinese/       # optional local checkpoint for scoring
├── results/                     # generated model outputs and scores
├── README.md
└── requirements.txt
```


## Data splits

| File | Records | Intended use |
|---|---:|---|
| `data/train.json` | 235 | Training or in-context method development |
| `data/validation.json` | 59 | Prompt and pipeline validation |
| `data/test.json` | 59 | Held-out benchmark inputs; gold labels are withheld |

The three splits are fixed components of this release. Users should not reshuffle or merge them when reporting results comparable with the accompanying manuscript.

## Record schema

Training and validation records contain complete gold annotations. Public test records retain the same schema, but `question_keywords` is an empty list and `question_answer`, `six_meridian_answer`, and `explanation` are empty strings; the real gold fields are withheld for author-side evaluation.

| Field | Description |
|---|---|
| `question_id` | Unique question identifier |
| `question` | Chinese question stem |
| `question_keywords` | Expert-reference keywords extracted from the question |
| `question_options` | Ten answer options encoded A–J |
| `question_answer` | Reference answer letter in A–J |
| `six_meridian_options` | Six-Meridian label choices encoded A–F |
| `six_meridian_answer` | One or more reference Six-Meridian labels |
| `explanation` | Expert-reference explanation |

## Installation

Python 3.8.20 was used for the experiments reported in the manuscript. Create a virtual environment and install the dependencies from the repository root:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the four-stage evaluation

The evaluation pipeline reads the fixed test split by default and writes model responses to `results/validation_results.json`:

```bash
python code/run_evaluation.py
```

API credentials are read from environment variables. Depending on the selected models, configure the relevant variables, such as `DASHSCOPE_API_KEY`, `OPENAI_API_KEY`, and provider endpoint variables documented in `code/run_evaluation.py`.

Create a local configuration file from the public template:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

To run a subset of the configured models or a small smoke test:

```bash
python code/run_evaluation.py --models qwen3-8b --limit 2
```

The benchmark protocol uses four sequential zero-shot prompts with `temperature=0`, `top_p=1`, `max_tokens=3000`, and no seed, retrieval, web access, or external tools.

## Scoring predictions

The public test split excludes gold labels. Only maintainers with the private `data/test_labels_private.json` file can score test predictions. Public users may develop and score on the labelled training and validation splits.

Scoring requires a local `bert-base-chinese` checkpoint containing its configuration, tokenizer, vocabulary, and model weights. By default, the script looks for:

```text
models/bert-base-chinese/
```

After generating predictions, run:

```bash
python code/score_predictions.py
```

Alternatively, provide a checkpoint explicitly:

```bash
python code/score_predictions.py --bertscore-model /path/to/bert-base-chinese
```

The scoring implementation follows the manuscript: Task 1 combines ROUGE-L recall and BERTScore F1; Task 2 uses set-level F1; Task 3 uses strict answer-letter exact match; and Task 4 combines ROUGE-L F1 and BERTScore F1. The composite weights are 0.20, 0.30, 0.40, and 0.10, respectively.

## Reproducibility notes

- Run commands from the repository root so that relative paths resolve correctly.
- Training and validation files contain gold annotations. The public test file preserves the same fields but uses empty gold-label placeholders.
- Online model outputs can vary when providers update hosted model versions, even with deterministic request parameters.

## Citation

If you use this dataset, please cite the Zenodo dataset record:

> Chen, Z., Tang, X., Zhong, L., Xu, L., Yin, X., Zhang, L., Yu, D., Zhang, Z., Li, H., & Qin, P. (2026). TCMTheoryEval: A Benchmark Dataset for Evaluating Large Language Models on Six-Meridian Syndrome Differentiation (Version 1.0.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.22660878

Citation metadata is provided in [CITATION.cff](CITATION.cff). The all-versions DOI is https://doi.org/10.5281/zenodo.22660877.

## Licence

The source code and documentation are available under the [MIT License](LICENSE). The data files under `data/` are available under [CC BY 4.0](DATA_LICENSE.md).

## Contact

Add the corresponding author's name, institutional affiliation, and durable contact address here before release.








