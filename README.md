# FLEX

FLEX is a benchmark and framework for unified, rigorous few-shot NLP evaluation.
FLEX enables:
- First-class NLP support
- Support for meta-training
- Reproducible fewshot evaluations
- Extensible benchmark creation (benchmarks defined using [HuggingFace Datasets](https://huggingface.co/datasets))
- Advanced sampling functions for creating episodes with class imbalance, etc.

For more context, see our paper (appearing on Arxiv shortly).

Leaderboard (coming soon): <https://leaderboard.allenai.org/flex>

## Installation

- Clone the repository `git clone git@github.com:allenai/flex.git`
- Create a Python 3 environment (3.7 or greater), eg using `conda create --name flex python=3`
- Activate the environment `conda activate flex`
- Install the package locally with `pip install -e .`

## Model evaluation

"Challenges" are datasets of sampled tasks for evaluation. They are defined in `fewshot/challenges/__init__.py`.

To evaluate a model on challenge `flex` (our first challenge), you should write a program that produces
a `predictions.json`, for example:
```python
#!/usr/bin/env python3
import random
from typing import Iterable, Dict, Any, Sequence
from fewshot import make_challenge, Model


class YourModel(Model):
    def fit_and_predict(
        self,
        support_x: Iterable[Dict[str, Any]],
        support_y: Iterable[str],
        target_x: Iterable[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> Sequence[str]:
        """Return random label predictions for a fewshot task."""
        train_x = [d['txt'] for d in support_x]
        train_y = support_y
        test_x = [d['txt'] for d in target_x]
        test_y = [random.choice(metadata['labels']) for _ in test_x]
        # >>> print(test_y)
        # ['some', 'list', 'of', 'label', 'predictions']
        return test_y


if __name__ == '__main__':
    evaluator = make_challenge("flex")
    model = YourModel()
    evaluator.save_model_predictions(model=model)
```

Warning: Running the above script will download and all the required datasets for sampling and creating the challenge.

Evaluation produces `predictions.json` output files with the format:
```
{
    "[QUESTION_ID]": {
        "label": "[CLASS_LABEL]",  # Currently an integer converted to a string
        "score": float  # Only used for ranking tasks
    },
    ...
}
```
### [Optional] Parallelizing evaluation
Two options are available for parallelizing evaluation.

First, one can restrict evaluation to a subset of tasks with indices from `[START]` to `[STOP]` (exclusive) via
```python
evaluator.save_model_predictions(model=model, start_task_index=[START], stop_task_index=[STOP])
```
Notes:
- You may use `stop_task_index=None` (or omit it) to avoid specifying an end.
- You can find the total number of tasks in the challenge with `fewshot.get_challenge_spec([CHALLENGE]).num_tasks`.
- To merge partial evaluation outputs into a complete `predictions.json` file, use `fewshot merge partial1.json partial2.json ... predictions.json`.

The second option will call your model's `.fit_and_predict()` method with batches of `[BATCH_SIZE]` tasks, via
```python
evaluator.save_model_predictions(model=model, batched=True, batch_size=[BATCH_SIZE])
```

## Result validation and scoring

To validate the contents of your predictions, run:

`fewshot validate --challenge_name flex --predictions /path/to/predictions.json` 

This validates all the inputs and takes some time. Substitute `flex` for another challenge to evaluate on a different challenge.

There is also a `score` CLI command which should not be used on the final challenge except when reporting final results.

## Model training

For an example of how to train a model on the benchmark, see <https://github.com/allenai/unifew> (coming soon). An additional example based on the following paper is available in the `/baselines/bao/` directory:

> Yujia Bao*, Menghua Wu*, Shiyu Chang, and Regina Barzilay. Few-shot Text Classification with Distributional Signatures. In International Conference on Learning Representations 2020

## Benchmark construction and optimization

To add a new benchmark (challenge), you must edit `fewshot/challenges/__init__.py` or otherwise add it to the registry.

For an example of how to optimize the sample size of the challenge, see `scripts/README-sample-size.md`.
