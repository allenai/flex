# FLEX

FLEX is a benchmark and framework for unified, rigorous few-shot NLP evaluation.
FLEX enables:
- First-class NLP support
- Support for meta-training
- Reproducible fewshot evaluations
- Extensible benchmark creation (benchmarks defined using [HuggingFace Datasets](https://huggingface.co/datasets))
- Advanced sampling functions for creating episodes with class imbalance, etc.

For more context, see our paper (appearing on Arxiv shortly).

**Leaderboards (coming soon)**

## Installation

- Clone the repository `git clone git@github.com:allenai/flex.git`
- Create a Python 3 environment (3.7 or greater), eg using `conda create --name flex python=3.9`
- Activate the environment `conda activate flex`
- Install the package locally with `pip install -e .`

## Data preparation

Creating the data for the flex challenge for the first time takes about 10 minutes (using a recent Macbook Pro on a broadband connection) and requires 3GB of disk space.
You can initiate this process by running
```bash
python -c "import fewshot; fewshot.make_challenge('flex');"
```

You can control the location of the cached data by setting the environment variable `HF_DATASETS_CACHE`.
If you have not set this variable, the location should default to `~/.cache/huggingface/datasets/`.
See the [HuggingFace docs](https://huggingface.co/docs/datasets/installation.html#caching-datasets-and-metrics) for more details.

## Model evaluation

"Challenges" are datasets of sampled tasks for evaluation. They are defined in `fewshot/challenges/__init__.py`.

To evaluate a model on challenge `flex` (our first challenge), you should write a program that produces
a `predictions.json`, for example:
```python
#!/usr/bin/env python3
import random
from typing import Iterable, Dict, Any, Sequence
import fewshot


class YourModel(fewshot.Model):
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
    evaluator = fewshot.make_challenge("flex")
    model = YourModel()
    evaluator.save_model_predictions(model=model, save_path='/path/to/predictions.json')
```

Warning: Calling `fewshot.make_challenge("flex")` above requires some time to prepare all the necessary data (see "Data preparation" section).

Running the above script produces `/path/to/predictions.json` with contents formatted as:
```
{
    "[QUESTION_ID]": {
        "label": "[CLASS_LABEL]",  # Currently an integer converted to a string
        "score": float  # Only used for ranking tasks
    },
    ...
}
```
Each `[QUESTION_ID]` is an ID for a test example in a few-shot problem.

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

For the meta-training protocol (e.g., the [FLEX-META leaderboard](https://leaderboard.allenai.org/flex_meta)), challenges come with a set of related training and validation data.
This data is most easily accessible in one of two formats:

1. **Iterable from sampled episodes.** `fewshot.get_challenge_spec('flex').get_sampler(split='[SPLIT]')` returns an iterable that samples datasets and episodes from meta-training or meta-validation datasets, via `[SPLIT]='train'` or `[SPLIT]='val'`, respectively. The sampler defaults to the `fewshot.samplers.Sample2WayMax8ShotCfg` sampler configuration (for the `fewshot.samplers.sample.Sampler` class), but can be reconfigured.

2. **Raw dataset stores.** This option is for directly accessing the raw data. `fewshot.get_challenge_spec('flex').get_stores(split='[SPLIT'])` returns a mapping from dataset names to `fewshot.datasets.store.Store` instances. Each `Store` instance has a `Store.store` attribute containing a raw [HuggingFace Dataset](https://huggingface.co/docs/datasets/exploring.html) instance. The `Store` instance has a `Store.label` attribute with the Dataset object key for accessing the target label (e.g., via `Store.store[Store.label]`) and the FLEX-formatted text available at the `flex.txt` key (e.g., via `Store.store['flex.txt']`).

Two examples of these respective approaches are available at:
1. The [UniFew model repository](https://github.com/allenai/unifew). For more details on Unifew, see also the FLEX Arxiv paper.
2. The `baselines/bao/` directory, for training and evaluating the approach described in the following paper:
> Yujia Bao*, Menghua Wu*, Shiyu Chang, and Regina Barzilay. Few-shot Text Classification with Distributional Signatures. In International Conference on Learning Representations 2020

## Benchmark construction and optimization

To add a new benchmark (challenge) named `[NEW_CHALLENGE]`, you must edit `fewshot/challenges/__init__.py` or otherwise add it to the registry.
The above usage instructions would change to substitute `[NEW_CHALLENGE]` in place of `flex` when calling `fewshot.get_challenge_spec('[NEW_CHALLENGE]')` and `fewshot.make_challenge('[NEW_CHALLENGE]')`.

For an example of how to optimize the sample size of the challenge, see `scripts/README-sample-size.md`.
