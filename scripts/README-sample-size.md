# sample size simulations

To perform sample size simulations, we use `sample_size_simulation.py`.  

There are a few configurations in the file to consider:

* `BUDGET_HOURS` - a grid of budgets (in GPU hours) to consider.  We recommend defining at most 6 at a time due to plotting library constraints.
* `EPISODES` - A grid of `n_episodes` to consider.  The script will automatically calculate the corresponding `n_test_examples` that aligns with a given `(budget_hours, n_episodes)`.
* `PROBS_OF_CORRECT_ANSWER` - A grid of true model accuracy values.  Should be between 0 and 1.
* `EPISODE_NOISE` - A hyperparameter representing amount of Gaussian noise we assume causes variance in episode-specific model accuracies.  Should be a float > 0.
* `NUM_REPS` - Number of total simulations for a given configuration.  Each simulation results in a bootstrap CI.  Based on simulations, we compute the CI coverage and width.


### Running

Using Python, run `python sample_size_simulation.py` which will output a CSV of simulation results (i.e. the CI coverage or width), one per `(acc, budget_hours, n_episode, n_test_examples)` configuration.

### Plotting

Using [RStudio](https://www.rstudio.com/), open the scripts `plotCICoverage.R` or `plotCIWidth.R`.  Make sure the path to the CSV output from previous Python step is correct.  Then run the entire script in RStudio to produce their respective plots (as seen in paper).

This likely first requires installation of R packages:
```
install.packages('ggplot2')
install.packages('dplyr')
install.packages('viridis')
remotes::install_github("slowkow/ggrepel")
```

These dependencies have the following licenses:

* dplyr - MIT - https://cran.r-project.org/web/packages/dplyr/index.html
* ggplot2 - MIT - https://cran.r-project.org/web/packages/ggplot2/index.html
* viridis - MIT - https://cran.r-project.org/web/packages/viridis/index.html
* ggrepel - GPL 3.0 - https://github.com/slowkow/ggrepel/blob/master/LICENSE




### `(budget_hours, n_episode, n_test_example)` relationship

The relationship between these variables is defined in function `get_test_examples_given_budget_hours`.  The constants are estimated by hand.  We work through these here:

1. For few-shot setting, the cost is `(Setup + Train + Test) * 12` where the `12` is the number of datasets involved in FLEET meta-testing.  Zero-shot setting cost looks similar, but without the `Train` cost.

2. We estimated in few-shot and zero-shot settings, `Setup = 1.5208 sec/episode` and `Test = 0.0673 sec/test instance`.  In few-shot setting, `Train = 94.1293 sec/episode`.

3. Let's do an example with 48 GPU hours as budget. Our total cost formula to run all of FLEET is: `48 hours = 172,800 sec = (1.5 sec * n_episode * 12 * 2) + (94 sec * n_episode * 12) + (0.067 sec * n_test_examples * n_episode * 12 * 2)`.

4. We solve for `n_test_examples = (172800 budget - 1166.05 * n_episodes) / (7.5 * n_episodes)`, which are the values we see in `get_test_examples_given_budget_hours`.

One can repeat this simulation but for different estimates of `Setup`, `Train`, and `Test` costs by solving this same equation and replacing the final values.


 