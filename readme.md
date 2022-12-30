# Databoost

This github repo is the source code for the blog post
[Boosting as a scheme for transfer learning](https://www.chrisstucchio.com/blog/2022/boosting_for_knowledge_transfer.html).

You should not attempt to understand what is here before reading that post.

That done? Great. Here's what's in this repo:

1. A few datasets (e.g. santander) used for testing.
2. A notebook [multi_scenario.ipynb](multi_scenario.ipynb) which was used to actually run the simulations and generate the graphs/tables in the blog post.
3. Source code in [databoost/][databoost/].

The source code has two files, [utils.py](databoost/utils.py) which just has utilities for summing ML models, and [scenario.py](databoost/scenario.py) which has loaders for the different datasets used in testing.

The actual boosting code is in the [notebook](multi_scenario.ipynb).
