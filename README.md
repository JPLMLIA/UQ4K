# UQ4k: Uncertaininty Quantification of the 4th Kind
This package accompines the [Uncertainty Quantification of the 4th kind; optimal posterior accuracy-uncertainty tradeoff with the minimum enclosing ball
](https://arxiv.org/abs/2108.10517) paper.

## Paper's Abstract
There are essentially three kinds of approaches to Uncertainty Quantification (UQ): (A) robust optimization, (B) Bayesian, (C) decision theory. Although (A) is robust, it is unfavorable with respect to accuracy and data assimilation. (B) requires a prior, it is generally brittle and posterior estimations can be slow. Although (C) leads to the identification of an optimal prior, its approximation suffers from the curse of dimensionality and the notion of risk is one that is averaged with respect to the distribution of the data. We introduce a 4th kind which is a hybrid between (A), (B), (C), and hypothesis testing. It can be summarized as, after observing a sample x, (1) defining a likelihood region through the relative likelihood and (2) playing a minmax game in that region to define optimal estimators and their risk. The resulting method has several desirable properties (a) an optimal prior is identified after measuring the data, and the notion of risk is a posterior one, (b) the determination of the optimal estimate and its risk can be reduced to computing the minimum enclosing ball of the image of the likelihood region under the quantity of interest map (which is fast and not subject to the curse of dimensionality). The method is characterized by a parameter in [0,1] acting as an assumed lower bound on the rarity of the observed data (the relative likelihood). When that parameter is near 1, the method produces a posterior distribution concentrated around a maximum likelihood estimate with tight but low confidence UQ estimates. When that parameter is near 0, the method produces a maximal risk posterior distribution with high confidence UQ estimates. In addition to navigating the accuracy-uncertainty tradeoff, the proposed method addresses the brittleness of Bayesian inference by navigating the robustness-accuracy tradeoff associated with data assimilation.

## Installation
The package can be simply installed by:

```
$ pip install uq4k
```

## Usage

> To fully run the provided examples, you'll need to have `matplotlib` in your environment

There are two versions of `uq4k` that can be used: gradient-based version and non-gradient version. For the gradient-based, consult the [gradient-based quadratic model example](examples/demo_quadratic_model_gradient.ipynb) for the usage. For the non-gradient version, check:

- [Example on predator-prey model (Lotka-Volterra)](examples/demo_pred_prey_model.ipynb)
- [Example on a quadratic model with no gradients.](examples/demo_quadratic_model_blackbox.ipynb)

_Detailed instruction on applying `uq4k` for your custom models will be provided soon_
