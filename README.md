# Evaluating probabilistic and classical saliency models on the MIT/Tuebingen Saliency Benchmark

This repository contains code to evaluate probabilistic and classical saliency models on the [MIT Saliency Benchmark](https://saliency.mit.edu).
Probabilistic saliency models are evaluated as proposed in [Kümmerer et al, Saliency Benchmarking Made Easy: Separating Models, Maps and Metrics](http://openaccess.thecvf.com/content_ECCV_2018/html/Matthias_Kummerer_Saliency_Benchmarking_Made_ECCV_2018_paper.html) by computing metric specific saliency maps from the predicted fixation densities.

## Inconsistencies with the MIT300 evaluation code

There are a few inconsistencies between the [original evaluation code of the MIT Saliency Benchmark](https://github.com/cvzoya/saliency/tree/master/code_forMetrics) and how the saliency maps are computed and evaluated here:

* the original MIT Saliency Benchmark uses 8 cycles/image for
  computing gaussian convolutions and does so via the Fourier domain,
  i.e. with zero-padding the image to be square and then cyclic extension.
  according to the paper, 8 cycles/image corresponds to 1 dva or about 35pixel
  and therefore we use a Gaussian convolution with 35 pixels and nearest
  padding (which shouldn't make a lot of a difference due to the sparse
  fixations)
* We don't have correct saliency maps for the Earth Mover's Distance yet since
  for this metric there is no analytic solution of the optimization problem which
  saliency map has the highest expected metric performance under the predicted fixation
  distribution.
* In the original MIT Saliency Benchmark, multiple fixations at the same image location
  are treated as only one fixation in that location. This happens quite a few times (380
  times in MIT1003 and more often on MIT300 due to the higher number of subjects per image)
  and gives rise to artefacts especially in the case of many fixations. Here, each fixation
  contributes equally to the performance on a specific image.
* While the AUC_Judd metric of the original MIT Saliency Benchmark uses all pixels that
  are not fixated as nonfixations, we use all pixels as nonfixations. We consider this 
  more principled since it behaves better in the limit of many fixations.
* Originally, the AUC_Judd metric added some random noise on the saliency map to make sure
  there are no pixels with the same saliency value, since the AUC implementation could not
  handle that case. Our implementation can handle this case (including a constant saliency map
  that will result in an AUC score of 0.5), and therefore the noise is not needed anymore.
* In the MIT Saliency Benchmark, the shuffled AUC metric:
    * took the fixations of 10 other images
    * removed doublicate fixation locations among them
    * 100 times choose a random subset of those that is as big as the number of actual
      fixations and computed the AUC score between the fixations and those nonfixations
    * averaged the scores
  Instead, we just take all fixation locations of all other images as nonfixations.
  As for the normal AUC, fixations and nonfixations can have repeated locations, which
  here is even more important than for the normal AUC due to the higher fixation density
  in the image center.
* So far, the MIT Saliency Benchmark accepted saliency maps of arbitrary sizes and rescaled
  them to the size of the input images. However, the resaling operation has a lot of
  parameters that can be choosen (nearest neighbour upsampling, kubic, etc) and
  that might affect the metric performance especially of metrics that are very sensitive
  in some scale regions (e.g. KLDiv). This choice should be made by the submitting
  researcher. Therefore, new submissions have to have saliency maps of the same size as
  the evaluated images
* In the MIT Saliency Benchmark, the similarity metric rescaled each saliency map to range
  from 0 to 1 before dividing by the sum to make it a distribution. Since there is no
  reason why at least on pixel should have a probability of zero, this is not done anymore.
  Instead the original saliency map is directly divided by its sum.
