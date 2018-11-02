# Evaluating probabilistic saliency models on the MIT Saliency Benchmark

This repository contains code to evaluate probabilistic saliency models on the [MIT Saliency Benchmark](https://saliency.mit.edu) as proposed in [Kümmerer et al, Saliency Benchmarking Made Easy: Separating Models, Maps and Metrics](http://openaccess.thecvf.com/content_ECCV_2018/html/Matthias_Kummerer_Saliency_Benchmarking_Made_ECCV_2018_paper.html) by computing metric specific saliency maps from the predicted fixation densities.

## Inconsistencies with the MIT300 evaluation code

There are a few inconsistencies between the original evaluation code and how the saliency maps are computed here:

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
