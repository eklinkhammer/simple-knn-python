# simple-knn-python
K-nearest neighbors written in python. It will use either MSL or Cuda as a backend to accelerate.

This was written because most research code for gaussian splatting related projects hard-codes as a submodule simple-knn, which
is a CUDA file. Ideally there would be a .metal equivilent for optimal speedup, but this works for now.
