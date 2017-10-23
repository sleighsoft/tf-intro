# Setup
1. Install miniconda from https://conda.io/miniconda.html
   (for bash installer just run) `bash <FILENAME>`
2. In `project-root` run `conda env create -f tf-intro/environment.yml`
3. Run `activate tf-intro` in Windows (best is cmd) or `source activate tf-intro` in Linux/MacOS

# Development Tips
- When using SublimeText Change `python_interpreter` and `extra_paths` in `.anaconda` to resemble the path where the conda environment has been installed to (probably no \\ for linux user), to allow anaconda to use the documentation.

# Order Of Examples
1. `constants.py`
2. `variables.py`
3. `graphs_and_sessions.py`
4. `saver_and_tensorboard.py`

# Useful Readings
- [Tensorflow Best Practices](https://github.com/aicodes/tf-bestpractice)
- [Tensorflow Graphs & Sessions Explained](https://www.tensorflow.org/programmers_guide/graphs)
- [Tensorflow Variable Initialization](https://www.tensorflow.org/programmers_guide/variables#initializing_variables)
- [Tensorflow Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)
- [Tensorflow Saving & Restoring](https://www.tensorflow.org/programmers_guide/saved_model)

# Useful Functions For Debugging
- tf.Print()
- tf.InteractiveSession()