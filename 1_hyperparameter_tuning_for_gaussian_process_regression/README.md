# Blog post: *Hyperparameter tuning via Maximum Likelihood estimation for Gaussian process regression*

Here, you find supplementary material accomapanying the blog post [*Hyperparameter tuning via Maximum Likelihood estimation for Gaussian process regression*](https://chrishoffmann.ai/post/hypparam_tuning_for_gp_regression), which was published on my personal website. 

Namely, we provide:
- the Jupyter notebook `notebook.ipynb` that the post is based on.
- source code for the python package ```gp_regression```—a simple object-oriented implementation
of Gaussian Process regression that is used throughout the post.

  The package provides functionality for:
  - hyperparameter tuning
  - sampling from the prior or posterior distribution
  - performing predictions for unknown outputs
  - as well as several methods for visualization purposes

  The package also comes with unittests and is well-documented via docstrings. <br>
  If you are interested in installing ```gp_regression```, please follow the detailed instructions below.


## Installation of the python package ```gp_regression```
STEP 0: I recommend to create and activate a virtual environment for Python 3. <br>
Then follow the steps below by running the highlighted commands from a terminal.

STEP 1: Prepare download
> create a new directory and change into this:  <br>
> `mkdir new_dir` <br>
> `cd new_dir`

> initialize an empty git repository and add the remote repository: <br>
> `git init` <br>
> `git remote add origin -f https://github.com/chris-hoffmann/blog_posts`

> this ensures that you only download the files for this particular blog post: <br>
> `echo '1_hyperparameter_tuning_for_gaussian_process_regression' >> .git/info/sparse-checkout`

STEP 2: Download files and build the ```gp_regression``` package locally 
> `git pull origin master` 

> install the required libraries (Numpy, Scipy, Matplotlib): <br>
> `pip install -r requirements.txt`

> build the local package <br>
> `pip install -e .`

STEP 3: Verify the installation
> run the unittests: <br>
> ```python -m unittest test.py```
