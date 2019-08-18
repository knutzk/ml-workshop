# ML workshop: hands-on jupyter notebooks

This repository is a collection of jupyter notebooks to teach you the basics of machine learning. They provide easy access to out-of-the-box code examples, and they are meant to give insights into the most basic machine-learning algorithms.

If you are unfamiliar with installing python and python packages on your computer, you might want to check out the [installation instructions](INSTALLATION.md). In these instructions, we give a step-by-step guidance how to install the _anaconda_ package management system on your computer.


## Getting started

Anaconda is our recommended way to install and maintain python on your system. If you have a python installation already and do not need the guidance through anaconda, there is a brief expert section further below. For everyone else: if you have followed the above installation instructions for anaconda, the following section will guide you through setting up these notebooks with anaconda on your computer step by step.

### Setup with anaconda (recommended)

Still to be written ...

### Setup without anaconda (experts)

This instruction is meant for people who are already familiar with python package managers, possibly use virtual environments etc. In that case, we assume you are familiar with installing, updating and maintaining python packages, and that your environment is configured correctly. Make sure you have the [required packages](ml-environment.yml) installed in a recent version. These include:
* numpy, scipy
* matplotlib, seaborn
* pandas
* sklearn
* tensorflow
* jupyter

You can then get started by either clicking on the individual notebooks to view them, or use the slightly more comfortable [notebook viewer](https://nbviewer.jupyter.org/github/knutzk/ml-workshop/). Both of these options will only give you a rendered version of the notebooks. To work on the notebooks yourself and to execute the code cells, make a clone of the repository:
```
git clone https://github.com/knutzk/ml-workshop.git
```
Then, start a jupyter notebook kernel by opening a shell and typing:
```
jupyter notebook
```
This opens a browser tab. Navigate to the `ml-workshop` folder and get started with the notebooks.
