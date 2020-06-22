# ML workshop: hands-on jupyter notebooks

[![Build Status](https://travis-ci.org/knutzk/ml-workshop.svg?branch=master)](https://travis-ci.org/knutzk/ml-workshop)

This repository is a collection of jupyter notebooks to teach you the basics of machine learning. They provide easy access to out-of-the-box code examples, and they are meant to give insights into the most basic machine-learning algorithms.

Many of the code examples are taken from or inspired by Ref. [1], a book I can only recommend to study for hands-on examples of machine learning. All code examples of the book are available in the A. Geron's [github repository](https://github.com/ageron/handson-ml). To acknowledge his work, this repository is made available under the same licensing terms using the Apache License 2.0. Please make sure to follow the respective conditions of this license when distributing or using this repository.

[1] A. Geron, _Hands-On Machine Learning with Scikit-Learn and TensorFlow_, O'Reilly 2017, ISBN: 978-1491962299

If you are unfamiliar with installing python and python packages on your computer, please check out the [installation instructions](INSTALLATION.md). In these instructions, we give a step-by-step guidance how to install the _anaconda_ package management system on your computer. Anaconda is our recommended way to install and maintain python on your system. If you have a python installation already and do not need the guidance through anaconda, there is a brief expert section further below. For everyone else: please go through the following steps to set up your machine-learning environment with anaconda.

Even without setup, the content of the Jupyter Notebooks can be opened directly by just clicking on them in the list of files above. Or use the slightly more comfortable [notebook viewer](https://nbviewer.jupyter.org/github/knutzk/ml-workshop/) (which is also faster).


### Obtaining a copy of this repository

The first step is to download a copy of this repository. This can be done by using the green download button at the top of this page and clicking on "Download ZIP", which will give you a zip file with the contents of this repository. Unpack the zip file to obtain the contents. An alternative – if you want to use the command line – is to clone the repository with the `git clone` command:
```
git clone https://github.com/knutzk/ml-workshop.git
```
Then, continue with one of the two following steps.


### Setup with Anaconda Navigator (Windows and macOS, recommended)

If you have followed the installation instructions for anaconda linked at the top, you should have the Anaconda Navigator application (on Windows and macOS). Open the application, either from the start menu (Windows) or the dashboard (macOS). Then, navigate to "Environments" on the left-hand side. Click on the Import button at the bottom of the list of environments and choose the [ml-environment.yml](ml-environment.yml) file, which is part of your downloaded copy of this repository. You don't have to type the name, the field should be filled automatically after choosing the import file. Confirm the import. This will create a custom environment called "ml" to contain all packages needed for the hands-on tutorials. The setup might take a moment, because some the packages have to be downloaded and installed.

After the environment "ml" is created, it shows up in the list of environments. Activate the environment by clicking on it (it should be marked with a green marker). Click on the triangular "play" button next of the "ml" environment and choose "Open with Jupyter Notebook". This opens a new tab in your default web browser, with the Jupyter logo at the top of the page, and a display of your user directory. Navigate to the folder, to which you downloaded the copy of this repository, enter the folder, then open the test file [01_test_notebook.ipynb](01_test_notebook.ipynb). Go through the notebook to verify your python installation and the anaconda environemnt setup.


### Setup with conda in the command line (Linux, recommended)

After installing anaconda on Linux (and also macOS), you can also use the `conda` command in the shell to set up your machine-learning environment. In case the `conda` command cannot be found on your system, make sure to source the activation script and then call the `conda init` command:
```
source <path to conda>/bin/activate
conda init
```
Afterwards, the custom "ml" environment can be set up. Navigate to the copy of this repository that you obtained in the previous step with the `cd` command. Then, create a new conda environment by providing the [ml-environment.yml](ml-environment.yml) file and activate the "ml" environment. The commands are:
```
cd <path to downloaded copy>/ml-workshop/
conda env create -f ./ml-environment.yml
conda activate ml
```
Now, the environemt is set up and activated! To verify the installation, start a Jupyter Notebook server:
```
jupyter notebook
```
This opens a new tab in your default web browser, with the Jupyter logo at the top of the page, and a display of your user directory. Navigate to the folder, to which you downloaded the copy of this repository, enter the folder, then open the test file [01_test_notebook.ipynb](01_test_notebook.ipynb). Go through the notebook to verify your python installation and the conda environemnt setup. After closing the browser tab, you can shutdown the Jupyter notebook server with CTRL+C. The conda environment can be deactivated with:
```
conda deactivate
```


### Setup without anaconda (experts)

This instruction is meant for people who are already familiar with python package managers, possibly use virtual environments etc. In that case, we assume you are familiar with installing, updating and maintaining python packages, and that your environment is configured correctly. Make sure you have the [required packages](ml-environment.yml) installed in a recent version. These include:
* sklearn 0.21.*
* tensorflow 1.14.*
* jupyter 1.0.*

After installing all packages listed in the file, obtain a copy of this repository as described above. Start a Jupyter Notbeook server, navigate to the location of your repository copy, and open the Jupyter Notebook files to get started. Please make sure to go through the [01_test_notebook.ipynb](01_test_notebook.ipynb) to verify your installation.
