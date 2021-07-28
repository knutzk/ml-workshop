# ML workshop: hands-on jupyter notebooks

[![Build Status](https://travis-ci.org/knutzk/ml-workshop.svg?branch=master)](https://travis-ci.org/knutzk/ml-workshop)

This repository is a collection of jupyter notebooks to teach you the basics of
machine learning. They provide easy access to out-of-the-box code examples, and
they are meant to give insights into the most basic machine-learning algorithms.

Almost all of the code examples are taken from or inspired by Ref. [2], a book
we can only recommend to study for hands-on examples of machine learning. All
code examples of the book are available in A. Geron's [github
repository](https://github.com/ageron/handson-ml2). To acknowledge his work,
this repository is made available under the same licensing terms using the
Apache License 2.0. Please make sure to follow the respective conditions of this
license when distributing or using this repository.

The book exists in two versions:

[1] A. Geron, _Hands-On Machine Learning with Scikit-Learn and TensorFlow_, O'Reilly 2017, ISBN: 978-1491962299
[2] A. Geron, _Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow_, 2nd edition, O'Reilly 2019, ISBN: 978-1492032649


## I only want to browse the notebooks!

Even without setting up anything on your own computer, you can browse the
content of the notebooks by just clicking on them in the list of files above. To
see what the executed notebooks look like, you can also navigate to the
`md_output` subdirectory and open any of the notebooks there. Or you can use the
[notebook viewer](https://nbviewer.jupyter.org/github/knutzk/ml-workshop/).


## How to get started with this repository

Our recommended way to install python on your private computers is through the
anaconda suite and/or through virtual environments. If these terms don't mean
anything to you, or if you are not sure what to do, please check out the
detailed [anaconda installation instructions](INSTALLATION.md). There we give a
step-by-step guidance how to install the _anaconda_ package management system on
your computer.

The following steps assume that you have anaconda installed on your computer.

> N.B. If you have a python installation already, you can of course skip the
> following steps and follow your own best practice for installing the necessary
> packages. Please check out the expert section below for some hints.


#### 1) Obtain a copy of this repository

The first step is to download a copy of this repository. To do so, click on the
green download button at the top of this page, and then on "Download ZIP", which
downloads a zip file with the contents of this repository. Then unpack the zip
file (for most operating systems: double click). An alternative – if you're
familiar with the command line – is to clone the repository with the `git clone`
command:

```
git clone https://github.com/knutzk/ml-workshop.git
```

Then continue with one of the two following steps.


#### 2a) Setup with Anaconda Navigator (Windows and macOS, recommended)

If you have followed the installation instructions for anaconda linked at the
top, the Anaconda Navigator application should be installed on your computer (on
Windows and macOS). Open the application, either from the start menu (Windows)
or the dashboard (macOS). Then, navigate to "Environments" on the left-hand
side. Click on the Import button at the bottom of the list of environments and
choose the [ml-environment.yml](ml-environment.yml) file, which is inside your
downloaded copy of this repository. The name field should be filled
automatically, otherwise choose something descriptive like `ml-workshop`.
Confirm the import. This creates a custom environment called `ml-workshop` to
contain all packages needed for the hands-on tutorials. The setup might take a
moment, because some the packages have to be downloaded and installed.

After the environment `ml-workshop` is created, it shows up in the list of
environments. Activate the environment by clicking on it (it should be marked
with a green marker). Click on the triangular "play" button next of the
`ml-environment` environment and choose "Open with Jupyter Notebook". 


#### 2b) Setup with conda in the command line (Linux, recommended)

After installing anaconda on Linux (and also macOS), you can also use the
`conda` command in the shell to set up your machine-learning environment. In
case the `conda` command cannot be found on your system, make sure to source the
activation script and then call the `conda init` command:

```
source <path to conda>/bin/activate
conda init
```

Afterwards, the custom "ml" environment can be set up. Navigate to the copy of
this repository that you obtained in the previous step with the `cd` command.
Then, create a new conda environment by providing the
[ml-environment.yml](ml-environment.yml) file and activate the `ml-environment`
environment. The commands are:

```
cd <path to downloaded copy>/ml-workshop/
conda env create -f ./ml-environment.yml
conda activate ml
```

Now, the environment is set up and activated! To verify the installation, start a
Jupyter Notebook server:

```
jupyter notebook
```


#### 3) Exploring the first notebook

If you have followed (2a) or (2b) above, the last command opens your default web
browser with a Jupyter webpage. The Jupyter logo should be at the top of the
page, and the page itself should display your user directory. Now it's time to
explore the first notebook and to verify the setup!

Navigate to the folder, to which you downloaded the copy of this repository,
enter the folder, then open the test file
[01_test_notebook.ipynb](01_test_notebook.ipynb). Follow the instructions inside
the notebook to verify your python installation and the anaconda environment
setup.

After closing the browser tab, your Jupyter instance closes automatically if you
used the Anaconda Navigator in step (2a). If you opened Jupyter from the command
line in (2b), you can shut down the Jupyter notebook server with CTRL+C. The
conda environment can be deactivated with:

```
conda deactivate
```


## Setup without anaconda (experts)

The above instructions are meant for people not familiar with setups of python
versions and/or virtual environments. If you are familiar with both, of course
you can stick to your best practices to set up an environment for this
repository. Inter alia, this repository comes with a docker file that builds an
image with all packages required for the execution of the notebooks. The image
can either be downloaded as `knutzk/ml-workshop:latest` from the [docker
hub](https://hub.docker.com/r/knutzk/ml-workshop) or it can be built from this
repository directly:

```
cd <path to downloaded copy>/ml-workshop/
docker build -t ml-workshop-image .
```

Then, a jupyter notebook server can be started as a docker container with the
following command:

```
docker run --rm -u $(id -u):$(id -g) -p 8888:8888 -v $PWD:/data ml-workshop-image
```

Transmitting the user and group ID is necessary to avoid running the docker
container as root (which is discouraged for jupyter notebooks). It also binds
port 8888 to that of the localhost and mounts the working directory to the
`/data` endpoint in the notebook. The notebook server can then be reached by
opening http://localhost:8888/ in a browser of choice. The webpage will require
a token that can be found in the command line output when starting the docker
container.

As an alternative, you can also use virtual environments (e.g.
`pyenv-virtualenv` or `conda`) to set up the necessary packages for this
repository. Make sure you have the [required packages](ml-environment.yml)
installed in a recent version. These include:

* sklearn 0.24.*
* tensorflow 2.4.*

After installing all packages listed in the file, obtain a copy of this
repository through your preferred method. Start a Jupyter Notbeook server,
navigate to the location of your repository copy, and open the Jupyter Notebook
files to get started. Please make sure to go through the
[01_test_notebook.ipynb](01_test_notebook.ipynb) to verify your installation.
