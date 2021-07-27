# ML workshop: hands-on jupyter notebooks

[![Build Status](https://travis-ci.org/knutzk/ml-workshop.svg?branch=master)](https://travis-ci.org/knutzk/ml-workshop)

This repository is a collection of jupyter notebooks to teach you the basics of
machine learning. They provide easy access to out-of-the-box code examples, and
they are meant to give insights into the most basic machine-learning algorithms.

Almost all of the code examples are taken from or inspired by Ref. [1], a book
we can only recommend to study for hands-on examples of machine learning. All
code examples of the book are available in A. Geron's [github
repository](https://github.com/ageron/handson-ml2). To acknowledge his work,
this repository is made available under the same licensing terms using the
Apache License 2.0. Please make sure to follow the respective conditions of this
license when distributing or using this repository.

[1] A. Geron, _Hands-On Machine Learning with Scikit-Learn and TensorFlow_, O'Reilly 2017, ISBN: 978-1491962299
[2] A. Geron, _Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow_, 2nd edition, O'Reilly 2019, ISBN: 978-1492032649


## I only want to browse the notebooks!

Even without setup, the content of the Jupyter Notebooks can be opened directly
by just clicking on them in the list of files above. Or you can use the slightly
more comfortable (and faster)[notebook
viewer](https://nbviewer.jupyter.org/github/knutzk/ml-workshop/).


## How to get started with this repository

Our recommended way to install python on your private computers is through the
anaconda suite and/or through virtual environments. If these terms don't mean
anything to you, or if you are not sure what to do, please check out the
detailed [installation instructions](INSTALLATION.md). In these instructions, we
give a step-by-step guidance how to install the _anaconda_ package management
system on your computer. 

If you have a python installation already, you can of course follow your own
best practice for installing the necessary packages. There is also a brief
expert section further below that you can check for more details.



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

This opens a new tab in your default web browser, with the Jupyter logo at the
top of the page, and a display of your user directory. Navigate to the folder,
to which you downloaded the copy of this repository, enter the folder, then open
the test file [01_test_notebook.ipynb](01_test_notebook.ipynb). Go through the
notebook to verify your python installation and the anaconda environment setup.


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

This opens a new tab in your default web browser, with the Jupyter logo at the
top of the page, and a display of your user directory. Navigate to the folder,
to which you downloaded the copy of this repository, enter the folder, then open
the test file [01_test_notebook.ipynb](01_test_notebook.ipynb). Go through the
notebook to verify your python installation and the conda environemnt setup.
After closing the browser tab, you can shut down the Jupyter notebook server
with CTRL+C. The conda environment can be deactivated with:

```
conda deactivate
```


#### 2c) Setup with docker (experts)

This introduction is meant for people who are familiar with using docker and
executing codes in docker images/containers. Docker images provide
self-contained setups of libraries and packages and allow executing code on
different systems under the same conditions. This repository comes with a docker
file that builds an image with all packages required for the execution of the
notebooks. The image can either be downloaded as `knutzk/ml-workshop:latest`
from the [docker hub](https://hub.docker.com/r/knutzk/ml-workshop) or it can be
built from this repository directly:

```
cd <path to downloaded copy>/ml-workshop/
docker build -t ml-workshop-image .
```

Then, a jupyter notebook server can be started as a docker container with the
following command:

```
docker run --rm -u $(id -u):$(id -g) -p 8888:8888 -v $PWD:/data ml-workshop-image
```

This starts a docker container with the correct user ID (via `id -u`, running as
root is not allowed for jupyter notebooks), binds port 8888 to that of the
localhost and mounts the working directory to the `/data` endpoint in the
notebook. The notebook server can then be reached by opening
http://localhost:8888/ in a browser of choice. The webpage will require a token
that can be found in the command line output when starting the docker container.


#### 2d) Setup without anaconda (experts)

This instruction is meant for people who are already familiar with python
package managers, possibly use virtual environments etc. In that case, we assume
you are familiar with installing, updating and maintaining python packages, and
that your environment is configured correctly. Make sure you have the [required
packages](ml-environment.yml) installed in a recent version. These include:

* sklearn 0.21.*
* tensorflow 1.14.*
* jupyter 1.0.*

After installing all packages listed in the file, obtain a copy of this
repository as described above. Start a Jupyter Notbeook server, navigate to the
location of your repository copy, and open the Jupyter Notebook files to get
started. Please make sure to go through the
[01_test_notebook.ipynb](01_test_notebook.ipynb) to verify your installation.
