# Anaconda installation instructions

This page provides a short guide for installing python on your computer using
the _anaconda_ package management system. _anaconda_ is a distribution for the
python programming language, that not only installs python on your system, but
also provides functionalities for downloading, installing and maintaining python
packages.

_anaconda_ can be obtained from the project's website:
[https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/).
Downloads exist for Windows, macOS as well as Linux – please choose the
appropriate installer for your operating system. Graphical installers are
provided for Windows and macOS, that is, you just open the downloaded `.exe` or
`.pkg` file and follow the on-screen instructions for installation. For Linux,
there is a command-line-based installer, which is to be downloaded and then
executed in the shell.

We strongly recommend installing _anaconda_ for your personal user account only.
That is, on macOS and Linux, the installation will be done in
`home/<user>/anaconda3`. This way, the installation will not affect any
system-wide settings, and is generally a little more robust. If you need more
help with downloading and installing _anaconda_ on your system, the documentation
pages provide [installation
instructions](https://docs.anaconda.com/anaconda/install/), too. Here are the
direct links for Windows, macOS and Linux:

* [Installing on Windows](https://docs.anaconda.com/anaconda/install/windows/)
* [Installing on macOS](https://docs.anaconda.com/anaconda/install/mac-os/)
* [Installing on Linux](https://docs.anaconda.com/anaconda/install/linux/)

For graphical installers (on Windows and macOS) you can verify the installation
by opening the newly installed application "Anaconda Navigator". If this opens
the Anaconda Navigator window, you're done with the installation!

In case you used a command-line-based installer (Linux, but also macOS), you
will also be prompted: "Do you wish the installer to initialize Anaconda3 by
running conda init?" We recommend entering "yes" to set up _anaconda_ for
command-line use correctly. Close and reopen your shell, then check whether the
conda command exists and returns something, e.g. by typing `conda list`. This
should return a (more or less) long list of installed python packages. In the
next step, type `python`, which opens the python interpreter in the shell. If
_anaconda_ is installed and working, the version information it displays when it
starts up will include “Anaconda”. To exit the Python shell, enter the command
`quit()`. If the `conda` command cannot be found, check out steps (7) ff. in the
Linux installation instructions (for both macOS and Linux).
