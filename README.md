# Automatic Tree Cadastre


## Description
This project was developed with in the context of the master thesis "Automated Generation of a Tree Cadastre From Point Clouds in Different Urban Environments".
The program automatically creates a tree cadastre from a point cloud. The tree cadastre resulting from the automated workflow is subsequently available as a tree parameter list in a CSV file and also as a CityGML file. In the CityGML file, the trees also have a geometric representation in the form of a cylinder-sphere model based on the respective tree parameters.


## Getting started
- you will need to install:
  - Python 3.10 
  - FME 2022.2.5 ([get FME](https://engage.safe.com/support/downloads/#official)) 
  - VS Code 
  - Microsoft Visual C++ 14.0 or greater. Get it with [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
- you will need pointcloud data
- open your favourite terminal
- install pipenv: `pip install pipenv`
- clone this git repository: `git clone --recursive https://gitlab.lrz.de/000000000149B597/automatic-tree-cadastre.git`
- open VS Code in the automatic-tree-cadastre folder
- open a new terminal in VS Code
- install all needed packages from the Pipfile: `pipenv install`
- set all parameters and file paths according to your data in main.py (or main.ipynb)
- run main.py (or main.ipynb) and enjoy the results in the output folder
  - either run main.py in the terminal: `pipenv run python main.py`
  - or switch to the automatic-tree-cadastre environment in VS Code and run main.py or main.ipynb from there


## Support
For help or further information: s.zagst@tum.de


