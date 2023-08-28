# Automatic Tree Cadastre


## Description
This project was developed as part of the master thesis "Automated Generation of a Tree Cadastre From Point Clouds in Different Urban Environments". The full-text of the thesis was published via MediaTUM and can be found [here](https://mediatum.ub.tum.de/1713266). If you use this code or my master thesis please cite this document:
```
@mastersthesis{
	author = {Zagst, Sabine},
	title = {Automatisierte Generierung eines Baumkatasters aus Punktwolken in unterschiedlichen urbanen Umgebungen},
	year = {2023},
	school = {Technische Universität München},
	adress = {München},
	language = {de}
}
```

The program automatically creates a tree cadastre from a point cloud. The tree cadastre resulting from the automated workflow is subsequently available as a tree parameter list in a CSV file and also as a CityGML file. In the CityGML file, the trees also have a geometric representation in the form of a cylinder-sphere model based on the respective tree parameters. For more details and example results please have a look at the [mastersthesis](https://mediatum.ub.tum.de/1713266).


## Getting started
- you will need to install:
  - Python 3.10 (3.11 will not work because Open3D needs version 3.10)
  - FME 2022.2.5 ([get FME](https://engage.safe.com/support/downloads/#official)) with a license
  - VS Code 
  - Microsoft Visual C++ 14.0 or greater. Get it with [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
- you will need pointcloud data
- open your favourite terminal
- install pipenv: `pip install pipenv`
- clone this git repository: 
`git clone --recursive https://github.com/SabineZa/Automatic_Tree_Cadastre.git`
- open VS Code in the automatic-tree-cadastre folder
- open a new terminal in VS Code
- install all needed packages from the Pipfile: `pipenv install`
- set all parameters and file paths according to your data in main.py (or main.ipynb)
- run main.py (or main.ipynb) and enjoy the results in the output folder
  - either run main.py in the terminal: `pipenv run python main.py`
  - or switch to the automatic-tree-cadastre environment in VS Code and run main.py or main.ipynb from there

If you don't have a FME-license you can nevertheless use the code and create a tree cadastre out of your pointclouds. The only drawback will be that you can't create a geometric representation of the trees in CityGML. For that you just have to disable the correspnding lines of code.

## Support
For help or further information feel free to contact me: s.zagst@tum.de


