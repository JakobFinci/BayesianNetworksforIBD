<p align="center">
  <img src="https://www.ibdrelief.com/uploads/files/ibd1_gallery_image_924_131.jpg"/>
</p>
<h1 align="center">IBD Biomarker Network Analysis</h1>

### A computational framework for inferring conditional dependency networks between clinical biomarkers and tissue inflammation in pediatric inflammatory bowel disease.

## About the Project
This repository contains code and analysis tools for studying the network structure of clinical biomarkers and biopsy-derived inflammation measures in pediatric IBD. The project applies probabilistic graphical models—including Hill-Climbing Bayesian Networks and the PC algorithm—to infer conditional dependencies between biomarkers, inflammation sites, and diagnostic variables.

The pipeline includes tools for dataset preprocessing (MissForest imputation, handling of left-censored laboratory values), network inference, graph centrality analysis, and visualization. The goal is to explore how systemic biomarkers connect to localized tissue inflammation and to identify which markers function as central bridges within the inferred clinical network.

<p align="center">
  <img src="https://github.com/JakobFinci/BayesianNetworksforIBD/blob/main/images/network.jpg"/>
</p>
<p align="center">
    <em>Basic HillClimb Network for Full Cohort: Crohn's and UC</em>
</p>

## Installation
This project was developed for conda-based python environments. As such, we do not guarantee that all dependencies are UNIX tested, nor that the core source code is compatible with the rendering interfacing on that OS.

## Setup
While this project is intended for use on a local setup, it may still be run on a smaller computer, such as a Raspberry Pi, with limited functionality.

For best results, we suggest having
- 2GB of RAM storage
- stable internet connection
- a means of displaying visuals
- [Jupyter](https://jupyter.org/)

### Local Setup
1. First, clone the repository to your computer:
    ```
    git clone https://github.com/JakobFinci/BayesianNetworksforIBD.git
    ```

2. It's recommended that you run this project from a Python virtual environment with an anaconda interpreter, which can be set up like this:
    ```
    source /home/your-user/anaconda3/bin/activate
    conda activate base
    ```

3. If you don't already have pip installed, run this command from your new virtual environment:
    ```
    sudo apt install python-pip
    ```

4. Finally, use pip to install the required packages from the default source, PyPi, for packages and dependencies:
    ```
    pip install -r requirements.txt
    ```
    
## Running the Code

### On your local machine

1. The dataset used for this project is not included in the repository for patient privacy reasons. To run the code in the `src` folder or reproduce the Jupyter-hosted paper, manually place your dataset as a `.csv` file in the `data` folder and name it `final_cleaned.csv`.

The file must satisfy the following requirements:
- biomarker columns must contain real-valued numeric data  
- ordinal biopsy columns must contain either real-valued numeric data or strings in `["Quiescent","Mild","Moderate","Severe"]`  
- diagnostic or other label columns must contain strings  
- the dataset must not contain any `NaN` values  
- all columns must be labeled  

2. If your dataset is not already cleaned, place it in the `data` folder as a `.csv` file named `labs_and_biopsies.csv`, then run `build_clean_dataset` to preprocess it.

If your dataset uses biomarker, ordinal, or label columns that differ from those used in the report, call the function with:
- `unique_cols` (`list[str]`) for all column names in the dataset  
- `unique_ordinals` (`list[str]`) for all ordinal columns  
- `unique_label` (`str`) for the label variable  

Only one label variable is currently supported.

3. From a new terminal, navigate to the `bin` folder of your repo. (Note that you may need to run a different version of the executable depending on your environment)
    ```
    cd [your-path-to-repo]/bin
    ```

4. Begin a Python kernel. We recommend using Jupyter Lab:
    ```
    jupyter lab
    ```
    
4. From here, you can easily use all functions with plain-text explanations by following the report in:
   ```
   main/report.ipynb
   ```
    
## External Dependencies
The following are required to run this program. Note that requirements may already be satisfied and additional platform-specific dependencies may be required depending on your target environment.

### General Use
- [numpy](https://pypi.org/project/numpy/)
- [pip](https://pypi.org/project/pip/)
- [pandas](https://pypi.org/project/pandas/)

### Data Analysis
- [igraph](https://pypi.org/project/igraph/)
- [scikit-learn](https://pypi.org/project/scikit-learn/)
- [pgmpy](https://pypi.org/project/pgmpy/)
- [causal-learn](https://pypi.org/project/causal-learn/)

### Data Visualization
- [matplotlib](https://pypi.org/project/matplotlib/)

