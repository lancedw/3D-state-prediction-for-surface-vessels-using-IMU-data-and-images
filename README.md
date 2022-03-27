# Thesis
3D state estimation and prediction for surface vessels using sensor fusion

# Layout of repository
The main folder in the repo contains:
- worklog: where I log all my hours that I work on the thesis (necessary by thesis commitee)
- Nazar's paper: so I can easily revisit his work to aid with our work
- commands.txt: these are the commands used to run nazar's code but don't work for me
- environment.yml: conda environment for the notebooks
- code/ : this folder contains all code; 
  - 3dmodel/ : contains all data
  - Nazotron/ : contains all code based on or directly from Nazotron
  - Notebooks/ : contains all notebooks written by me with the code for all the different models
  - results/ : contains all plot results from training and testing
- Documentation/ : contains paper, presentations and guideline documents for my paper 

# Execution of the code
Only the notebooks can be executed. To execute them, create a conda environment with environment.yml file (Windows) 

or alternatively: try to run the first cell of any notebook and manually import all the packages that are imported in the notebook. 

Python 3.8._ was used. If GPU support is desired, Pytorch should be installed in the conda environment with the cuda toolkit extensions. To do this, follow the link to their website and select your OS, cuda version, etc and run the provided command in the environment. 
https://pytorch.org/get-started/locally/
(If your GPU only supports cuda 10.2, you will need to install a older version of pytorch)

# Udemy course
https://www.udemy.com/course/pytorch-for-deep-learning-with-python-bootcamp/learn/lecture/15027090#overview

# Links
- Deadlines: http://masterproef.tiwi.ugent.be/verplichte-taken/
- MarSur/MarLand: https://mecatron.rma.ac.be/index.php/projects/
- Reference Book (with opensource code + data) and Youtube playlist: http://databookuw.com/
- Reference Github page of Dynamicslab: https://github.com/dynamicslab
- Blender dataset: https://github.com/Nazotron1923/ship-ocean_simulation_BLENDER
- Nazotron github: https://github.com/Nazotron1923/Deep_learning_models_for_ship_motion_prediction_from_images/tree/master/Pre
- Literatures: https://we.tl/t-jCiFFMFqzo
- Example paper: https://libstore.ugent.be/fulltxt/RUG01/000/782/394/RUG01-000782394_2010_0001_AC.pdf

# sources
- chunked data: https://machinelearningmastery.com/how-to-load-visualize-and-explore-a-complex-multivariate-multistep-time-series-forecasting-dataset/
- multi-step multivariate: https://pangkh98.medium.com/multi-step-multivariate-time-series-forecasting-using-lstm-92c6d22cd9c2
- LSTM btc preprocessing: https://www.youtube.com/watch?v=jR0phoeXjrc&ab_channel=VenelinValkov
- LSTM btc predictor: https://www.youtube.com/watch?v=ODEGJ_kh2aA&ab_channel=VenelinValkov
- PR correlation: https://arc.aiaa.org/doi/10.2514/3.62949
- Dutch roll: https://en.wikipedia.org/wiki/Dutch_roll
- Power consumption forecasting: https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
- 
