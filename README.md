# ML_chem_course

This project contains several python notebooks to generate chemistry learning data and learn ML models.

Description of notebooks:

+ *0D_database_generation.ipynb*: in this notebook we generate a database of 0D reactors (at identical pressure), and process them to build a training database for ML models. It includes scaling and potentially transform (log.,...) of the data.

+ *0D_ann_learning.ipynb*: in this notebook we train and test neural networks to replace the CVODE solver for 0D reactors computations. We also investigate methodologies to add constraints in the learned model. The notebook is split in three main parts:
    - Learning of a baseline model
    - Learning of a model with soft physical constraints
    - Learning of a model with hard constraints