# Towards Safer Cities: A Safety-Oriented Framework for Urban Traffic Signal Control Testing and Enhancement
## Setup
The code was developed and tested on Windows 10 using Python 3.11 and TensorFlow 2.18.0.
The required thired-party packages are:
```
tensorflow==2.18.0
numpy==1.26.4
pandas==2.1.4
pywinauto==0.6.9
alive_progress==3.2.0
joblib==1.2.0
tensorflow_probability==0.25.0
```
The microscopic traffic simulation environment is based on PTV VISSIM 2022, which requires an active subscription license to run.
The code_p1 folder contains safety-testing implementations for the following Traffic Signal Control (TSC) models: A2C, DQDQN, DQN, IQN, PPO, REINFORCE, and SAC.
The code_p2 folder includes implementations of the Webster-1 and Webster-2 models.
The code_p3 folder is for the MP model.
## Execute
For each folder, unzip SSAM.7z archive. Configure the model and mode (original or after applying SCRT) in the script. For example:
```
TSCmodel = 'A2C'
Mode = "bef SCRT"
```
Run run.py to generate the testing curves. Use plot.py to compute and visualize HCRE and its corresponding curve.
