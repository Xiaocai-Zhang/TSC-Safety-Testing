# Towards Safer Cities: A Safety-Oriented Framework for Urban Traffic Signal Control Testing and Enhancement
## Setup
Code was developed and tested on Windows 10 with Python 3.11 and TensorFlow 2.18.0.
The thired-party packages requirements are:
```
tensorflow==2.18.0
numpy==1.26.4
pandas==2.1.4
pywinauto==0.6.9
alive_progress==3.2.0
joblib==1.2.0
tensorflow_probability==0.25.0
```
The microscopic traffic simulation is using PTV VISSIM 2022, you will need subscribtion membershp before using it. code_p1 folder are for safety testing for TSC models: A2C, DQDQN, DQN, IQN, PPO, REINFORCE, and SAC. code_p2 folder for Webster-1 and Webster-2 models. code_p2 folder are for MP model.
For each folder, unzip SSAM.7z folder first and configue model and mode, for example,
```
TSCmodel = 'A2C'
Mode = "bef SCRT"
```
execute run.py to generate the curve of tetsing, using plot.py to get HCRE and its curve.
