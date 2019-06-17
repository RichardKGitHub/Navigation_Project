working Version:
source udacity_nav_project
git nav_09 Hübsch für Abgabe ; Weniger Netze (changed in init_agent(2x)S; existence of Agents; config.json) argparse fct. nicht
git Pnav_05
git nav_08  Einige Kommentare weg double DQN läuft prinzipiell eventl. incl. Fehler
checkpoint.pth gerassel | checkpoint_01.pth second solved net in 428 Episodes | checkpoint_02 first solved net
checkpoint_03.pth stop if mean(100)=24 --> Average Score: 14.52	Episode 4999/4999	Time since start: 0:57:08	deltaTime: 0:00:17 
checkpoint_04.pth stop if min(100)=13 --> Episodes: 999 | Test: gleichbleibend Average Score: 15.36 Time since start: 0:07:19	deltaTime: 0:00:10
back_up:
source udacity_nav_project_copy_env_running
git Pnav_04
git nav_07  alle Kommentare noch da
Old not everyting installed environment (without packages and instructions from git) = udacity_nav_project_2
<file_name="/data/Banana_Linux_NoVis/Banana.x86_64>
##Project Details 
This Project was completed in the course of the Deep Reinforcement Learning Nanodegree Program from Udacity Inc. \
In this Project a Agent gets trained to pick up yellow bananas and to avoid blue bananas
- Action space: 4
- state space: 37
- A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana
- one Episode takes 300 frames (300 decisions of the Agent) 
- the environment is solved when the agent gets an average score of +13 over 100 consecutive episodes

##Getting Started - dependencies

####Python version
- python3.6 
####Packages
- Install the required pip packages:
  ```
  pip install -r requirements.txt
  ```

- Only if your hardware supports it: install pytorch_gpu (otherwise skip it since torch will be installed with the environment anyway)  
  ```
  conda install pytorch_gpu
  ```
####Environment
- Install gym 
  - [gym](https://github.com/openai/gym) 
  - follow the given instructions to set up gym (instructions can be found in README.md at the root of the repository)
  - make `gym` a Sources Root of your Project
- The environment for this project is included in the following Git-repository
  - [Git-repository](https://github.com/udacity/deep-reinforcement-learning#dependencies)
  - follow the given instructions to set up the Environment (instructions can be found in `README.md` at the root of the repository)
  - make the included `python` folder a Sources Root of your Project
- Insert the below provided Unity environment into the `p1_navigation/` folder of your `deep-reinforcement-learning/` folder from the previous step and unzip (or decompress) the file
  - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
  - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
  - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
  - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
##Instructions - Run the Script
In your shell run:
```
python3.6 Navigation.py
```
For specification of interaction-mode and -config-file run:
```
python3.6 Navigation.py --train True --config_file config.json
```
Info: \
The UnityEnvironment is expected at `"environment_path": "/data/Banana_Linux_NoVis/Banana.x86_64"`. \
This can be changed in the `config.json` file if necessary.