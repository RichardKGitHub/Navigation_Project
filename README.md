working Version:
source udacity_nav_project
git Pnav_04
git nav_04
back_up:
source udacity_nav_project_copy_env_running
git Pnav_03
git nav_03
Old not everyting installed environment (without packages and instructions from git) = udacity_nav_project_2
<file_name="/data/Banana_Linux_NoVis/Banana.x86_64>
##dependencies

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
##Run the Script
In your shell run:
```
python3.6 Navigation.py
```
For specification of interaction-mode and -config-file run:
```
python3.6 Navigation.py --train True --config_file config_01.json
```
Info: \
The UnityEnvironment is expected at `/data/Banana_Linux_NoVis/Banana.x86_64`. \
This can be changed in the `config.json` file if necessary.