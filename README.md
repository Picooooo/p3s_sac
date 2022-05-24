# Population-Guided Parallel Policy Search (P3S)

The algorithm is based on the paper "Population-Guided Parallel Policy Search for Reinforcement Learning" submitted to ICLR 2020.
The P3S codes are modified from the code of Soft Actor-Critic (SAC) (https://github.com/haarnoja/sac)

# Getting Started

To get the environment installed correctly, you will first need to clone [rllab](https://github.com/rll/rllab), and have its path added to your PYTHONPATH environment variable.

1. Clone rllab

```
cd <installation_path_of_your_choice>
git clone https://github.com/rll/rllab.git
cd rllab
export PYTHONPATH=$(pwd):${PYTHONPATH}
```

2. [Download](https://www.roboti.us/index.html) and copy mujoco files to rllab path:
   If you're running on OSX, download https://www.roboti.us/download/mjpro131_osx.zip instead, and copy the `.dylib` files instead of `.so` files.

```
mkdir -p /tmp/mujoco_tmp && cd /tmp/mujoco_tmp
wget -P . https://www.roboti.us/download/mjpro131_linux.zip
unzip mjpro131_linux.zip
mkdir <installation_path_of_your_choice>/rllab/vendor/mujoco
cp ./mjpro131/bin/libmujoco131.so <installation_path_of_your_choice>/rllab/vendor/mujoco
cp ./mjpro131/bin/libglfw.so.3 <installation_path_of_your_choice>/rllab/vendor/mujoco
cd ..
export MUJOCO_PY_MJPRO_PATH=/tmp/mujoco_tmp/mjpro131
```

3. Copy your Mujoco license key (mjkey.txt) to rllab path:

```
cp <mujoco_key_folder>/mjkey.txt <installation_path_of_your_choice>/rllab/vendor/mujoco
export MUJOCO_PY_MJKEY_PATH=<installation_path_of_your_choice>/rllab/vendor/mujoco/mjkey.txt
```
5. Install Mujoco210:
```
cd <installation_path _of_your_choice>
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz
cp -r mujoco210/bin/* /usr/lib/
mkdir ~/.mujoco
cp -r mujoco210 ~/.mujoco
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:<installation_path_of_your_choice>/mujoco210/bin:/root/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/nvidia
```

4. Export path at the end of .bashrc file:

```
vi ~/bashrc
copy "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<installation_path_of_your_choice>/.mujoco/mujoco210/bin" to the end of file
```

5. Install nescessary packages - python==3.7

```
pip install tensorflow==1.14.0 path gtimer lasagne Theano dateutils
pip3 install -U 'mujoco-py<2.2,>=2.1'
pip install pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/v0.1/requirements.txt
pip install gym==0.14 joblib gtimer pandas matplotlib pyprind
sudo apt install patchelf
```
6. Trainning

```
cd <p3s_folder>
python main.py --env ant


<!-- 4. Go to "p3s" directory

```
cd <p3s_folder>
```

5. Create and activate conda environment

```
<!-- cd p3s # TODO.before_release: update folder name
conda env create -f environment.yml
source activate p3s -->
```

The environment should be ready to run. See examples section for examples of how to train and simulate the agents.

Finally, to deactivate and remove the conda environment:

```
<!-- source deactivate
conda remove --name p3s --all -->
```
 -->
## Examples

### Training and simulating an agent

```
<!-- python ./examples/mujoco_all_p3s_sac.py --env=ant
python ./examples/mujoco_all_p3s_sac.py --env=half-cheetah
python ./examples/mujoco_all_p3s_sac.py --env=hopper
python ./examples/mujoco_all_p3s_sac.py --env=walker
python ./examples/mujoco_all_p3s_sac.py --env=delayed_ant
python ./examples/mujoco_all_p3s_sac.py --env=delayed_half-cheetah
python ./examples/mujoco_all_p3s_sac.py --env=delayed_hopper
python ./examples/mujoco_all_p3s_sac.py --env=delayed_walker -->
```
