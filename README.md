# CISL_roboracer_gym
## Setup

1. Run the docker compose command. This runs the detached container. 
```bash
docker-compose up -d
```
2. Excecute the container in an interactive terminal,
```bash
docker exec -it <container-id> bash
```
3. Start the simulator
```bash
colcon build # build the workspsace
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

4. Start the training node
```bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
ros2 launch f1tenth_rl rl_agent_launch.py training_mode:=true model_type:=dqn
```

## Troubleshooting

1. If you get the following error, simply run the build command again.
```bash
--------------------
  46 |     COPY . /sim_ws/src/f1tenth_gym_ros
  47 | >>> RUN source /opt/ros/foxy/setup.bash && \
  48 | >>>     cd sim_ws/ && \
  49 | >>>     apt-get update --fix-missing && \
  50 | >>>     rosdep install -i --from-path src --rosdistro foxy -y && \
  51 | >>>     colcon build
  52 |     
--------------------
ERROR: failed to solve: process "/bin/bash -c source /opt/ros/foxy/setup.bash &&     cd sim_ws/ &&     apt-get update --fix-missing &&     rosdep install -i --from-path src --rosdistro foxy -y &&     colcon build" did not complete successfully: exit code: 1
ERROR: Service 'sim' failed to build : Build failed
```
2. If error, Open docker desktop and go to Settings → Resources → File Sharing. Add /tmp to the virtual file share
```bash
```
