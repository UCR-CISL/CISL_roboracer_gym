# CISL_roboracer_gym
## Setup

1. Run the docker compose command. This runs the detached container. 
```bash
docker-compose up -d
```

2. Your docker container should be running if you see
```bash
Creating cisl_roboracer_gym_sim_1 ... done
```

3. Excecute the container in an interactive terminal,
```bash
docker exec -it cisl_roboracer_gym-sim-1 bash
```
4. Start the simulator
```bash
colcon build # build the workspsace
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

5. Start the training node
```bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
ros2 launch f1tenth_rl rl_agent_launch.py training_mode:=true model_type:=dqn
```



# Troubleshooting

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
Starting cisl_roboracer_gym_sim_1 ... error

ERROR: for cisl_roboracer_gym_sim_1  Cannot start service sim: Mounts denied: 
The path /tmp/.X11-unix is not shared from the host and is not known to Docker.
You can configure shared paths from Docker -> Preferences... -> Resources -> File Sharing.
See https://docs.docker.com/ for more info.

ERROR: for sim  Cannot start service sim: Mounts denied: 
The path /tmp/.X11-unix is not shared from the host and is not known to Docker.
You can configure shared paths from Docker -> Preferences... -> Resources -> File Sharing.
See https://docs.docker.com/ for more info.
ERROR: Encountered errors while bringing up the project.
```
3. If using docker desktop, you will need to configure the sokcet to run from the docker 
```bash
docker.errors.DockerException: Error while fetching server API version: ('Connection aborted.', ConnectionRefusedError(111, 'Connection refused'))
```
1. Set the DOCKER_HOST environment variable to point to the correct socket:
```bash
export DOCKER_HOST=unix:///var/run/docker.sock
```

2. You can add this to your .bashrc or .zshrc to make it permanent:
```bash
echo 'export DOCKER_HOST=unix:///var/run/docker.sock' >> ~/.bashrc
source ~/.bashrc
```

3. You could also create a symbolic link to point Docker Desktop's socket path to the system socket:
```bash
mkdir -p ~/.docker/desktop
sudo ln -sf /var/run/docker.sock ~/.docker/desktop/docker.sock
```

After implementing one of these fixes, try running docker ps again to confirm it works, and then try your original docker-compose up -d command.

If this error comes up...

```bash
 => ERROR [internal] load metadata for docker.io/library/ros:foxy                                                                            0.2s
------
 > [internal] load metadata for docker.io/library/ros:foxy:
------
Dockerfile:23
--------------------
  21 |     # SOFTWARE.
  22 |     
  23 | >>> FROM ros:foxy
  24 |     
  25 |     SHELL ["/bin/bash", "-c"]
--------------------
ERROR: failed to solve: ros:foxy: failed to resolve source metadata for docker.io/library/ros:foxy: error getting credentials - err: exec: "docker-credential-desktop": executable file not found in $PATH, out: ``
ERROR: Service 'sim' failed to build : Build failed
```
Run this command and remove credStore
``bash
nano ~/.docker/config.json
```
