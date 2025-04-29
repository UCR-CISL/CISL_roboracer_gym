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
