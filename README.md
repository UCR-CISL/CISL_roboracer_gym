# CISL_roboracer_gym
## Setup
1. Run docker compose
```bash
docker-compose up -d
```

2. Edit the container in an interactive bash shell
```bash
docker exec -it cisl-gym_sim_1 bash
``` 
3. Source and run simulator
```bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

4. In a new terminal, launch the RL agent in training mode:
```bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
ros2 launch f1tenth_rl rl_agent_launch.py training_mode:=true model_type:=dqn
```