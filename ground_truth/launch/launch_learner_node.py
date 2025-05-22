from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """Launch file for DAgger learner node"""
    
    return LaunchDescription([
        Node(
            package='ground_truth',  # Change to your actual package name if different
            executable='learner_node',
            name='dagger_learner',
            output='screen',
            parameters=[{
                'save_path':'/sim_ws/src/ground_truth/raw_data' ,  # Change this path as needed
                'max_samples': 5000,                             # Adjust as needed
                'save_interval': 500                             # Adjust as needed
            }]
        )
    ])
