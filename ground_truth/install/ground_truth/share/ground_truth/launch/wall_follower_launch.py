from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """Generate launch description for farthest point navigator with faster, more aggressive settings"""
    
    return LaunchDescription([
        Node(
            package='ground_truth',
            executable='wall_follower',
            name='farthest_point_navigator',
            output='screen',
            parameters=[{
                'speed': 5.0,                 
                'max_steering': 0.35,         
                'min_distance': 0.4,          
                'max_distance': 12.0,         
                'angle_window': 45.0,         
                'front_weight': 1.0,          
                'safety_threshold': 0.1,      
                'reactive_distance': 2.0      
            }]
        )
    ])
