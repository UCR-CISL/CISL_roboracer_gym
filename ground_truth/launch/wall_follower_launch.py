from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """Generate launch description for farthest point navigator with wider turns"""
    
    return LaunchDescription([
        Node(
            package='ground_truth',
            executable='wall_follower',  # Executable name remains the same for compatibility
            name='farthest_point_navigator',
            output='screen',
            parameters=[{
                'speed': 0.8,                # Reduced speed for safer wider turns
                'max_steering': 0.25,        # Reduced maximum steering angle for wider arcs
                'min_distance': 0.5,
                'max_distance': 10.0,
                'angle_window': 60.0,        # Increased window for smoother, wider turns
                'front_weight': 1.2,         # Reduced front weight to allow more side exploration
                'safety_threshold': 1.2,     # Increased safety threshold to start turns earlier
                'reactive_distance': 3.0     # Increased to be more reactive at greater distances
            }]
        )
    ])
