import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import torch
import torch.nn as nn
import numpy as np
from geometry_msgs.msg import Pose, Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf_transformations import quaternion_from_euler
import time


reset=False
class DaggerInferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        # Load trained model parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = 108  # <-- Change this to your actual input feature dimension
        output_dim = 2  # steering, speed

        self.model = SimpleNet(input_dim=input_dim, output_dim=output_dim)
        model_path = "/sim_ws/src/ground_truth/models/dagger.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.start_position = [0.0,0.0]

        # Subscriber to sensor or processed features
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
    
        # Publisher to drive commands
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.get_logger().info("DAgger inference node started and model loaded.")

    def preprocess_scan(self, scan_msg):
        ranges = np.array(scan_msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=0.0, neginf=0.0)

        # Downsample by averaging every 10 points
        downsampled = ranges.reshape(-1, 10).mean(axis=1)
        # downsampled.shape should be (108,)
        return downsampled
    
    def reset_car_position(self):

        # Create the pose message
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.frame_id = 'map'
        pose_msg.header.stamp.sec = int(time.time())
        
        # Set position (x, y, z)
        pose_msg.pose.pose.position.x = self.start_position[0]
        pose_msg.pose.pose.position.y = self.start_position[1]
        pose_msg.pose.pose.position.z = 0.0
        
        # Set orientation (as quaternion from yaw)
        # q = quaternion_from_euler(0, 0, self.start_position[2])
        # pose_msg.pose.pose.orientation.x = q[0]
        # pose_msg.pose.pose.orientation.y = q[1]
        # pose_msg.pose.pose.orientation.z = q[2]
        # pose_msg.pose.pose.orientation.w = q[3]
        
        # Publish the pose
        self.reset_pub.publish(pose_msg)
        
        # Sleep a bit to let the simulator respond
        time.sleep(0.5)
        
        return True


    def scan_callback(self, scan_msg):
        # Preprocess LaserScan to model input

        min_distance = min(scan_msg.ranges)
        if min_distance < 0.2:
            self.reset_car_position()

        input_tensor = self.preprocess_scan(scan_msg)
        input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).to(self.device)  # batch size 1

        # Run model inference
        with torch.no_grad():
            output = self.model(input_tensor).cpu().numpy()[0]

        steering_angle = float(output[0])
        speed = float(output[1])

        # Construct and publish drive command
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DaggerInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


# Paste your SimpleNet class here exactly as in train.py
class SimpleNet(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
    def forward(self, x):
        return self.net(x)
