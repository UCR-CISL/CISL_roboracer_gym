import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import torch
import torch.nn as nn
import numpy as np

class DaggerLoggerNode(Node):
    def __init__ (self):
        super().__init__('dagger collection node')

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.expert_drive_sub = self.create_subscription(AckermannDriveStamped, '/expert_drive', self.expert_callback,10)
        self.learner_drive_sub = self.create_subscription(AckermannDriveStamped, '/drive', self.learner_callback,10)


        self.latest_scan=None
        self.latest_expert_drive=None
        self.latest_learner_drive=None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = 108  # <-- Change this to your actual input feature dimension
        output_dim = 2  # steering, speed

        self.model = SimpleNet(input_dim=input_dim, output_dim=output_dim)
        model_path = "/sim_ws/src/ground_truth/models/dagger.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def preprocess_scan(self, scan_msg):
        ranges = np.array(scan_msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=0.0, neginf=0.0)

        # Downsample by averaging every 10 points
        downsampled = ranges.reshape(-1, 10).mean(axis=1)
        # downsampled.shape should be (108,)
        return downsampled



    def log_data(self):
        pass

    def scan_callback(self,scan_msg):
        self.latest_scan=scan_msg
        self.log_data()

    def expert_callback(self, drive_msg):
        self.latest_expert_drive=drive_msg
        self.log_data()

    def learner_callback(self, drive_msg):
        self.latest_learner_drive=drive_msg
        self.log_data()

def main(args=None):
    rclpy.init(args=args)
    node = DaggerLoggerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ =="main":
    main()

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

        

    
