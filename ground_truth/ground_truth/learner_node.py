import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
import csv
import os
from datetime import datetime

class DAggerLearner(Node):
    def __init__(self):
        super().__init__('dagger_learner')

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.drive_sub = self.create_subscription(AckermannDriveStamped, '/drive', self.drive_callback, 10)

        self.expert_action = None
        self.data = []  # Stores (input, label) pairs

        self.sample_count = 0
        self.declare_parameter('save_path', 'dagger_dataset.csv')
        self.save_path = self.get_parameter('save_path').get_parameter_value().string_value

        self.get_logger().info('DAgger learner initialized and recording.')

    def scan_callback(self, scan_msg):
        if self.expert_action is None:
            return  # No expert data yet

        # Example: downsample and normalize laser scan
        ranges = np.array(scan_msg.ranges)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=0.0, neginf=0.0)
        downsampled = ranges[::10]  # Reduce dimensionality

        # Normalize input (assume max range is 10.0)
        normalized_scan = np.clip(downsampled / 10.0, 0.0, 1.0)

        input_features = normalized_scan.tolist()
        label = [self.expert_action.drive.steering_angle, self.expert_action.drive.speed]

        self.data.append((input_features, label))
        self.sample_count += 1

        if self.sample_count % 500 == 0:
            self.get_logger().info(f'Samples collected: {self.sample_count}')

    def drive_callback(self, msg):
        self.expert_action = msg

    def destroy_node(self):
        self.get_logger().info(f'Total samples collected: {self.sample_count}')
        self.save_dataset()
        super().destroy_node()

    def save_dataset(self):
        if not self.data:
            self.get_logger().warn('No data to save.')
            return

        try:
            # Ensure save_path is directory
            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path, exist_ok=True)

            # Generate a unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'dataset_{timestamp}.csv'
            full_path = os.path.join(self.save_path, filename)

            with open(full_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                for features, label in self.data:
                    writer.writerow(features + label)
            self.get_logger().info(f'Dataset saved to {full_path}')

        except Exception as e:
            self.get_logger().error(f'Failed to save dataset: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = DAggerLearner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
