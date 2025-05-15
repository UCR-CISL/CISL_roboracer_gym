import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from rcl_interfaces.msg import ParameterDescriptor, FloatingPointRange
import numpy as np
import math

class FarthestPointNavigator(Node):
    """
    A controller that navigates towards the farthest point in a laser scan.
    This helps the vehicle find open spaces and navigate through them.
    """
    def __init__(self):
        super().__init__('farthest_point_navigator')
        
        # Parameter descriptors for validation
        float_range = FloatingPointRange(from_value=0.0, to_value=5.0, step=0.1)
        speed_desc = ParameterDescriptor(
            description='Base speed of the car',
            floating_point_range=[float_range]
        )
        
        # Parameters
        self.declare_parameter('speed', 1.0, speed_desc)  # Base speed
        self.declare_parameter('max_steering', 0.4)  # Max steering angle
        self.declare_parameter('min_distance', 0.5)  # Minimum distance to consider valid
        self.declare_parameter('max_distance', 10.0)  # Maximum distance to consider valid
        self.declare_parameter('angle_window', 30.0)  # Angular window in degrees for smoothing
        self.declare_parameter('front_weight', 2.0)  # Weight for preferring forward direction
        self.declare_parameter('safety_threshold', 0.8)  # Safety threshold distance
        self.declare_parameter('reactive_distance', 1.5)  # Distance to start being reactive
        
        # Get parameters
        self.speed = self.get_parameter('speed').value
        self.max_steering = self.get_parameter('max_steering').value
        self.min_distance = self.get_parameter('min_distance').value
        self.max_distance = self.get_parameter('max_distance').value
        self.angle_window = self.get_parameter('angle_window').value * (math.pi / 180.0)  # Convert to radians
        self.front_weight = self.get_parameter('front_weight').value
        self.safety_threshold = self.get_parameter('safety_threshold').value
        self.reactive_distance = self.get_parameter('reactive_distance').value
        
        # State variables
        self.last_valid_steering = 0.0
        self.is_emergency = False
        self.consecutive_invalid_scans = 0
        self.max_invalid_scans = 5
        
        # Create timer for parameter updates
        self.create_timer(1.0, self.update_parameters)
        
        # Create subscribers and publishers
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10)
        
        self.get_logger().info('Farthest point navigator initialized')
    
    def update_parameters(self):
        """Update parameters periodically from ROS parameters"""
        self.speed = self.get_parameter('speed').value
        self.max_steering = self.get_parameter('max_steering').value
        self.min_distance = self.get_parameter('min_distance').value
        self.max_distance = self.get_parameter('max_distance').value
        self.angle_window = self.get_parameter('angle_window').value * (math.pi / 180.0)
        self.front_weight = self.get_parameter('front_weight').value
        self.safety_threshold = self.get_parameter('safety_threshold').value
        self.reactive_distance = self.get_parameter('reactive_distance').value
    
    def scan_callback(self, scan_msg):
        """Process laser scan data and navigate toward the farthest point"""
        try:
            # Check if we have a valid scan
            if len(scan_msg.ranges) == 0:
                self.handle_invalid_scan("Empty scan")
                return
            
            # Safety check first
            if self.safety_check(scan_msg):
                return
                
            # Find farthest point with smoothing
            target_angle, max_distance = self.find_farthest_point(scan_msg)
            
            if target_angle is None:
                self.handle_invalid_scan("No valid farthest point")
                return
            
            # Reset emergency state and counter if we got valid data
            self.is_emergency = False
            self.consecutive_invalid_scans = 0
            
            # Calculate steering angle (target_angle is already relative to front)
            steering_angle = self.calculate_steering_angle(target_angle, max_distance)
            
            # Limit steering angle
            steering_angle = max(-self.max_steering, min(self.max_steering, steering_angle))
            self.last_valid_steering = steering_angle
            
            # Calculate adaptive speed
            adaptive_speed = self.adaptive_speed(steering_angle, max_distance)
            
            # Publish drive command
            self.publish_drive_command(steering_angle, adaptive_speed)
            
        except Exception as e:
            self.get_logger().error(f'Error in scan_callback: {str(e)}')
            self.handle_invalid_scan(f"Exception: {str(e)}")
    
    def find_farthest_point(self, scan_msg):
        """Find the farthest point in the scan with smoothing over an angular window, 
        optimized for finding wide open areas"""
        # Convert scan to numpy array
        ranges = np.array(scan_msg.ranges)
        
        # Replace invalid values (inf/nan) with 0
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Filter by min/max distance
        filtered_ranges = np.copy(ranges)
        filtered_ranges[(ranges < self.min_distance) | (ranges > self.max_distance)] = 0.0
        
        if np.max(filtered_ranges) == 0.0:
            return None, 0.0
        
        # Apply front direction weighting
        angle_array = np.linspace(
            scan_msg.angle_min, 
            scan_msg.angle_max, 
            len(scan_msg.ranges)
        )
        
        # Weight by how close to front (0 radians) the angle is, but with a wider preference zone
        # This front_factor will give higher weight to a wider frontal area
        front_factor = 1.0 + self.front_weight * np.cos(angle_array)**2
        weighted_ranges = filtered_ranges * front_factor
        
        # Apply smoothing using a sliding window over angles
        smoothed_ranges = np.zeros_like(weighted_ranges)
        num_points = len(weighted_ranges)
        half_window = int(self.angle_window / scan_msg.angle_increment / 2)
        
        # Enhanced smoothing for identifying wide passages
        for i in range(num_points):
            # Calculate window indices with wrapping
            window_indices = [(i + j) % num_points for j in range(-half_window, half_window + 1)]
            
            # Calculate weighted average - gives higher weight to shorter distances
            # This ensures we prefer passages that are wide, not just a single far point
            window_ranges = weighted_ranges[window_indices]
            valid_ranges = window_ranges[window_ranges > 0]
            
            if len(valid_ranges) > 0:
                # Average the central portion more heavily to find truly open passages
                # and use min values to ensure we have width
                central_weight = 0.7
                
                # Find the minimum range within a narrower central window
                central_half_window = half_window // 2
                central_indices = [(i + j) % num_points for j in range(-central_half_window, central_half_window + 1)]
                central_ranges = weighted_ranges[central_indices]
                central_valid = central_ranges[central_ranges > 0]
                
                # If we have valid central measurements, compute a weighted score
                if len(central_valid) > 0:
                    central_min = np.min(central_valid)
                    outer_min = np.min(valid_ranges)
                    # Combine the values, weighting the central portion more
                    smoothed_ranges[i] = central_weight * central_min + (1 - central_weight) * outer_min
                else:
                    smoothed_ranges[i] = np.mean(valid_ranges)
            else:
                smoothed_ranges[i] = 0
        
        # Find the angle with maximum distance
        max_idx = np.argmax(smoothed_ranges)
        max_angle = angle_array[max_idx]
        
        # Use the actual unweighted range for the max calculation
        max_distance = ranges[max_idx]  # Use actual distance, not weighted
        
        return max_angle, max_distance
    
    def safety_check(self, scan_msg):
        """Check for obstacles that are too close"""
        # Create a safety zone in front
        front_arc_angle = 60 * (math.pi / 180)  # 60 degrees in front
        center_idx = len(scan_msg.ranges) // 2
        angle_increment = scan_msg.angle_increment
        idx_range = int(front_arc_angle / angle_increment / 2)
        
        # Get front arc indices
        start_idx = max(0, center_idx - idx_range)
        end_idx = min(len(scan_msg.ranges) - 1, center_idx + idx_range)
        
        # Get valid ranges
        front_ranges = np.array(scan_msg.ranges[start_idx:end_idx+1])
        valid_ranges = front_ranges[(~np.isnan(front_ranges)) & (front_ranges < 10.0) & (front_ranges > 0.01)]
        
        if len(valid_ranges) > 0 and np.min(valid_ranges) < self.safety_threshold:
            self.get_logger().warning(f'Obstacle too close: {np.min(valid_ranges)}m')
            
            # If very close, emergency stop
            if np.min(valid_ranges) < self.safety_threshold / 2:
                self.emergency_stop()
                return True
            
            # Otherwise, execute avoidance maneuver
            self.execute_avoidance_maneuver(scan_msg)
            return True
            
        return False
    
    def execute_avoidance_maneuver(self, scan_msg):
        """Simple avoidance maneuver when obstacles are close"""
        # Find clearest direction to turn
        left_ranges = np.array(scan_msg.ranges[:len(scan_msg.ranges)//4])
        right_ranges = np.array(scan_msg.ranges[3*len(scan_msg.ranges)//4:])
        
        # Filter valid ranges
        left_valid = left_ranges[(~np.isnan(left_ranges)) & (left_ranges < 10.0) & (left_ranges > 0.01)]
        right_valid = right_ranges[(~np.isnan(right_ranges)) & (right_ranges < 10.0) & (right_ranges > 0.01)]
        
        # Average distances
        left_avg = np.mean(left_valid) if len(left_valid) > 0 else 0
        right_avg = np.mean(right_valid) if len(right_valid) > 0 else 0
        
        # Choose direction
        if left_avg > right_avg:
            steering_angle = -self.max_steering  # Turn left
        else:
            steering_angle = self.max_steering   # Turn right
            
        # Slow down
        self.publish_drive_command(steering_angle, self.speed * 0.5)
    
    def calculate_steering_angle(self, target_angle, distance):
        """Calculate steering angle based on target angle and distance"""
        # Reduce the steering angle effect for wider turns
        dampening_factor = 0.6  # Lower value = wider turns
        
        # Basic proportional control - steer proportionally to the target angle
        base_steering = target_angle * dampening_factor
        
        # Add reactive component - steer more aggressively when close to obstacles
        reactivity = max(0, 1.0 - (distance / self.reactive_distance))
        
        # For wider turns, reduce the impact of reactivity on close obstacles
        reactivity_factor = 0.7
        reactive_steering = base_steering * (1.0 + reactivity * reactivity_factor)
        
        # Add turn smoothing - this helps make turns less abrupt
        if abs(reactive_steering) > 0.1:
            # Apply a minimum turn radius effect
            turn_direction = 1.0 if reactive_steering > 0 else -1.0
            min_turn_radius = 0.12  # Minimum steering angle value for making any turn
            reactive_steering = turn_direction * max(min_turn_radius, abs(reactive_steering))
            
        return reactive_steering
    
    def adaptive_speed(self, steering_angle, distance):
        """Calculate adaptive speed based on steering angle and distance"""
        # More aggressive speed reduction when turning for wider, more controlled turns
        steering_factor = 1.0 - (abs(steering_angle) / self.max_steering) * 0.9
        
        # More conservative distance factor 
        distance_factor = min(1.0, distance / (self.reactive_distance * 2.5))
        
        # Apply a lower max speed factor for corners
        max_corner_speed_factor = 0.8
        
        # Return the adapted speed with a higher minimum to prevent getting too slow
        return self.speed * max(0.4, min(max_corner_speed_factor, steering_factor * distance_factor))
    
    def handle_invalid_scan(self, reason):
        """Handle invalid scan data"""
        self.consecutive_invalid_scans += 1
        self.get_logger().warning(f'Invalid scan ({reason}): {self.consecutive_invalid_scans}/{self.max_invalid_scans}')
        
        if self.consecutive_invalid_scans >= self.max_invalid_scans:
            if not self.is_emergency:
                self.get_logger().error('Too many invalid scans, stopping vehicle')
                self.emergency_stop()
                self.is_emergency = True
        else:
            # Use last valid steering command with reduced speed
            self.publish_drive_command(self.last_valid_steering, self.speed * 0.5)
    
    def emergency_stop(self):
        """Perform emergency stop"""
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = 0.0
        self.drive_pub.publish(drive_msg)
    
    def publish_drive_command(self, steering_angle, speed):
        """Publish drive command"""
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    try:
        navigator = FarthestPointNavigator()
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Cleanup
        if 'navigator' in locals():
            navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
