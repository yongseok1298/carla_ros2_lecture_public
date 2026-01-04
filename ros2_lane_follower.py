#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

class LaneController(Node):
    def __init__(self):
        super().__init__('lane_controller')

        self.sub_angle = self.create_subscription(
            Float32,
            '/lane/steering_angle',
            self.angle_callback,
            10
        )
        self.pub_cmd = self.create_publisher(Twist, '/carla/hero/cmd_vel', 10)

        self.target_speed_kph = 10.0
        self.current_steering_angle = 0.0
        
        self.get_logger().info(f"Lane Controller Started! Target Speed: {self.target_speed_kph} km/h")

    def angle_callback(self, msg: Float32):
        self.current_steering_angle = msg.data

        cmd_msg = Twist()

        # N km/h = N / 3.6 m/s
        cmd_msg.linear.x = self.target_speed_kph / 3.6 
        cmd_msg.angular.z = self.current_steering_angle 
        cmd_msg.linear.y = 0.0

        self.pub_cmd.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LaneController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.linear.y = 1.0
        node.pub_cmd.publish(stop_msg)
        node.get_logger().info("Stopping Vehicle...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()