#!/usr/bin/env python3
import time, rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

V_MPS = 10.0
STEER_D = 0.0
BRAKE= 0.0

carla_vehicle_control_topic = "/carla/hero/cmd_vel"

class VehicleControl(Node):
    def __init__(self):
        super().__init__("ros2_vehicle_control")
        while True:
            pub = self.create_publisher(Twist,
            carla_vehicle_control_topic, 10)
            msg = Twist()
            msg.linear.x = V_MPS
            msg.linear.y = BRAKE
            msg.angular.z = STEER_D
            pub.publish(msg)
            self.get_logger().info(f"publish -> v={V_MPS:.2f} m/s, brake={BRAKE:.2f}, steer={STEER_D:.1f} deg")
            time.sleep(0.05) # 20hz
def main():
    rclpy.init()
    rclpy.spin(VehicleControl())
if __name__ == "__main__":
    main()