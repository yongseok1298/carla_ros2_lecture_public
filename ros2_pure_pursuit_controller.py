#!/usr/bin/env python3
import math
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist


class PurePursuitFromGNSS(Node):
    def __init__(self):
        super().__init__("pure_pursuit_controller")

        self.sub_gnss = self.create_subscription(
            NavSatFix, "/carla/hero/gnss", self.gnss_cb, 10)
        self.sub_path = self.create_subscription(
            Path, "/carla/path/local", self.path_cb, 10)

        self.pub_cmd = self.create_publisher(
            Twist, "/carla/hero/cmd_vel", 10)

        self.lat0 = None
        self.lon0 = None
        self.cos_lat0 = 1.0

        self.curr_xy: Optional[Tuple[float, float]] = None
        self.prev_xy: Optional[Tuple[float, float]] = None
        self.local_xy: List[Tuple[float, float]] = []

        self.lookahead = 6.0
        self.wheel_base = 2.7
        self.target_speed = 5.0  # m/s

        self.timer = self.create_timer(0.05, self.control_loop)

    def gnss_cb(self, msg: NavSatFix):
        lat = msg.latitude
        lon = msg.longitude
        if self.lat0 is None:
            self.lat0 = lat
            self.lon0 = lon
            self.cos_lat0 = math.cos(math.radians(lat))
            xy = (0.0, 0.0)
        else:
            xy = self.latlon_to_xy(lat, lon)

        self.prev_xy = self.curr_xy
        self.curr_xy = xy

    def latlon_to_xy(self, lat: float, lon: float):
        dx = (lon - self.lon0) * (111320.0 * self.cos_lat0)
        dy = (lat - self.lat0) * 110540.0
        return dx, dy

    def path_cb(self, msg: Path):
        self.local_xy = [
            (p.pose.position.x, p.pose.position.y) for p in msg.poses
        ]

    def control_loop(self):
        if self.curr_xy is None or self.prev_xy is None or len(self.local_xy) < 2:
            return

        x, y = self.curr_xy
        px, py = self.prev_xy

        yaw = math.atan2(y - py, x - px)

        cos_y = math.cos(-yaw)
        sin_y = math.sin(-yaw)

        pts_local = []
        for gx, gy in self.local_xy:
            dx = gx - x
            dy = gy - y
            x_l = dx * cos_y - dy * sin_y
            y_l = dx * sin_y + dy * cos_y
            pts_local.append((x_l, y_l))

        target = None
        for x_l, y_l in pts_local:
            d = math.hypot(x_l, y_l)
            if x_l > 0.0 and d >= self.lookahead:
                target = (x_l, y_l)
                break
        if target is None:
            return

        xt, yt = target
        ld = math.hypot(xt, yt)
        alpha = math.atan2(yt, xt)
        delta = math.atan2(2.0 * self.wheel_base * math.sin(alpha), ld)
        steer_deg = math.degrees(delta)

        cmd = Twist()
        cmd.linear.x = float(self.target_speed)
        cmd.linear.y = 0.0      # brake
        cmd.angular.z = float(steer_deg)
        self.pub_cmd.publish(cmd)

    # optional: clamp function if needed later
    # def clamp(self, v, lo, hi):
    #     return hi if v > hi else lo if v < lo else v


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitFromGNSS()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

