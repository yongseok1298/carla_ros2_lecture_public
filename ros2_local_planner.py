#!/usr/bin/env python3
import math
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseArray, PoseStamped


class LocalPathAvoid(Node):
    def __init__(self):
        super().__init__("local_path_avoid")

        self.sub_gnss = self.create_subscription(
            NavSatFix, "/carla/hero/gnss", self.gnss_cb, 10)
        self.sub_obs = self.create_subscription(
            PoseArray, "/carla/obstacles_2d", self.obs_cb, 10)

        self.pub_local = self.create_publisher(
            Path, "/carla/path/local", 10)

        self.lat0 = None
        self.lon0 = None
        self.cos_lat0 = 1.0

        self.current_xy: Optional[Tuple[float, float]] = None
        self.global_xy: List[Tuple[float, float]] = []   # loaded from csv
        self.obstacles: List[Tuple[float, float]] = []   # in vehicle frame (x,y)

        self.L = 20.0
        self.ds = 0.5
        self.safe_lat = 2.0
        self.max_offset = 3.0

        self.load_global_path("global_path.csv")
        if not self.global_xy:
            self.get_logger().warn("global_path.csv is empty or not found.")

        self.timer = self.create_timer(0.1, self.timer_cb)

    def load_global_path(self, filename: str):
        """Load global path from csv file: x,y per line."""
        try:
            with open(filename, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",")
                    if len(parts) < 2:
                        continue
                    x = float(parts[0])
                    y = float(parts[1])
                    self.global_xy.append((x, y))
            self.get_logger().info(f"Loaded {len(self.global_xy)} points from {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to load {filename}: {e}")

    def gnss_cb(self, msg: NavSatFix):
        lat = msg.latitude
        lon = msg.longitude
        if self.lat0 is None:
            self.lat0 = lat
            self.lon0 = lon
            self.cos_lat0 = math.cos(math.radians(lat))
            self.current_xy = (0.0, 0.0)
        else:
            self.current_xy = self.latlon_to_xy(lat, lon)

    def latlon_to_xy(self, lat: float, lon: float):
        dx = (lon - self.lon0) * (111320.0 * self.cos_lat0)
        dy = (lat - self.lat0) * 110540.0
        return dx, dy

    def obs_cb(self, msg: PoseArray):
        self.obstacles = [(p.position.x, p.position.y) for p in msg.poses]

    def timer_cb(self):
        if self.current_xy is None or len(self.global_xy) < 2:
            return

        x, y = self.current_xy
        idx = self.find_nearest_index(x, y, self.global_xy)
        if idx is None:
            return

        if idx + 1 < len(self.global_xy):
            x2, y2 = self.global_xy[idx + 1]
        else:
            x2, y2 = self.global_xy[idx]
        yaw = math.atan2(y2 - y, x2 - x)

        side = self.decide_side(self.obstacles)

        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map"

        s = 0.0
        prev_px, prev_py = x, y
        i = idx
        while i < len(self.global_xy) and s <= self.L:
            gx, gy = self.global_xy[i]

            seg = math.hypot(gx - prev_px, gy - prev_py)
            s += seg
            prev_px, prev_py = gx, gy

            px, py = gx, gy
            if side != 0:
                offset = self.max_offset * math.sin(math.pi * min(s, self.L) / self.L)
                nx = -math.sin(yaw)
                ny = math.cos(yaw)
                px += offset * side * nx
                py += offset * side * ny

            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(px)
            ps.pose.position.y = float(py)
            ps.pose.position.z = 0.0
            path.poses.append(ps)

            i += 1

        self.pub_local.publish(path)

    def find_nearest_index(self, x: float, y: float, pts: List[Tuple[float, float]]):
        min_d = float("inf")
        idx = None
        for i, (px, py) in enumerate(pts):
            d = (px - x) ** 2 + (py - y) ** 2
            if d < min_d:
                min_d = d
                idx = i
        return idx

    def decide_side(self, obs_xy: List[Tuple[float, float]]) -> int:
        if not obs_xy:
            return 0
        front = []
        for ox, oy in obs_xy:
            if 0.0 < ox < self.L and abs(oy) < self.safe_lat:
                front.append((ox, oy))
        if not front:
            return 0
        front.sort(key=lambda p: p[0])
        _, y_local = front[0]
        return +1 if y_local < 0.0 else -1  # obstacle right -> go left


def main(args=None):
    rclpy.init(args=args)
    node = LocalPathAvoid()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
