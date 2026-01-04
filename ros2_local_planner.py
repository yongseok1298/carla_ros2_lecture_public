#!/usr/bin/env python3
# ros2_local_planner.py (improved: frame-consistent + simple lattice candidates)

import math
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseArray, PoseStamped


def quat_to_yaw(q):
    # yaw from quaternion (z-w plane)
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def rot2d(x, y, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (c * x - s * y, s * x + c * y)


class LocalPathAvoid(Node):
    """
    - Global path: CSV(x,y) assumed in CARLA map frame (same as /carla/hero/odometry pose frame)
    - Obstacles: PoseArray positions assumed in HERO(vehicle) frame (x forward, y left)
    - Output: Path in 'map' frame (consistent with RViz fixed frame 'map')
    - Simple lattice: generate 3 candidates (center / left / right), choose safest against obstacles.
    """

    def __init__(self):
        super().__init__("local_path_avoid")

        # ---- Params ----
        self.declare_parameter("global_path_csv", "global_path.csv")
        self.declare_parameter("output_frame", "map")  # keep 'map' for RViz
        self.declare_parameter("L", 20.0)              # lookahead length [m]
        self.declare_parameter("ds", 0.5)              # (not strictly used; CSV spacing assumed)
        self.declare_parameter("safe_lat", 2.0)        # obstacle y-range (hero frame) to consider
        self.declare_parameter("max_offset", 3.0)      # lateral max offset [m]
        self.declare_parameter("n_candidates", 3)      # 1 or 3 (center, left, right)
        self.declare_parameter("collision_radius", 1.5)  # [m] for scoring

        self.L = float(self.get_parameter("L").value)
        self.safe_lat = float(self.get_parameter("safe_lat").value)
        self.max_offset = float(self.get_parameter("max_offset").value)
        self.n_candidates = int(self.get_parameter("n_candidates").value)
        self.collision_r = float(self.get_parameter("collision_radius").value)
        self.output_frame = str(self.get_parameter("output_frame").value)

        # ---- State ----
        self.current_xy_yaw: Optional[Tuple[float, float, float]] = None  # map frame
        self.global_xy: List[Tuple[float, float]] = []
        self.obstacles_hero: List[Tuple[float, float]] = []  # hero frame (x,y)

        # ---- ROS I/O ----
        self.sub_odom = self.create_subscription(
            Odometry, "/carla/hero/odometry", self.odom_cb, 10
        )
        self.sub_obs = self.create_subscription(
            PoseArray, "/carla/obstacles_2d", self.obs_cb, 10
        )
        self.pub_local = self.create_publisher(Path, "/carla/path/local", 10)

        # ---- Load global path ----
        csv_path = str(self.get_parameter("global_path_csv").value)
        self.load_global_path(csv_path)
        if not self.global_xy:
            self.get_logger().warn(f"{csv_path} is empty or not found.")

        self.timer = self.create_timer(0.1, self.timer_cb)

    def load_global_path(self, filename: str):
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

    def odom_cb(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        yaw = quat_to_yaw(msg.pose.pose.orientation)
        self.current_xy_yaw = (px, py, yaw)

    def obs_cb(self, msg: PoseArray):
        # obstacles in HERO frame (x forward, y left)
        self.obstacles_hero = [(p.position.x, p.position.y) for p in msg.poses]

    def timer_cb(self):
        if self.current_xy_yaw is None or len(self.global_xy) < 2:
            return

        ex, ey, eyaw = self.current_xy_yaw

        idx = self.find_nearest_index(ex, ey, self.global_xy)
        if idx is None:
            return

        # heading from global path local segment (map frame)
        if idx + 1 < len(self.global_xy):
            x2, y2 = self.global_xy[idx + 1]
        else:
            x2, y2 = self.global_xy[idx]
        path_yaw = math.atan2(y2 - ey, x2 - ex)

        # obstacles -> map frame for scoring
        obs_map = []
        for ox_h, oy_h in self.obstacles_hero:
            # consider only those in front-ish and near lateral in HERO frame
            if 0.0 < ox_h < self.L and abs(oy_h) < self.safe_lat:
                dx_m, dy_m = rot2d(ox_h, oy_h, eyaw)  # rotate by ego yaw into map
                obs_map.append((ex + dx_m, ey + dy_m))

        # candidate sides: center(0), left(+1), right(-1)
        sides = [0]
        if self.n_candidates >= 3:
            sides = [0, +1, -1]

        best_path = None
        best_score = -1.0

        for side in sides:
            cand = self.build_offset_path(idx, ex, ey, path_yaw, side)
            score = self.score_path(cand, obs_map)
            if score > best_score:
                best_score = score
                best_path = cand

        if best_path is None:
            return

        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.output_frame  # 'map'

        for (px, py) in best_path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(px)
            ps.pose.position.y = float(py)
            ps.pose.position.z = 0.0
            msg.poses.append(ps)

        self.pub_local.publish(msg)

    def build_offset_path(self, start_idx: int, ex: float, ey: float, yaw: float, side: int):
        """
        Build path by walking global points and adding smooth lateral offset.
        """
        pts = []
        s = 0.0

        prev_x, prev_y = ex, ey
        i = start_idx

        # normal vector in map frame (left normal)
        nx = -math.sin(yaw)
        ny = math.cos(yaw)

        while i < len(self.global_xy) and s <= self.L:
            gx, gy = self.global_xy[i]
            seg = math.hypot(gx - prev_x, gy - prev_y)
            s += seg
            prev_x, prev_y = gx, gy

            px, py = gx, gy
            if side != 0:
                # smooth "bump" offset: 0 -> max -> 0
                bump = math.sin(math.pi * min(s, self.L) / self.L)
                offset = self.max_offset * bump
                px += offset * side * nx
                py += offset * side * ny

            pts.append((px, py))
            i += 1

        return pts

    def score_path(self, path_xy: List[Tuple[float, float]], obs_map: List[Tuple[float, float]]):
        """
        Simple safety score: penalize if any point comes too close to any obstacle.
        Higher is better. If collision, score very low.
        """
        if not obs_map:
            return 1000.0  # free

        min_d = float("inf")
        for px, py in path_xy:
            for ox, oy in obs_map:
                d = math.hypot(px - ox, py - oy)
                if d < min_d:
                    min_d = d

        if min_d < self.collision_r:
            return -1000.0 + min_d  # collision-ish

        # prefer larger clearance
        return min_d

    def find_nearest_index(self, x: float, y: float, pts: List[Tuple[float, float]]):
        min_d = float("inf")
        idx = None
        for i, (px, py) in enumerate(pts):
            d = (px - x) ** 2 + (py - y) ** 2
            if d < min_d:
                min_d = d
                idx = i
        return idx


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
