#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class GlobalPathFromGNSS(Node):
    def __init__(self):
        super().__init__("global_path_from_gnss")
        self.sub = self.create_subscription(
            NavSatFix, "/carla/hero/gnss", self.gnss_cb, 10)
        self.pub = self.create_publisher(Path, "/carla/path/global", 10)

        self.lat0 = None
        self.lon0 = None
        self.cos_lat0 = 1.0
        self.path_xy = []  # list of (x, y)
        self.min_dist = 0.5  # [m]

        self.timer = self.create_timer(0.2, self.timer_cb)

    def gnss_cb(self, msg: NavSatFix):
        lat = msg.latitude
        lon = msg.longitude

        if self.lat0 is None:
            self.lat0 = lat
            self.lon0 = lon
            self.cos_lat0 = math.cos(math.radians(lat))
            x, y = 0.0, 0.0
        else:
            x, y = self.latlon_to_xy(lat, lon)

        if not self.path_xy:
            self.path_xy.append((x, y))
            return

        last_x, last_y = self.path_xy[-1]
        if math.hypot(x - last_x, y - last_y) >= self.min_dist:
            self.path_xy.append((x, y))

    def latlon_to_xy(self, lat: float, lon: float):
        dx = (lon - self.lon0) * (111320.0 * self.cos_lat0)
        dy = (lat - self.lat0) * 110540.0
        return dx, dy

    def timer_cb(self):
        if not self.path_xy:
            return
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map"
        for (x, y) in self.path_xy:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            path.poses.append(ps)
        self.pub.publish(path)

    def save_path_to_file(self, filename: str = "global_path.csv"):
        if not self.path_xy:
            self.get_logger().warn("No path to save.")
            return
        try:
            with open(filename, "w") as f:
                # optional header
                f.write("# origin_lat,origin_lon\n")
                f.write(f"{self.lat0},{self.lon0}\n")
                f.write("# x[m],y[m]\n")
                for x, y in self.path_xy:
                    f.write(f"{x},{y}\n")
            self.get_logger().info(f"Saved global path to {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save path: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = GlobalPathFromGNSS()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    # save on shutdown
    node.save_path_to_file("global_path.csv")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
