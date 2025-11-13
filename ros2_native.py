#!/usr/bin/env python3

import argparse
import json
import logging

import numpy as np
import carla
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist


class DepthColorizer:
    """Convert depth image to colored image and publish as ROS Image."""

    def __init__(self, node: Node, topic: str):
        self.node = node
        self.bridge = CvBridge()
        self.pub = node.create_publisher(RosImage, topic, 10)
        self.node.get_logger().info(f"[DepthColorizer] publish -> {topic}")

    def handle(self, image: carla.Image):
        image.convert(carla.ColorConverter.LogarithmicDepth)
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        msg = self.bridge.cv2_to_imgmsg(arr, encoding="bgr8")
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "camera_depth"
        self.pub.publish(msg)


class SemanticColorizer:
    """Convert semantic segmentation image to colored image and publish as ROS Image."""

    def __init__(self, node: Node, topic: str):
        self.node = node
        self.bridge = CvBridge()
        self.pub = node.create_publisher(RosImage, topic, 10)
        self.node.get_logger().info(f"[SemanticColorizer] publish -> {topic}")

    def handle(self, image: carla.Image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        msg = self.bridge.cv2_to_imgmsg(arr, encoding="bgr8")
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "camera_semantic"
        self.pub.publish(msg)


def show_spectator(image: carla.Image):
    """Show spectator camera with OpenCV (800x600)."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
    view = cv2.resize(arr, (800, 600))
    cv2.imshow("spectator", view)
    cv2.waitKey(1)


def _setup_vehicle(world, config):
    logging.debug("Spawning vehicle: {}".format(config.get("type")))

    bp_library = world.get_blueprint_library()
    map_ = world.get_map()

    bp = bp_library.filter(config.get("type"))[0]
    bp.set_attribute("role_name", config.get("id"))
    bp.set_attribute("ros_name", config.get("id"))

    return world.spawn_actor(
        bp,
        map_.get_spawn_points()[0],
        attach_to=None
    )


def _setup_sensors(
    world,
    vehicle,
    sensors_config,
    depth_colorizer: DepthColorizer = None,
    semantic_colorizer: SemanticColorizer = None,
):
    bp_library = world.get_blueprint_library()

    sensors = []
    for sensor in sensors_config:
        logging.debug("Spawning sensor: {}".format(sensor))

        bp = bp_library.filter(sensor.get("type"))[0]
        bp.set_attribute("ros_name", sensor.get("id"))
        bp.set_attribute("role_name", sensor.get("id"))
        for key, value in sensor.get("attributes", {}).items():
            bp.set_attribute(str(key), str(value))

        wp = carla.Transform(
            location=carla.Location(
                x=sensor["spawn_point"]["x"],
                y=-sensor["spawn_point"]["y"],
                z=sensor["spawn_point"]["z"]
            ),
            rotation=carla.Rotation(
                roll=sensor["spawn_point"]["roll"],
                pitch=-sensor["spawn_point"]["pitch"],
                yaw=-sensor["spawn_point"]["yaw"]
            )
        )

        actor = world.spawn_actor(bp, wp, attach_to=vehicle)
        actor.enable_for_ros()

        if depth_colorizer is not None and sensor.get("type") == "sensor.camera.depth":
            actor.listen(lambda img, dc=depth_colorizer: dc.handle(img))

        if semantic_colorizer is not None and sensor.get("type") == "sensor.camera.semantic_segmentation":
            actor.listen(lambda img, sc=semantic_colorizer: sc.handle(img))

        if sensor.get("id") == "spectator" and sensor.get("id") != "camera_front" and sensor.get("type").startswith("sensor.camera.rgb"):
            actor.listen(lambda img: show_spectator(img))

        sensors.append(actor)

    return sensors


def main(args):

    world = None
    vehicle = None
    sensors = []
    original_settings = None

    rclpy.init(args=None)
    node = rclpy.create_node("carla_ros2_depth_bridge")

    depth_colorizer = DepthColorizer(node, "/carla/hero/camera_depth/image_depth")
    semantic_colorizer = SemanticColorizer(
        node,
        "/carla/hero/camera_semantic_segmentation/image_color"
    )

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        world = client.get_world()

        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # traffic_manager = client.get_trafficmanager()
        # traffic_manager.set_synchronous_mode(True)

        with open(args.file) as f:
            config = json.load(f)

        vehicle = _setup_vehicle(world, config)
        sensors = _setup_sensors(
            world,
            vehicle,
            config.get("sensors", []),
            depth_colorizer=depth_colorizer,
            semantic_colorizer=semantic_colorizer,
        )

        _ = world.tick()
        # vehicle.set_autopilot(False)

        MAX_STEER_DEG = 35.0    # Model 3 Tire Angle 
        KP_THR        = 0.25    # P acceleration gain
        KP_BRK        = 0.35    # P brake gain

        def _speed_mps(vec):
            return float((vec.x**2 + vec.y**2 + vec.z**2) ** 0.5)

        def _clamp(v, lo, hi):
            return hi if v > hi else lo if v < lo else v

        # === Twist(linear.x[m/s], angular.z[deg], linear.y[0..1]) -> VehicleControl ===
        def _on_cmd(msg: Twist):
            v_meas = _speed_mps(vehicle.get_velocity()) # current velocity [m/s]

            v_ref_mps   = float(msg.linear.x)     # target speed [m/s]
            steer_deg   = float(msg.angular.z)    # [deg] (left: -, right: +)
            brake_manual= _clamp(float(msg.linear.y), 0.0, 1.0)  # brake (optional)

            steer_norm = _clamp(steer_deg / MAX_STEER_DEG, -1.0, 1.0)

            err = v_ref_mps - v_meas
            thr_cmd = _clamp(KP_THR * max(err,  0.0), 0.0, 1.0)
            brk_cmd = _clamp(KP_BRK * max(-err, 0.0), 0.0, 1.0)

            if brake_manual > 0.0:
                brk_cmd = brake_manual
                thr_cmd = 0.0

            vehicle.apply_control(carla.VehicleControl(
                throttle=thr_cmd,
                brake=brk_cmd,
                steer=steer_norm
            ))

        node.create_subscription(Twist, '/carla/hero/cmd_vel', _on_cmd, 10)
        node.get_logger().info("[control] Subscribed /carla/hero/cmd_vel (x=thr, y=brk, z=steer)")


        logging.info("Running...")

        while rclpy.ok():
            _ = world.tick()
            rclpy.spin_once(node, timeout_sec=0.0)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

    finally:
        if original_settings:
            world.apply_settings(original_settings)

        for sensor in sensors:
            sensor.destroy()

        if vehicle:
            vehicle.destroy()

        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='CARLA ROS2 native (with depth & semantic image & spectator view)')
    argparser.add_argument('--host', metavar='H', default='localhost',
                           help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument('--port', metavar='P', default=2000, type=int,
                           help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument('-f', '--file', default='', required=True,
                           help='File to be executed (e.g. original_tesla.json)')
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug',
                           help='print debug information')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('Listening to server %s:%s', args.host, args.port)

    main(args)