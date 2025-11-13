#!/usr/bin/env python3
import os, time, threading
import cv2, numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D
from cv_bridge import CvBridge
import torch

class Yolo2DDetector(Node):
    def __init__(self):
        super().__init__('yolo_2d_detector')

        self.declare_parameter('image_topic', '/carla/hero/camera_front/image')
        self.declare_parameter('model_path', 'carla_yolov8n.pt')
        self.declare_parameter('out_topic', '/carla/object_detection_2d/bounding_box')
        self.declare_parameter('debug_image_topic', '/carla/object_detection_2d/debug_image')
        self.declare_parameter('publish_debug_image', True)

        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        self.declare_parameter('img_width', 512)    
        self.declare_parameter('img_height', 320)
        self.declare_parameter('process_every_n', 2)  # process every n frame
        self.declare_parameter('use_half', True)      # cuda FP16
        self.declare_parameter('drop_if_busy', True) # frame drop

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        model_path  = self.get_parameter('model_path').get_parameter_value().string_value
        out_topic   = self.get_parameter('out_topic').get_parameter_value().string_value
        dbg_topic   = self.get_parameter('debug_image_topic').get_parameter_value().string_value
        self.pub_dbg_enabled = self.get_parameter('publish_debug_image').get_parameter_value().bool_value

        self.conf_th = float(self.get_parameter('conf_threshold').get_parameter_value().double_value)
        self.iou_th  = float(self.get_parameter('iou_threshold').get_parameter_value().double_value)
        self.device  = self.get_parameter('device').get_parameter_value().string_value
        self.img_w   = int(self.get_parameter('img_width').get_parameter_value().integer_value)
        self.img_h   = int(self.get_parameter('img_height').get_parameter_value().integer_value)
        self.proc_n  = int(self.get_parameter('process_every_n').get_parameter_value().integer_value)
        self.use_half = bool(self.get_parameter('use_half').get_parameter_value().bool_value)
        self.drop_if_busy = bool(self.get_parameter('drop_if_busy').get_parameter_value().bool_value)

        try:
            from ultralytics import YOLO
        except Exception as e:
            self.get_logger().error("pip install ultralytics")
            raise e

        if not os.path.exists(model_path):
            self.get_logger().warn(f'model not found: {model_path}')

        self.get_logger().info(f'Loading YOLO model: {model_path} (device={self.device})')
        t0 = time.time()
        self.model = YOLO(model_path)
        self.model.to(self.device)

        if self.device.startswith('cuda') and self.use_half:
            try:
                self.model.model.half()
                self.get_logger().info('Using half precision (FP16).')
            except Exception:
                self.get_logger().warn('FP16 failed. FP32 will be used.')

        torch.set_grad_enabled(False)
        try:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
        except Exception:
            pass

        try:
            _ = self.model.predict(
                np.zeros((max(1,self.img_h), max(1,self.img_w), 3), dtype=np.uint8),
                conf=self.conf_th, iou=self.iou_th, device=self.device, verbose=False
            )
        except Exception:
            pass
        self.get_logger().info(f'Model loaded in {time.time() - t0:.2f}s')

        self.class_names = None
        try:
            self.class_names = self.model.model.names if hasattr(self.model, 'model') else self.model.names
        except Exception:
            pass

        # -------- ROS IO --------
        self.bridge = CvBridge()
        self.frame_count = 0
        self.busy = False
        self.lock = threading.Lock()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.sub = self.create_subscription(Image, image_topic, self.image_callback, sensor_qos)
        self.pub_det = self.create_publisher(Detection2DArray, out_topic, QoSProfile(depth=10))
        self.pub_dbg = self.create_publisher(Image, dbg_topic, QoSProfile(depth=1)) if self.pub_dbg_enabled else None

        self.get_logger().info(f'Subscribed: {image_topic}')
        self.get_logger().info(f'Publishing: {out_topic} (vision_msgs/Detection2DArray)')
        if self.pub_dbg_enabled:
            self.get_logger().info(f'Publishing: {dbg_topic} (sensor_msgs/Image, bgr8)')

    def image_callback(self, msg: Image):
        # throttle
        self.frame_count += 1
        if self.proc_n > 1 and (self.frame_count % self.proc_n) != 0:
            return

        if self.drop_if_busy:
            with self.lock:
                if self.busy:
                    return
                self.busy = True

        # ROS -> CV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion failed: {e}')
            if self.drop_if_busy:
                with self.lock:
                    self.busy = False
            return

        # resize
        resized = cv_image
        if self.img_w > 0 and self.img_h > 0:
            resized = cv2.resize(cv_image, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)

        # inference
        try:
            with torch.inference_mode():
                results = self.model.predict(
                    resized, conf=self.conf_th, iou=self.iou_th,
                    device=self.device, verbose=False
                )
        except Exception as e:
            self.get_logger().error(f'YOLO failed inference: {e}')
            if self.drop_if_busy:
                with self.lock:
                    self.busy = False
            return

        det_array = Detection2DArray()
        det_array.header = msg.header

        # for debug overlay
        overlay = cv_image.copy() if self.pub_dbg_enabled else None

        if len(results) > 0:
            r0 = results[0]
            if r0.boxes is not None and len(r0.boxes) > 0:
                xyxy = r0.boxes.xyxy.detach().float().cpu().numpy()
                conf = r0.boxes.conf.detach().float().cpu().numpy()
                cls  = r0.boxes.cls.detach().int().cpu().numpy()

                sx = float(cv_image.shape[1]) / float(resized.shape[1])
                sy = float(cv_image.shape[0]) / float(resized.shape[0])

                for i in range(xyxy.shape[0]):
                    x1, y1, x2, y2 = xyxy[i]
                    x1 *= sx; x2 *= sx; y1 *= sy; y2 *= sy
                    w = float(max(0.0, x2 - x1))
                    h = float(max(0.0, y2 - y1))
                    cx = float(x1 + w/2.0); cy = float(y1 + h/2.0)
                    score = float(conf[i]); cid = int(cls[i])

                    det = Detection2D()
                    det.header = msg.header

                    bbox = BoundingBox2D()
                    bbox.center.position.x = cx
                    bbox.center.position.y = cy
                    bbox.size_x = w
                    bbox.size_y = h
                    det.bbox = bbox

                    hyp = ObjectHypothesisWithPose()
                    name = str(self.class_names[cid]) if self.class_names and 0 <= cid < len(self.class_names) else str(cid)
                    hyp.hypothesis.class_id = name
                    hyp.hypothesis.score = score
                    det.results.append(hyp)
                    det_array.detections.append(det)

                    # draw overlay
                    if overlay is not None:
                        p1 = (int(x1), int(y1)); p2 = (int(x2), int(y2))
                        cv2.rectangle(overlay, p1, p2, (0, 255, 0), 2)
                        label = f"{name} {score:.2f}"
                        # text bg
                        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(overlay, (p1[0], max(p1[1]-th-6, 0)), (p1[0]+tw+4, p1[1]), (0,255,0), -1)
                        cv2.putText(overlay, label, (p1[0]+2, max(p1[1]-4, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

        # publish detections
        self.pub_det.publish(det_array)

        # publish debug image
        if self.pub_dbg_enabled and self.pub_dbg is not None:
            try:
                dbg_msg = self.bridge.cv2_to_imgmsg(overlay if overlay is not None else cv_image, encoding='bgr8')
                dbg_msg.header = msg.header
                self.pub_dbg.publish(dbg_msg)
            except Exception as e:
                self.get_logger().warn(f'debug image publish 실패: {e}')

        if self.drop_if_busy:
            with self.lock:
                self.busy = False

def main():
    rclpy.init()
    node = Yolo2DDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
