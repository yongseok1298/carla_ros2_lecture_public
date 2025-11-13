#!/usr/bin/env python3
# lane_window_fit_node.py
import math, cv2, numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

class LaneWindowFit(Node):
    def __init__(self):
        super().__init__('lane_window_fit')

        # ---------- Params ----------
        self.declare_parameter('image_topic', '/carla/hero/camera_semantic_segmentation/image_color')
        self.declare_parameter('use_colorized', True)         # True: HSV, False: 라벨맵 ID
        self.declare_parameter('hsv_low',  [35, 30, 60])      # 연두 라인
        self.declare_parameter('hsv_high', [90,255,255])
        self.declare_parameter('lane_ids', [7, 8])            # 라벨맵 ID 사용시

        # ROI & BEV (트랩 좌/우 하부 x 비율, 상부 x 비율, 상/하 y 비율)
        self.declare_parameter('trap_bottom_left',  0.13)
        self.declare_parameter('trap_bottom_right', 0.87)
        self.declare_parameter('trap_top_left',     0.40)
        self.declare_parameter('trap_top_right',    0.60)
        self.declare_parameter('trap_top_y',        0.55)     # 화면 상부(0~1)
        self.declare_parameter('trap_bottom_y',     0.95)     # 화면 하부(0~1)
        self.declare_parameter('bev_width',  640)
        self.declare_parameter('bev_height', 480)

        # Sliding window
        self.declare_parameter('n_windows', 16)
        self.declare_parameter('margin_px', 55)      # 윈도우 반폭 (점선 대비 넓게)
        self.declare_parameter('minpix',  30)        # 다음 윈도우 리센터 임계
        self.declare_parameter('hist_frac', 0.5)     # 바닥 히스토그램 높이 영역 비율

        # Poly & smoothing
        self.declare_parameter('poly_order', 2)      # 2 or 3
        self.declare_parameter('ema_alpha', 0.4)     # 프레임간 EMA
        self.declare_parameter('m_per_pix_x', 0.01)  # BEV 가로 scale[m/px]
        self.declare_parameter('m_per_pix_y', 0.01)  # BEV 세로 scale[m/px]

        # Read params
        self.use_colorized = self.get_parameter('use_colorized').get_parameter_value().bool_value
        self.hsv_low  = np.array(self.get_parameter('hsv_low').get_parameter_value().integer_array_value or [35,30,60], dtype=np.uint8)
        self.hsv_high = np.array(self.get_parameter('hsv_high').get_parameter_value().integer_array_value or [90,255,255], dtype=np.uint8)
        self.lane_ids = list(self.get_parameter('lane_ids').get_parameter_value().integer_array_value or [7,8])
        self.bw = int(self.get_parameter('bev_width').get_parameter_value().integer_value or 640)
        self.bh = int(self.get_parameter('bev_height').get_parameter_value().integer_value or 480)
        self.nw = int(self.get_parameter('n_windows').get_parameter_value().integer_value or 12)
        self.margin = int(self.get_parameter('margin_px').get_parameter_value().integer_value or 60)
        self.minpix = int(self.get_parameter('minpix').get_parameter_value().integer_value or 40)
        self.hist_frac = float(self.get_parameter('hist_frac').get_parameter_value().double_value or 0.5)
        self.poly_order = int(self.get_parameter('poly_order').get_parameter_value().integer_value or 2)
        self.alpha = float(self.get_parameter('ema_alpha').get_parameter_value().double_value or 0.3)
        self.mx = float(self.get_parameter('m_per_pix_x').get_parameter_value().double_value or 0.01)
        self.my = float(self.get_parameter('m_per_pix_y').get_parameter_value().double_value or 0.01)

        self.tb_l = float(self.get_parameter('trap_bottom_left').get_parameter_value().double_value)
        self.tb_r = float(self.get_parameter('trap_bottom_right').get_parameter_value().double_value)
        self.tt_l = float(self.get_parameter('trap_top_left').get_parameter_value().double_value)
        self.tt_r = float(self.get_parameter('trap_top_right').get_parameter_value().double_value)
        self.ty   = float(self.get_parameter('trap_top_y').get_parameter_value().double_value)
        self.by   = float(self.get_parameter('trap_bottom_y').get_parameter_value().double_value)

        # ROS I/O
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        img_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.sub = self.create_subscription(Image, img_topic, self.cb, qos)
        self.pub_center = self.create_publisher(PointStamped, '/carla/lane/center', 10)
        self.pub_dbg    = self.create_publisher(Image, '/carla/lane/debug_image', 1)
        self.bridge = CvBridge()

        # temporal memory (EMA)
        self.prev_left  = None   # poly coef
        self.prev_right = None

        self.get_logger().info(f'[LaneWindowFit] subscribe: {img_topic}')
        self.get_logger().info('[LaneWindowFit] publish  : /carla/lane/center, /carla/lane/debug_image')

    # -------- utils --------
    def _warp_perspective(self, img):
        H, W = img.shape[:2]
        src = np.float32([
            [W*self.tb_l, H*self.by],
            [W*self.tb_r, H*self.by],
            [W*self.tt_r, H*self.ty],
            [W*self.tt_l, H*self.ty]
        ])
        dst = np.float32([
            [0, self.bh],
            [self.bw, self.bh],
            [self.bw, 0],
            [0, 0]
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(img, M, (self.bw, self.bh), flags=cv2.INTER_NEAREST)
        return warped, M, Minv

    def _binary_from_seg(self, img):
        """Semantic seg -> binary lane mask"""
        if self.use_colorized:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)
        else:
            raw = img if img.ndim == 2 else img[:,:,2]
            mask = np.zeros_like(raw, dtype=np.uint8)
            for tid in self.lane_ids:
                mask |= (raw == tid).astype(np.uint8) * 255
        # morphology
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
        return mask

    def _sliding_window_fit(self, bin_bev):
        h, w = bin_bev.shape
        # histogram(half of bottom)
        hist = np.sum(bin_bev[int(h*(1-self.hist_frac)):,:]//255, axis=0)
        midpoint = w//2
        leftx_base  = np.argmax(hist[:midpoint]) if hist[:midpoint].max()>0 else int(w*0.25)
        rightx_base = np.argmax(hist[midpoint:]) + midpoint if hist[midpoint:].max()>0 else int(w*0.75)

        window_height = int(h / self.nw)
        nonzero = bin_bev.nonzero()
        nonzeroy = np.array(nonzero[0]); nonzerox = np.array(nonzero[1])

        leftx_current, rightx_current = leftx_base, rightx_base
        left_lane_inds = []; right_lane_inds = []

        for win in range(self.nw):
            win_y_low  = h - (win+1)*window_height
            win_y_high = h - win*window_height
            win_xleft_low  = max(leftx_current - self.margin, 0)
            win_xleft_high = min(leftx_current + self.margin, w)
            win_xright_low  = max(rightx_current - self.margin, 0)
            win_xright_high = min(rightx_current + self.margin, w)

            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)

            # 점선 대비: 픽셀이 적으면 과거 위치 유지
            if len(good_left) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left]))
            elif self.prev_left is not None:
                # 이전 예측값으로 re-centering
                y_mid = (win_y_low + win_y_high)//2
                px = np.polyval(self.prev_left, y_mid)
                leftx_current = int(px)

            if len(good_right) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right]))
            elif self.prev_right is not None:
                y_mid = (win_y_low + win_y_high)//2
                px = np.polyval(self.prev_right, y_mid)
                rightx_current = int(px)

        left_lane_inds  = np.concatenate(left_lane_inds)  if len(left_lane_inds)  else np.array([])
        right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) else np.array([])

        leftx  = nonzerox[left_lane_inds];  lefty  = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]; righty = nonzeroy[right_lane_inds]

        # 최소 포인트 미만이면 실패 처리
        if len(leftx) < 200 or len(rightx) < 200:
            return None, None, hist

        #Quadratic or cubic polynomial fitting
        order = 3 if self.poly_order >= 3 else 2
        left_fit  = np.polyfit(lefty,  leftx,  order)
        right_fit = np.polyfit(righty, rightx, order)

        # EMA smoothing
        if self.prev_left is None:  self.prev_left  = left_fit
        else:                        self.prev_left  = self.alpha*left_fit  + (1-self.alpha)*self.prev_left
        if self.prev_right is None: self.prev_right = right_fit
        else:                        self.prev_right = self.alpha*right_fit + (1-self.alpha)*self.prev_right

        return self.prev_left, self.prev_right, hist

    # -------- callback --------
    def cb(self, msg: Image):
        # seg -> cv2
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8' if self.use_colorized else 'passthrough')
        H, W = img.shape[:2]
        view = img.copy() if img.ndim==3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # binary mask
        mask = self._binary_from_seg(view if self.use_colorized else img)

        # BEV warp
        bev, M, Minv = self._warp_perspective(mask)
        bev_bin = (bev > 0).astype(np.uint8)*255

        # sliding window + poly fit
        lfit, rfit, hist = self._sliding_window_fit(bev_bin)

        # center/angle & debug draw
        offset_m, angle_rad = float('nan'), 0.0
        dbg = cv2.cvtColor(bev_bin, cv2.COLOR_GRAY2BGR)

        if lfit is not None and rfit is not None:
            y_vals = np.linspace(0, self.bh-1, self.bh)
            lx = np.polyval(lfit, y_vals)
            rx = np.polyval(rfit, y_vals)

            # 차선 중앙
            cx = (lx + rx) / 2.0
            cy = y_vals

            # 차량 기준 중앙과의 오프셋(하단 기준)
            lane_cx_px = cx[-1]
            offset_px = lane_cx_px - (self.bw/2)
            offset_m  = offset_px * self.mx

            # 진행 각: 하단 구간의 기울기
            idx2 = int(self.bh*0.9); idx1 = int(self.bh*0.7)
            if idx2>idx1:
                dy = (idx2-idx1)*self.my
                dx = (cx[idx2]-cx[idx1])*self.mx
                angle_rad = math.atan2(dy, dx) - math.pi/2.0  # 카메라 축 기준 조정

            pts_l = np.int32(np.column_stack([lx, y_vals]))
            pts_r = np.int32(np.column_stack([rx, y_vals]))
            for p in pts_l: cv2.circle(dbg, tuple(p), 1, (255,0,0), -1)
            for p in pts_r: cv2.circle(dbg, tuple(p), 1, (0,0,255), -1)
            for p in np.int32(np.column_stack([cx, y_vals]))[::8]:
                cv2.circle(dbg, tuple(p), 2, (0,255,255), -1)


        # inverse warp for overlay on original
        overlay = cv2.warpPerspective(dbg, Minv, (W, H))

        # original view+ overlay -> result
        out = cv2.addWeighted(view, 0.7, overlay, 0.6, 0)

        # drawing
        H, W = out.shape[:2]
        txt = f'offset={offset_m:.2f} m, angle={math.degrees(angle_rad):.1f} deg'

        scale = max(0.8, min(2.0, W/900.0))
        thick = max(2, int(2*scale))
        pad   = int(14*scale)

        (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x, y = pad, pad + th

        overlay2 = out.copy()
        cv2.rectangle(overlay2, (x - pad//2, y - th - pad//2),
                                (x + tw + pad//2, y + bl + pad//2),
                                (0, 0, 0), -1)
        out = cv2.addWeighted(overlay2, 0.5, out, 0.5, 0)

        cv2.putText(out, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)

        # publish
        imsg = self.bridge.cv2_to_imgmsg(out, encoding='bgr8')
        imsg.header = msg.header
        self.pub_dbg.publish(imsg)

def main():
    rclpy.init()
    node = LaneWindowFit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
