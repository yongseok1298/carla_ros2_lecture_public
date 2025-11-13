#!/usr/bin/env python3
# lidar_obstacles_simple.py
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2

class LidarObstaclesSimple(Node):
    def __init__(self):
        super().__init__('lidar_obstacles_simple')
        self.sub = self.create_subscription(
            PointCloud2, '/carla/hero/lidar/point_cloud', self.cb, 1)
        self.pub_pose = self.create_publisher(PoseArray, '/carla/obstacles_2d', 1)
        self.pub_mk   = self.create_publisher(MarkerArray, '/carla/obstacles_markers', 1)

        self.X_MAX = 35.0   # 전방 m
        self.Y_ABS = 6.0    # 좌우 m
        self.Z_MIN, self.Z_MAX = -3.0, 3.0
        self.VOX = 0.25     # voxel 크기(m)
        self.MIN_PTS = 8    # 최소 포인트 수
        self.GROUND_PCT = 12.0  # 지면 추정 하위 백분위
        self.GROUND_BAND_X = (2.0, 8.0)  # 지면 추정 대역
        self.GROUND_BAND_Y = 3.0
        self.GROUND_EPS = 0.18
        self.inflate_r = 0.7

    def cb(self, msg: PointCloud2):
        try:
            pts = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True).astype(np.float32, copy=False)
        except Exception:
            pts = np.asarray(list(pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True)),
                        dtype=np.float32)
        if pts.size == 0:
            return
        # ROI
        m = (pts[:,0] >= 0.0) & (pts[:,0] <= self.X_MAX) & \
            (np.abs(pts[:,1]) <= self.Y_ABS) & \
            (pts[:,2] >= self.Z_MIN) & (pts[:,2] <= self.Z_MAX)
        pts = pts[m]
        if pts.size == 0:
            return

        # 지면 z 추정
        band = pts[(pts[:,0] >= self.GROUND_BAND_X[0]) & (pts[:,0] <= self.GROUND_BAND_X[1]) &
                   (np.abs(pts[:,1]) <= self.GROUND_BAND_Y)]
        gz = np.percentile(band[:,2], self.GROUND_PCT) if band.size else -1.8
        pts = pts[pts[:,2] > gz + self.GROUND_EPS]
        if pts.size == 0:
            self._publish([], msg); return

        # 2D grid + 8-connected
        gx = np.floor(pts[:,0] / self.VOX).astype(np.int32)
        gy = np.floor((pts[:,1] + self.Y_ABS) / self.VOX).astype(np.int32)
        grid = {}
        for ix, iy, (x,y) in zip(gx, gy, pts[:, :2]):
            grid.setdefault((ix,iy), []).append((x,y))
        visited=set(); neigh=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        clusters=[]
        for key in list(grid.keys()):
            if key in visited: continue
            stack=[key]; visited.add(key); cells=[]
            while stack:
                k=stack.pop(); cells.append(k)
                ix,iy=k
                for dx,dy in neigh:
                    nk=(ix+dx,iy+dy)
                    if nk in grid and nk not in visited:
                        visited.add(nk); stack.append(nk)
            cpts=[]
            for c in cells: cpts.extend(grid[c])
            if len(cpts) >= self.MIN_PTS:
                clusters.append(np.array(cpts, dtype=np.float32))

        # center/size -> list
        obs=[]
        for c in clusters:
            cx, cy = np.mean(c, axis=0)
            minx, miny = np.min(c, axis=0); maxx, maxy = np.max(c, axis=0)
            w = max(0.3, (maxx-minx)); h = max(0.3, (maxy-miny))
            r = max(self.inflate_r, 0.5*max(w,h))
            obs.append((cx, cy, w, h, r))

        self._publish(obs, msg)

    def _publish(self, obs, src):
        pa = PoseArray(); pa.header = src.header
        ma = MarkerArray(); mid=0
        for (cx,cy,w,h,r) in obs:
            p = Pose(); p.position.x=float(cx); p.position.y=float(cy); p.orientation.w=1.0
            pa.poses.append(p)

            m = Marker(); m.header=pa.header; m.ns='obs'; m.id=mid; mid+=1
            m.type=Marker.CUBE; m.action=Marker.ADD
            m.pose.position.x=float(cx); m.pose.position.y=float(cy); m.pose.orientation.w=1.0
            m.scale.x=float(w); m.scale.y=float(h); m.scale.z=0.2
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 0.6
            ma.markers.append(m)

            c = Marker(); c.header=pa.header; c.ns='obs'; c.id=mid; mid+=1
            c.type=Marker.CYLINDER; c.action=Marker.ADD
            c.pose.position.x=float(cx); c.pose.position.y=float(cy); c.pose.orientation.w=1.0
            c.scale.x=c.scale.y=float(2.0*r); c.scale.z=0.05
            c.color.r, c.color.g, c.color.b, c.color.a = 1.0, 0.0, 0.0, 0.4
            ma.markers.append(c)

        self.pub_pose.publish(pa)
        self.pub_mk.publish(ma)

def main():
    rclpy.init()
    n=LidarObstaclesSimple()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.shutdown()
if __name__=='__main__': main()
