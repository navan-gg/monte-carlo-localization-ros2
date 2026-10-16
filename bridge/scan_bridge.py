#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import (
    PoseStamped,
    PoseWithCovarianceStamped,
    TransformStamped,
)
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from std_srvs.srv import Empty
from tf2_ros import TransformBroadcaster

import math
import numpy as np
from scipy.ndimage import distance_transform_edt


class ScanBridge(Node):
    def __init__(self):
        super().__init__('scan_bridge')

        self.num_beams = 36
        self.max_range = 15.0
        self.min_range = 0.1

        self.map_data       = None
        self.map_resolution = None
        self.map_origin     = None
        self.map_width      = None
        self.map_height     = None
        self.dist_map       = None

        self.tf_broadcaster = TransformBroadcaster(self)

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(OccupancyGrid, '/map',       self.map_cb,  map_qos)
        self.create_subscription(PoseStamped,   '/true_pose', self.pose_cb, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.amcl_pose_cb, 10
        )

        self.scan_pub      = self.create_publisher(LaserScan,   '/amcl_scan',     10)
        self.odom_pub      = self.create_publisher(PoseStamped, '/odom_pose', 10)
        self.amcl_viz_pub  = self.create_publisher(PoseStamped, '/amcl_pose_viz', 10)

        self.global_loc_client = self.create_client(Empty, '/reinitialize_global_localization')

        self.prev_x     = None
        self.prev_y     = None
        self.prev_theta = None

        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = 0.0

        self._kidnap_pending_reloc = False
        self._kidnap_suppress_cycles = 0
        self.get_logger().info('Scan Bridge started')

    def map_cb(self, msg):
        if self.map_data is not None:
            return
        self.map_data       = msg.data
        self.map_resolution = msg.info.resolution
        self.map_origin     = msg.info.origin
        self.map_width      = msg.info.width
        self.map_height     = msg.info.height

        grid      = np.array(self.map_data, dtype=np.int8).reshape(self.map_height, self.map_width)
        free_mask = (grid == 0).astype(np.float32)
        self.dist_map = distance_transform_edt(free_mask) * self.map_resolution
        self.get_logger().info('Scan Bridge: map loaded, distance map built')

    def _raycast(self, x, y, theta):
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        d = 0.0
        while d < self.max_range:
            tx = x + d * cos_t
            ty = y + d * sin_t
            mx = int((tx - self.map_origin.position.x) / self.map_resolution)
            my = int((ty - self.map_origin.position.y) / self.map_resolution)
            if mx < 0 or my < 0 or mx >= self.map_width or my >= self.map_height:
                return d
            if self.map_data[my * self.map_width + mx] != 0:
                return d
            d += max(0.02, self.dist_map[my, mx] - 0.01)
        return self.max_range

    @staticmethod
    def _wrap(a):
        return math.atan2(math.sin(a), math.cos(a))

    def _call_global_relocalization(self):
        if not self.global_loc_client.service_is_ready():
            self.get_logger().warn('Global localization service not ready')
            return False

        req = Empty.Request()
        future = self.global_loc_client.call_async(req)
        future.add_done_callback(self._global_reloc_done)
        return True

    def _global_reloc_done(self, future):
        try:
            future.result()
            self.get_logger().info('Global relocalization succeeded')
        except Exception as e:
            self.get_logger().warn(f'Global relocalization call failed: {e}')
        finally:
            self._kidnap_pending_reloc = False
            self._kidnap_suppress_cycles = 10

    def pose_cb(self, msg):
        if self.map_data is None:
            return

        x     = msg.pose.position.x
        y     = msg.pose.position.y
        qz    = msg.pose.orientation.z
        qw    = msg.pose.orientation.w
        theta = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
        stamp = msg.header.stamp

        if self.prev_x is None:
            self.prev_x = x
            self.prev_y = y
            self.prev_theta = theta
            return

        dx = x - self.prev_x
        dy = y - self.prev_y
        dtheta = self._wrap(theta - self.prev_theta)
        jump = math.sqrt(dx * dx + dy * dy)
        
        if jump > 1.5:
            if not self._kidnap_pending_reloc and self._kidnap_suppress_cycles == 0:
                self.get_logger().warn(f'Kidnap detected (jump {jump:.1f} m) -- API triggered')
                self._kidnap_pending_reloc = True
                if not self._call_global_relocalization():
                    self._kidnap_pending_reloc = False
        else:
            lx = math.cos(self.prev_theta) * dx + math.sin(self.prev_theta) * dy
            ly = -math.sin(self.prev_theta) * dx + math.cos(self.prev_theta) * dy
            self.odom_x += math.cos(self.odom_theta) * lx - math.sin(self.odom_theta) * ly
            self.odom_y += math.sin(self.odom_theta) * lx + math.cos(self.odom_theta) * ly
            self.odom_theta = self._wrap(self.odom_theta + dtheta)

        self.prev_x     = x
        self.prev_y     = y
        self.prev_theta = theta
        
        t = TransformStamped()
        t.header.stamp    = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id  = 'amcl_base'
        t.transform.translation.x = self.odom_x
        t.transform.translation.y = self.odom_y
        t.transform.translation.z = 0.0
        t.transform.rotation.z    = math.sin(self.odom_theta / 2.0)
        t.transform.rotation.w    = math.cos(self.odom_theta / 2.0)
        self.tf_broadcaster.sendTransform(t)

        odom_msg = PoseStamped()
        odom_msg.header = t.header
        odom_msg.pose.position.x = self.odom_x
        odom_msg.pose.position.y = self.odom_y
        odom_msg.pose.orientation.z = t.transform.rotation.z
        odom_msg.pose.orientation.w = t.transform.rotation.w
        self.odom_pub.publish(odom_msg)

        if self._kidnap_pending_reloc:
            return
            
        if self._kidnap_suppress_cycles > 0:
            self._kidnap_suppress_cycles -= 1
            return

        angle_inc = 2.0 * math.pi / self.num_beams
        scan                 = LaserScan()
        scan.header.stamp    = stamp
        scan.header.frame_id = 'amcl_base'
        scan.angle_min       = 0.0
        scan.angle_max       = 2.0 * math.pi - angle_inc
        scan.angle_increment = angle_inc
        scan.time_increment  = 0.0
        scan.scan_time       = 0.05
        scan.range_min       = self.min_range
        scan.range_max       = self.max_range
        scan.ranges = [
            self._raycast(x, y, theta + i * angle_inc)
            for i in range(self.num_beams)
        ]
        self.scan_pub.publish(scan)

    def amcl_pose_cb(self, msg):
        viz = PoseStamped()
        viz.header = msg.header
        viz.pose = msg.pose.pose
        self.amcl_viz_pub.publish(viz)


def main(args=None):
    rclpy.init(args=args)
    node = ScanBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
