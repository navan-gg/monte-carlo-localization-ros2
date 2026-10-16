#!/usr/bin/env python3
"""
Converts the Nav2 AMCL particle cloud into an RViz marker.

After a kidnap AMCL spreads up to 20000 particles over the whole map and
publishes them many times per second. This node only forwards a thinned-out
copy (at most viz_max_points points, at most 1 / viz_period times per second),
so RViz stays responsive. It runs as its own process so that decoding the
large messages does not slow down the scan bridge.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Point
from nav2_msgs.msg import ParticleCloud
from visualization_msgs.msg import Marker


class AmclParticleViz(Node):
    def __init__(self):
        super().__init__('amcl_particle_viz')

        self.viz_max_points = 2000
        self.viz_period = 0.2       # seconds
        self._last_viz_time = None

        # AMCL publishes best effort. Depth 1: if this node falls behind,
        # only the newest cloud is kept.
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(ParticleCloud, '/amcl_particles', self.particles_cb, qos)
        self.marker_pub = self.create_publisher(Marker, '/amcl_particles_marker', 10)

    def particles_cb(self, msg):
        now = self.get_clock().now()
        if self._last_viz_time is not None and \
                (now - self._last_viz_time).nanoseconds < self.viz_period * 1e9:
            return
        self._last_viz_time = now

        marker = Marker()
        marker.header = msg.header
        marker.ns = "amcl_particles"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.color.a = 1.0
        marker.color.r = 0.54
        marker.color.g = 0.17
        marker.color.b = 0.89

        step = max(1, -(-len(msg.particles) // self.viz_max_points))   # ceiling division
        for p in msg.particles[::step]:
            pt = Point()
            pt.x = p.pose.position.x
            pt.y = p.pose.position.y
            marker.points.append(pt)

        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = AmclParticleViz()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
