import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32

import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque


MAX_POINTS = 1500     # about 75 s at 20 Hz
WINDOW = 60.0       # seconds of history shown

times      = deque(maxlen=MAX_POINTS)
errors     = deque(maxlen=MAX_POINTS)
confidence = deque(maxlen=MAX_POINTS)

true_pose = None
t0        = None


class LivePlotNode(Node):

    def __init__(self):
        super().__init__('mcl_live_plotter')
        self.create_subscription(PoseStamped, '/true_pose', self.true_cb, 10)
        self.create_subscription(PoseStamped, '/mcl_pose',  self.mcl_cb,  10)
        self.create_subscription(Float32,     '/mcl_confidence', self.conf_cb, 10)
        self.get_logger().info("Live plotter running")

    def true_cb(self, msg):
        global true_pose
        true_pose = msg

    def mcl_cb(self, msg):
        global t0
        if true_pose is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if t0 is None:
            t0 = now

        dx  = msg.pose.position.x - true_pose.pose.position.x
        dy  = msg.pose.position.y - true_pose.pose.position.y
        err = math.sqrt(dx*dx + dy*dy)

        times.append(now - t0)
        errors.append(err)

    def conf_cb(self, msg):
        confidence.append(msg.data)


def main():
    rclpy.init()
    node = LivePlotNode()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    fig.canvas.manager.set_window_title('Custom MCL')

    line_err,  = ax1.plot([], [], color='tab:red', linewidth=1.2)
    line_conf, = ax2.plot([], [], color='tab:green', linewidth=1.2)

    ax1.set_ylabel('Error (m)')
    ax1.set_title('Custom MCL localization error')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Max particle weight')
    ax2.set_title('Particle confidence')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    def update(_):
        rclpy.spin_once(node, timeout_sec=0)

        if len(times) < 2:
            return line_err, line_conf

        t = list(times)

        line_err.set_data(t, list(errors))
        ax1.set_xlim(max(0, t[-1] - WINDOW), t[-1] + 1)
        ax1.set_ylim(0, max(1.0, max(errors) * 1.1))

        if confidence:
            c = list(confidence)[-len(t):]
            line_conf.set_data(t[-len(c):], c)
            ax2.set_ylim(0, max(c) * 1.2 + 1e-4)

        return line_err, line_conf

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False)
    plt.show()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
