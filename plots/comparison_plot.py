#!/usr/bin/env python3
"""
Live comparison plot: custom MCL vs Nav2 AMCL

Subscribes to:
  /true_pose           ground truth  (PoseStamped)
  /mcl_pose            custom MCL    (PoseStamped)
  /amcl_pose           Nav2 AMCL     (PoseWithCovarianceStamped)
  /mcl_confidence      custom MCL max particle weight (Float32)

Shows the localization error of both filters, the custom MCL confidence,
and a table of running statistics.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Float32

import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

MAX_POINTS = 3000     # about 150 s at 20 Hz
WINDOW = 120.0      # seconds of history shown
KIDNAP_JUMP = 3.0   # metres; a larger jump in true pose is a kidnap

# ── rolling buffers ─────────────────────────────────────────────────────────
times_mcl   = deque(maxlen=MAX_POINTS)
times_amcl  = deque(maxlen=MAX_POINTS)
errors_mcl  = deque(maxlen=MAX_POINTS)
errors_amcl = deque(maxlen=MAX_POINTS)
conf_mcl    = deque(maxlen=MAX_POINTS)
kidnap_ts   = []                        # timestamps of detected kidnaps

true_pose = None
prev_true = None
t0        = None

# ── running statistics ──────────────────────────────────────────────────────
mcl_err_sum    = 0.0;  mcl_err_n  = 0;  mcl_max_err  = 0.0
amcl_err_sum   = 0.0;  amcl_err_n = 0;  amcl_max_err = 0.0


class ComparisonNode(Node):

    def __init__(self):
        super().__init__('mcl_vs_amcl_comparison')
        self.create_subscription(PoseStamped,                '/true_pose',     self.true_cb, 10)
        self.create_subscription(PoseStamped,                '/mcl_pose',      self.mcl_cb,  10)
        self.create_subscription(PoseWithCovarianceStamped,  '/amcl_pose',     self.amcl_cb, 10)
        self.create_subscription(Float32,                    '/mcl_confidence', self.conf_cb, 10)
        self.get_logger().info('Comparison plotter running')

    # ── ground truth ────────────────────────────────────────────────────────
    def true_cb(self, msg):
        global true_pose, prev_true, t0
        now = self.get_clock().now().nanoseconds * 1e-9
        if t0 is None:
            t0 = now

        if prev_true is not None:
            dx = msg.pose.position.x - prev_true.pose.position.x
            dy = msg.pose.position.y - prev_true.pose.position.y
            if math.sqrt(dx*dx + dy*dy) > KIDNAP_JUMP:
                kidnap_ts.append(now - t0)

        prev_true = msg
        true_pose = msg

    # ── custom MCL ──────────────────────────────────────────────────────────
    def mcl_cb(self, msg):
        global mcl_err_sum, mcl_err_n, mcl_max_err
        if true_pose is None or t0 is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        dx  = msg.pose.position.x - true_pose.pose.position.x
        dy  = msg.pose.position.y - true_pose.pose.position.y
        err = math.sqrt(dx*dx + dy*dy)

        times_mcl.append(now - t0)
        errors_mcl.append(err)
        mcl_err_sum += err
        mcl_err_n   += 1
        mcl_max_err  = max(mcl_max_err, err)

    # ── Nav2 AMCL ──────────────────────────────────────────────────────────
    def amcl_cb(self, msg):
        global amcl_err_sum, amcl_err_n, amcl_max_err
        if true_pose is None or t0 is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        dx  = msg.pose.pose.position.x - true_pose.pose.position.x
        dy  = msg.pose.pose.position.y - true_pose.pose.position.y
        err = math.sqrt(dx*dx + dy*dy)

        times_amcl.append(now - t0)
        errors_amcl.append(err)
        amcl_err_sum += err
        amcl_err_n   += 1
        amcl_max_err  = max(amcl_max_err, err)

    # ── MCL confidence ──────────────────────────────────────────────────────
    def conf_cb(self, msg):
        conf_mcl.append(msg.data)


def main():
    rclpy.init()
    node = ComparisonNode()

    fig, axes = plt.subplots(3, 1, figsize=(10, 9),
                             gridspec_kw={'height_ratios': [3, 2, 1.2]})
    fig.canvas.manager.set_window_title('Custom MCL vs Nav2 AMCL')
    ax1, ax2, ax3 = axes

    # ── error comparison ────────────────────────────────────────────────────
    line_mcl,  = ax1.plot([], [], color='tab:red', lw=1.2, label='Custom MCL')
    line_amcl, = ax1.plot([], [], color='tab:blue', lw=1.2, label='Nav2 AMCL')
    ax1.set_ylabel('Error (m)')
    ax1.set_title('Localization error (dashed lines: kidnaps)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    kidnap_lines = []

    # ── MCL confidence ──────────────────────────────────────────────────────
    line_conf, = ax2.plot([], [], color='tab:green', lw=1.2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Max particle weight')
    ax2.set_title('Custom MCL particle confidence')
    ax2.grid(True, alpha=0.3)

    # ── statistics table ────────────────────────────────────────────────────
    ax3.axis('off')
    table = ax3.table(
        cellText=[['0', '0'], ['-', '-'], ['-', '-'], ['-', '-']],
        rowLabels=['Samples', 'Mean error (m)', 'Current error (m)', 'Max error (m)'],
        colLabels=['Custom MCL', 'Nav2 AMCL'],
        loc='center', cellLoc='center', colWidths=[0.25, 0.25],
    )
    table.scale(1, 1.3)
    kidnap_text = ax3.text(0.5, -0.05, '', transform=ax3.transAxes, ha='center')

    fig.tight_layout()

    def set_cell(row, col, text):
        table[row + 1, col].get_text().set_text(text)

    def update(_):
        rclpy.spin_once(node, timeout_sec=0)

        # error lines
        if len(times_mcl) >= 2:
            t = list(times_mcl)
            line_mcl.set_data(t, list(errors_mcl))
            hi = max(errors_mcl)
            if errors_amcl:
                hi = max(hi, max(errors_amcl))
            ax1.set_xlim(max(0, t[-1] - WINDOW), t[-1] + 2)
            ax1.set_ylim(0, max(1.0, hi * 1.1))

        if len(times_amcl) >= 2:
            line_amcl.set_data(list(times_amcl), list(errors_amcl))

        # kidnap markers
        while len(kidnap_lines) < len(kidnap_ts):
            kt = kidnap_ts[len(kidnap_lines)]
            kidnap_lines.append(ax1.axvline(kt, color='gray', ls='--', lw=0.8))

        # confidence
        if len(times_mcl) >= 2 and conf_mcl:
            t = list(times_mcl)
            c = list(conf_mcl)[-len(t):]
            line_conf.set_data(t[-len(c):], c)
            ax2.set_xlim(max(0, t[-1] - WINDOW), t[-1] + 2)
            ax2.set_ylim(0, max(c) * 1.2 + 1e-4)

        # statistics
        for col, (n, total, mx, errs) in enumerate([
                (mcl_err_n,  mcl_err_sum,  mcl_max_err,  errors_mcl),
                (amcl_err_n, amcl_err_sum, amcl_max_err, errors_amcl)]):
            set_cell(0, col, f'{n}')
            if n:
                set_cell(1, col, f'{total / n:.3f}')
                set_cell(2, col, f'{errs[-1]:.3f}')
                set_cell(3, col, f'{mx:.3f}')
        kidnap_text.set_text(f'Kidnaps so far: {len(kidnap_ts)}')

        return line_mcl, line_amcl, line_conf

    _ = animation.FuncAnimation(fig, update, interval=50, blit=False)
    plt.show()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
