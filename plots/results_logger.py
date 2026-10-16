#!/usr/bin/env python3
"""
Results Logger — Custom MCL vs Nav2 AMCL

Records the localization error of both filters against the ground truth and
saves it to disk, so results are kept after the live plot windows are closed.

Subscribes to:
  /true_pose           ground truth  (PoseStamped)
  /mcl_pose            custom MCL    (PoseStamped)
  /amcl_pose           Nav2 AMCL     (PoseWithCovarianceStamped)

Writes to the output directory (default /home/ubuntu/mcl_ws/results):
  custom_mcl_errors.csv   time_s, error_m
  amcl_errors.csv         time_s, error_m
  kidnaps.csv             time_s of each detected kidnap
  mcl_comparison.png      error over time, error histogram,
                          running mean error, statistics
"""

import csv
import math
import os
import signal

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

KIDNAP_JUMP = 1.5      # metres; a larger jump in true pose is a kidnap
SAVE_PERIOD = 10.0     # seconds between saves


class ResultsLogger(Node):

    def __init__(self):
        super().__init__('results_logger')

        self.declare_parameter('output_dir', '/home/ubuntu/mcl_ws/results')
        self.output_dir = self.get_parameter('output_dir').value
        os.makedirs(self.output_dir, exist_ok=True)

        self.true_pose = None
        self.t0 = None
        self.mcl_errors = []      # (time_s, error_m)
        self.amcl_errors = []
        self.kidnap_times = []

        self.create_subscription(PoseStamped, '/true_pose', self.true_cb, 10)
        self.create_subscription(PoseStamped, '/mcl_pose', self.mcl_cb, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.amcl_cb, 10)

        self.create_timer(SAVE_PERIOD, self.save)
        self.get_logger().info(f'Results logger running, output: {self.output_dir}')

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9 - self.t0

    def _error(self, pose):
        dx = pose.position.x - self.true_pose.pose.position.x
        dy = pose.position.y - self.true_pose.pose.position.y
        return math.sqrt(dx * dx + dy * dy)

    def true_cb(self, msg):
        if self.t0 is None:
            self.t0 = self.get_clock().now().nanoseconds * 1e-9
        elif self.true_pose is not None:
            dx = msg.pose.position.x - self.true_pose.pose.position.x
            dy = msg.pose.position.y - self.true_pose.pose.position.y
            if math.sqrt(dx * dx + dy * dy) > KIDNAP_JUMP:
                self.kidnap_times.append(self._now())
        self.true_pose = msg

    def mcl_cb(self, msg):
        if self.true_pose is None:
            return
        self.mcl_errors.append((self._now(), self._error(msg.pose)))

    def amcl_cb(self, msg):
        if self.true_pose is None:
            return
        self.amcl_errors.append((self._now(), self._error(msg.pose.pose)))

    # ── saving ───────────────────────────────────────────────────────────────
    def _write_csv(self, name, header, rows):
        path = os.path.join(self.output_dir, name)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        os.chmod(path, 0o666)

    def save(self):
        if not self.mcl_errors and not self.amcl_errors:
            return
        self._write_csv('custom_mcl_errors.csv', ['time_s', 'error_m'], self.mcl_errors)
        self._write_csv('amcl_errors.csv', ['time_s', 'error_m'], self.amcl_errors)
        self._write_csv('kidnaps.csv', ['time_s'], [[t] for t in self.kidnap_times])
        self._plot()

    def _plot(self):
        series = [
            ('Custom MCL', self.mcl_errors, 'tab:red'),
            ('Nav2 AMCL', self.amcl_errors, 'tab:blue'),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle('Custom MCL vs Nav2 AMCL')
        ax1, ax2, ax3, ax4 = axes.flatten()

        rows = ['Mean error (m)', 'Median error (m)', 'Std dev (m)',
                'Max error (m)', 'Samples']
        cols, cells = [], []

        for label, data, color in series:
            if not data:
                continue
            t, e = (np.array(v) for v in zip(*data))

            ax1.plot(t, e, color=color, linewidth=1.0, label=label)
            ax2.hist(e, bins=30, alpha=0.6, color=color, label=label, density=True)
            ax3.plot(t, np.cumsum(e) / np.arange(1, len(e) + 1),
                     color=color, linewidth=2, label=label)

            cols.append(label)
            cells.append([f'{np.mean(e):.3f}', f'{np.median(e):.3f}', f'{np.std(e):.3f}',
                          f'{np.max(e):.3f}', f'{len(e)}'])

        for kt in self.kidnap_times:
            ax1.axvline(kt, color='gray', linestyle='--', linewidth=0.8)

        ax1.set(xlabel='Time (s)', ylabel='Error (m)',
                title='Localization error (dashed lines: kidnaps)')
        ax2.set(xlabel='Error (m)', ylabel='Density', title='Error distribution')
        ax3.set(xlabel='Time (s)', ylabel='Mean error (m)', title='Running mean error')
        for ax in (ax1, ax2, ax3):
            ax.legend()
            ax.grid(True, alpha=0.3)

        ax4.axis('off')
        if cells:
            table = ax4.table(cellText=[list(r) for r in zip(*cells)], rowLabels=rows,
                              colLabels=cols, loc='center', cellLoc='center',
                              colWidths=[0.3] * len(cols))
            table.scale(1, 1.6)
        ax4.set_title(f'Summary ({len(self.kidnap_times)} kidnaps)')

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'mcl_comparison.png')
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        os.chmod(path, 0o666)


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = ResultsLogger()

    # Save once more on Ctrl+C / docker compose down before exiting
    running = [True]

    def stop(*_):
        running[0] = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while running[0] and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.2)

    node.save()
    node.get_logger().info(f'Results saved to {node.output_dir}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
