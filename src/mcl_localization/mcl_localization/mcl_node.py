import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Float32, Float32MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseArray, Pose
from nav_msgs.msg import OccupancyGrid

import random
import math


class MCLNode(Node):

    def __init__(self):
        super().__init__('mcl_node')

        self.declare_parameter('num_particles', 500)
        self.num_particles = self.get_parameter('num_particles').value
        self.max_sensor_range = 8.0

        # ---- Measurement model noise (must match or be >= robot sensor noise) ----
        self.sigma_r = 0.3       # range std dev (m)
        self.sigma_phi = 0.1     # bearing std dev (rad)

        # ---- Motion model noise parameters (report Eqs 8-10) ----
        self.alpha_rot = 0.05      # rotational noise coefficient
        self.alpha_trans = 0.02    # translational noise coefficient

        # ---- Kidnapped robot / Lost detection variables ----
        self.w_min = 1e-50      # min weight threshold for lost detection
        self.lost_threshold = 8
        self.lost_counter = 0

        self.map_data = None
        self.map_resolution = None
        self.map_origin = None
        self.map_width = None
        self.map_height = None

        self.particles = []
        self.prev_pose = None

        # Landmark positions received from robot node
        self.landmarks = []

        # Latest sensor readings
        self.latest_readings = None

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        lm_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos
        )

        self.create_subscription(
            PoseStamped,
            '/odom_pose',
            self.pose_callback,
            10
        )

        self.create_subscription(
            Float32MultiArray,
            '/landmarks',
            self.landmarks_callback,
            lm_qos
        )

        self.create_subscription(
            Float32MultiArray,
            '/landmark_readings',
            self.readings_callback,
            10
        )

        self.marker_pub = self.create_publisher(
            Marker,
            '/particle_cloud',
            10
        )

        self.mcl_pose_pub = self.create_publisher(PoseStamped, "/mcl_pose", 10)
        self.mcl_conf_pub = self.create_publisher(Float32, "/mcl_confidence", 10)
        self.get_logger().info("MCL Node Started (Landmark-based)")

    # ---------------- MAP CALLBACK ----------------
    def map_callback(self, msg):

        if self.map_data is not None:
            return

        self.map_data = msg.data
        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin
        self.map_width = msg.info.width
        self.map_height = msg.info.height

        self.initialize_particles()
        self.get_logger().info("Particles initialized")

    # ---------------- LANDMARKS CALLBACK ----------------
    def landmarks_callback(self, msg):
        # Flat array: [x0, y0, x1, y1, ...]
        data = msg.data
        self.landmarks = []
        for i in range(0, len(data), 2):
            self.landmarks.append((data[i], data[i + 1]))
        self.get_logger().info(f"Received {len(self.landmarks)} landmarks")

    # ---------------- READINGS CALLBACK ----------------
    def readings_callback(self, msg):
        # Flat array: [landmark_idx, range, bearing, ...] triples
        data = msg.data
        readings = []
        for i in range(0, len(data), 3):
            lm_idx = int(data[i])
            r = data[i + 1]
            phi = data[i + 2]
            readings.append((lm_idx, r, phi))
        self.latest_readings = readings

    # ---------------- INITIALIZE PARTICLES ----------------
    def initialize_particles(self):

        self.particles = []

        for _ in range(self.num_particles):
            while True:
                x = random.uniform(
                    self.map_origin.position.x,
                    self.map_origin.position.x +
                    self.map_width * self.map_resolution
                )

                y = random.uniform(
                    self.map_origin.position.y,
                    self.map_origin.position.y +
                    self.map_height * self.map_resolution
                )

                if self.is_valid(x, y):
                    theta = random.uniform(-math.pi, math.pi)
                    self.particles.append([x, y, theta])
                    break
        
        # Publish huge initial error before convergence
        wx_sum = sum(p[0] for p in self.particles) / self.num_particles
        wy_sum = sum(p[1] for p in self.particles) / self.num_particles
        wsin_sum = sum(math.sin(p[2]) for p in self.particles) / self.num_particles
        wcos_sum = sum(math.cos(p[2]) for p in self.particles) / self.num_particles

        est_theta = math.atan2(wsin_sum, wcos_sum)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = float(wx_sum)
        pose_msg.pose.position.y = float(wy_sum)
        pose_msg.pose.position.z = 0.0

        half_yaw = est_theta * 0.5
        pose_msg.pose.orientation.z = math.sin(half_yaw)
        pose_msg.pose.orientation.w = math.cos(half_yaw)

        self.mcl_pose_pub.publish(pose_msg)

    # ---------------- VALIDITY CHECK ----------------
    def is_valid(self, x, y):

        mx = int((x - self.map_origin.position.x) / self.map_resolution)
        my = int((y - self.map_origin.position.y) / self.map_resolution)

        if mx < 0 or my < 0 or mx >= self.map_width or my >= self.map_height:
            return False

        index = my * self.map_width + mx
        return self.map_data[index] == 0

    # ---------------- POSE CALLBACK ----------------
    def pose_callback(self, msg):

        if self.map_data is None:
            return

        if not self.landmarks:
            return

        if self.prev_pose is None:
            self.prev_pose = msg
            return

        dx = msg.pose.position.x - self.prev_pose.pose.position.x
        dy = msg.pose.position.y - self.prev_pose.pose.position.y

        q = msg.pose.orientation
        theta = math.atan2(2 * (q.w * q.z), 1 - 2 * (q.z * q.z))

        prev_q = self.prev_pose.pose.orientation
        prev_theta = math.atan2(
            2 * (prev_q.w * prev_q.z),
            1 - 2 * (prev_q.z * prev_q.z)
        )

        dtheta = theta - prev_theta
        self.prev_pose = msg

        # ---- Motion update (rot1-trans-rot2 decomposition) ----
        self.motion_update(dx, dy, dtheta, prev_theta)

        # ---- Measurement update (range-bearing to landmarks) ----
        if self.latest_readings is not None:
            self.measurement_update(self.latest_readings)

        self.publish_particles()

    # ---------------- MOTION UPDATE (rot1-trans-rot2) ----------------
    def motion_update(self, dx, dy, dtheta, prev_theta):
        """Odometry-based motion model with rot1-trans-rot2 decomposition.
        Matches report Eqs. 5-11."""

        delta_trans = math.sqrt(dx * dx + dy * dy)

        # Only decompose if there's meaningful translation
        if delta_trans > 1e-6:
            delta_rot1 = math.atan2(dy, dx) - prev_theta
            delta_rot1 = math.atan2(math.sin(delta_rot1), math.cos(delta_rot1))
        else:
            delta_rot1 = 0.0

        delta_rot2 = dtheta - delta_rot1
        delta_rot2 = math.atan2(math.sin(delta_rot2), math.cos(delta_rot2))

        for p in self.particles:
            # Add noise scaled by motion magnitude (report Eqs. 8-10)
            hat_rot1 = delta_rot1 + random.gauss(
                0,
                self.alpha_rot * abs(delta_rot1) +
                self.alpha_trans * delta_trans + 1e-4
            )
            hat_trans = delta_trans + random.gauss(
                0,
                self.alpha_trans * delta_trans +
                self.alpha_rot * abs(dtheta) + 1e-4
            )
            hat_rot2 = delta_rot2 + random.gauss(
                0,
                self.alpha_rot * abs(delta_rot2) +
                self.alpha_trans * delta_trans + 1e-4
            )

            # Propagate particle (report Eqs. 5-7)
            p[2] += hat_rot1
            p[0] += hat_trans * math.cos(p[2])
            p[1] += hat_trans * math.sin(p[2])
            p[2] += hat_rot2
            p[2] = math.atan2(math.sin(p[2]), math.cos(p[2]))

            # If particle ends up in invalid space, resample to random free pos
            if not self.is_valid(p[0], p[1]):
                while True:
                    x = random.uniform(
                        self.map_origin.position.x,
                        self.map_origin.position.x +
                        self.map_width * self.map_resolution
                    )

                    y = random.uniform(
                        self.map_origin.position.y,
                        self.map_origin.position.y +
                        self.map_height * self.map_resolution
                    )

                    if self.is_valid(x, y):
                        p[0] = x
                        p[1] = y
                        p[2] = random.uniform(-math.pi, math.pi)
                        break

    # ---------------- MEASUREMENT UPDATE (range-bearing) ----------------
    def measurement_update(self, readings):
        """Landmark-based measurement model.
        Computes weights using per-landmark Gaussian likelihood
        (report Eqs. 13-16)."""

        if not readings:
            return

        weights = []

        for p in self.particles:
            px, py, ptheta = p
            w = 1.0

            for lm_idx, r_obs, phi_obs in readings:
                if lm_idx >= len(self.landmarks):
                    continue

                lx, ly = self.landmarks[lm_idx]

                # Expected range (report Eq. 9)
                dx = lx - px
                dy = ly - py
                r_exp = math.sqrt(dx * dx + dy * dy)

                # Expected bearing (report Eq. 10)
                phi_exp = math.atan2(dy, dx) - ptheta
                phi_exp = math.atan2(math.sin(phi_exp), math.cos(phi_exp))

                # Range difference
                dr = r_obs - r_exp

                # Bearing difference with wrap-around (report Eq. 16)
                dphi = phi_obs - phi_exp
                dphi = math.atan2(math.sin(dphi), math.cos(dphi))

                # Per-landmark Gaussian likelihood (report Eq. 13)
                # p(z_j | x^[i], l_j) = 1/(2π σ_r σ_φ) * exp(...)
                norm = 1.0 / (2.0 * math.pi * self.sigma_r * self.sigma_phi)
                w *= norm * math.exp(
                    -dr * dr / (2 * self.sigma_r * self.sigma_r)
                    - dphi * dphi / (2 * self.sigma_phi * self.sigma_phi)
                )

            weights.append(w)

        max_unnormalized_weight = max(weights) if weights else 0.0

        # ---------------- LOST DETECTION ----------------
        if max_unnormalized_weight < self.w_min:
            self.lost_counter += 1
        else:
            self.lost_counter = 0

        if self.lost_counter >= self.lost_threshold:
            self.get_logger().warn("Robot lost! Global initialization triggered.")
            self.initialize_particles()
            self.lost_counter = 0
            weights = [1.0 / self.num_particles] * self.num_particles
        else:
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
            else:
                weights = [1.0 / self.num_particles] * self.num_particles

        # --- Publish Confidence ---
        conf_msg = Float32()
        conf_msg.data = float(max(weights))
        self.mcl_conf_pub.publish(conf_msg)

        # --- Publish Pose (weighted mean of all particles) ---
        wx_sum = 0.0
        wy_sum = 0.0
        wsin_sum = 0.0
        wcos_sum = 0.0
        for i, p in enumerate(self.particles):
            wx_sum += weights[i] * p[0]
            wy_sum += weights[i] * p[1]
            wsin_sum += weights[i] * math.sin(p[2])
            wcos_sum += weights[i] * math.cos(p[2])

        est_x = wx_sum
        est_y = wy_sum
        est_theta = math.atan2(wsin_sum, wcos_sum)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = float(est_x)
        pose_msg.pose.position.y = float(est_y)
        pose_msg.pose.position.z = 0.0

        # yaw to quaternion
        half_yaw = est_theta * 0.5
        pose_msg.pose.orientation.z = math.sin(half_yaw)
        pose_msg.pose.orientation.w = math.cos(half_yaw)

        self.mcl_pose_pub.publish(pose_msg)
        self.resample(weights)

    # ---------------- LOW VARIANCE RESAMPLING ----------------
    def resample(self, weights):

        new_particles = []

        # 15% random particles for kidnapped recovery/exploration (as per Report)
        num_random = int(self.num_particles * 0.15)
        num_resample = self.num_particles - num_random

        for _ in range(num_random):
            while True:
                x = random.uniform(
                    self.map_origin.position.x,
                    self.map_origin.position.x +
                    self.map_width * self.map_resolution
                )

                y = random.uniform(
                    self.map_origin.position.y,
                    self.map_origin.position.y +
                    self.map_height * self.map_resolution
                )

                if self.is_valid(x, y):
                    new_particles.append(
                        [x, y, random.uniform(-math.pi, math.pi)]
                    )
                    break

        # Low-Variance Resampling for the remaining particles (report Alg. 1)
        if num_resample > 0:
            M = num_resample
            r = random.uniform(0.0, 1.0 / M)
            c = weights[0]
            i = 0

            for m in range(M):
                U = r + m * (1.0 / M)
                while U > c and i < self.num_particles - 1:
                    i += 1
                    c += weights[i]
                new_particles.append(self.particles[i][:])

        self.particles = new_particles

    # ---------------- PUBLISH PARTICLES ----------------
    def publish_particles(self):

        msg = Marker()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ns = "particles"
        msg.id = 0
        msg.type = Marker.POINTS
        msg.action = Marker.ADD
        msg.scale.x = 0.05
        msg.scale.y = 0.05
        msg.color.a = 1.0
        msg.color.r = 1.0
        msg.color.g = 0.0
        msg.color.b = 0.0

        for p in self.particles:
            pt = Point()
            pt.x = float(p[0])
            pt.y = float(p[1])
            pt.z = 0.0
            msg.points.append(pt)

        self.marker_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MCLNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

