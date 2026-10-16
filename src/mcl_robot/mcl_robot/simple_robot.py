import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, PoseStamped, Point
from nav_msgs.msg import Path
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Float32MultiArray
from tf2_ros import TransformBroadcaster
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
import math
import random
import numpy as np
from scipy.ndimage import distance_transform_edt


class SimpleRobot(Node):

    ROBOT_RADIUS      = 0.25
    ROBOT_WIDTH       = 0.50   # full width — matches ROBOT_RADIUS * 2
    ROBOT_LENGTH      = 0.55   # full length front-to-back
    ROBOT_HEIGHT      = 0.20
    LOOKAHEAD_DIST    = 0.40
    LINEAR_SPEED      = 0.8
    MAX_SPEED         = 1.2
    MIN_SPEED         = 0.2
    TURN_SPEED        = 1.8
    TURN_ANGLE        = math.radians(135)
    RANDOM_TURN_MAX   = math.pi / 3
    KIDNAP_INTERVAL   = 30.0
    WHEELBASE         = 0.3
    MAX_STEER_ANGLE   = math.radians(35)
    SLOW_DIST         = 1.0
    STOP_DIST         = 0.4

    # ── Landmark positions (x, y) spread across the map ──────────────────
    # Map spans roughly -10 to +10 in both axes


    # Case 1: 5 landmarks
    LANDMARKS = [
        (-7.0,  7.0),  ( 7.0,  7.0),  
        ( 0.0,  0.0),  
        (-7.0, -7.0),  ( 7.0, -7.0)
    ]

    #Case 2: 15 landmarks
    #LANDMARKS = [
       # (-7.0,  7.0),  ( 0.0,  8.0),  ( 7.0,  7.0),
        #(-8.0,  3.0),  (-3.0,  4.0),  ( 3.0,  4.0),  ( 8.0,  3.0),
        #(-6.0,  0.0),  ( 0.0,  0.0),  ( 6.0,  0.0),
        #(-8.0, -3.0),  (-3.0, -4.0),  ( 3.0, -4.0),  ( 8.0, -3.0),
        #(-7.0, -7.0)
    #]

    # ── Sensor parameters ────────────────────────────────────────────────
    MAX_SENSOR_RANGE  = 8.0     # metres – max detection range
    SENSOR_SIGMA_R    = 0.2     # range noise std dev (m)
    SENSOR_SIGMA_PHI  = 0.05    # bearing noise std dev (rad)

    def __init__(self):
        super().__init__('simple_robot')

        self.tf_broadcaster = TransformBroadcaster(self)
        self.marker_pub     = self.create_publisher(MarkerArray, '/robot_marker', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/true_pose', 10)
        self.path_pub = self.create_publisher(Path, '/robot_path', 10)
        self.robot_path = Path()
        self.robot_path.header.frame_id = 'map'

        # ── Landmark publishers ──────────────────────────────────────────
        lm_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.landmark_pos_pub = self.create_publisher(
            Float32MultiArray, '/landmarks', lm_qos
        )
        self.landmark_reading_pub = self.create_publisher(
            Float32MultiArray, '/landmark_readings', 10
        )
        self.landmark_marker_pub = self.create_publisher(
            MarkerArray, '/landmark_markers', lm_qos
        )
        self._landmarks_published = False

        self.map_data       = None
        self.map_resolution = None
        self.map_origin     = None
        self.map_width      = None
        self.map_height     = None
        self.dist_map       = None

        qos = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos)

        self.x     = 5.0
        self.y     = -5.0
        self.theta = 0.0

        self.v_current   = 0.0
        self.v_target    = self.LINEAR_SPEED
        self.steer_angle = 0.0

        self.turning        = False
        self.turn_remaining = 0.0
        self.turn_direction = 1.0

        self.walk_time         = 0.0
        self.walk_duration     = random.uniform(2.0, 5.0)
        self.time_since_kidnap = 0.0
        self.kidnap_count      = 0

        self.last_time = self.get_clock().now()
        self.timer     = self.create_timer(0.05, self.update)

    def map_callback(self, msg):
        self.map_data       = msg.data
        self.map_resolution = msg.info.resolution
        self.map_origin     = msg.info.origin
        self.map_width      = msg.info.width
        self.map_height     = msg.info.height
        self._build_distance_map()

    def _build_distance_map(self):
        grid      = np.array(self.map_data, dtype=np.int8).reshape(self.map_height, self.map_width)
        free_mask = (grid == 0).astype(np.float32)
        self.dist_map = distance_transform_edt(free_mask) * self.map_resolution
        self.get_logger().info("Distance map built")

    def _obstacle_dist_at(self, x, y):
        if self.dist_map is None:
            return self.SLOW_DIST
        mx = int((x - self.map_origin.position.x) / self.map_resolution)
        my = int((y - self.map_origin.position.y) / self.map_resolution)
        if mx < 0 or my < 0 or mx >= self.map_width or my >= self.map_height:
            return 0.0
        return float(self.dist_map[my, mx])

    # ── Footprint collision — rectangle of check points ───────────────────────
    # Samples a grid of points covering the full robot body (length x width).
    # This means side corners actually register as collisions, not just the
    # centre and a circle around it.
    def is_collision(self, x, y, theta=None):
        if self.map_data is None:
            return False

        # ── Boundary margin — keep entire robot visible inside the map ──
        margin = self.ROBOT_RADIUS + 0.15          # ~0.40 m from each edge
        map_min_x = self.map_origin.position.x + margin
        map_max_x = self.map_origin.position.x + self.map_width  * self.map_resolution - margin
        map_min_y = self.map_origin.position.y + margin
        map_max_y = self.map_origin.position.y + self.map_height * self.map_resolution - margin
        if x < map_min_x or x > map_max_x or y < map_min_y or y > map_max_y:
            return True

        if theta is None:
            theta = self.theta

        ct = math.cos(theta)
        st = math.sin(theta)
        hw = self.ROBOT_WIDTH  / 2.0   # half width
        hl = self.ROBOT_LENGTH / 2.0   # half length

        # 9-point rectangular footprint:
        # centre, 4 corners, 4 edge midpoints
        local_points = [
            ( 0.0,  0.0),               # centre
            ( hl,   hw),                # front-left corner
            ( hl,  -hw),                # front-right corner
            (-hl,   hw),                # rear-left corner
            (-hl,  -hw),                # rear-right corner
            ( hl,   0.0),               # front midpoint
            (-hl,   0.0),               # rear midpoint
            ( 0.0,  hw),                # left midpoint
            ( 0.0, -hw),                # right midpoint
        ]

        for lx, ly in local_points:
            wx = x + ct * lx - st * ly
            wy = y + st * lx + ct * ly

            mx = int((wx - self.map_origin.position.x) / self.map_resolution)
            my = int((wy - self.map_origin.position.y) / self.map_resolution)

            if mx < 0 or my < 0 or mx >= self.map_width or my >= self.map_height:
                return True
            if self.map_data[my * self.map_width + mx] > 50:
                return True

        return False

    def _lookahead_blocked(self):
        for frac in [0.5, 0.75, 1.0]:
            d  = frac * self.LOOKAHEAD_DIST
            lx = self.x + d * math.cos(self.theta)
            ly = self.y + d * math.sin(self.theta)
            if self.is_collision(lx, ly, self.theta):
                return True
        return False

    def _compute_target_speed(self):
        dist = self._obstacle_dist_at(self.x, self.y)
        if dist >= self.SLOW_DIST:
            return self.MAX_SPEED
        elif dist <= self.STOP_DIST:
            return self.MIN_SPEED
        else:
            t = (dist - self.STOP_DIST) / (self.SLOW_DIST - self.STOP_DIST)
            return self.MIN_SPEED + t * (self.MAX_SPEED - self.MIN_SPEED)

    def _random_free_pose(self):
        while True:
            x = random.uniform(
                self.map_origin.position.x,
                self.map_origin.position.x + self.map_width * self.map_resolution
            )
            y = random.uniform(
                self.map_origin.position.y,
                self.map_origin.position.y + self.map_height * self.map_resolution
            )
            if not self.is_collision(x, y):
                return x, y

    def _kidnap(self):
        x, y             = self._random_free_pose()
        self.x           = x
        self.y           = y
        self.theta       = random.uniform(-math.pi, math.pi)
        self.turning     = False
        self.v_current   = 0.0
        self.steer_angle = 0.0
        self.walk_time   = 0.0
        self.walk_duration  = random.uniform(2.0, 5.0)
        self.kidnap_count  += 1
        self.get_logger().info(f"Kidnap #{self.kidnap_count} → ({x:.2f}, {y:.2f})")

    def update(self):
        now = self.get_clock().now()
        dt  = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        dt = min(dt, 0.1)

        if self.map_data is not None:
            self.time_since_kidnap += dt
            if self.time_since_kidnap >= self.KIDNAP_INTERVAL:
                self.time_since_kidnap = 0.0
                self._kidnap()
                self.publish_all()
                return

        if self.turning:
            self._do_turn(dt)
        else:
            self._do_walk(dt)

        self.publish_all()

    def _do_turn(self, dt):
        step = self.TURN_SPEED * dt
        if step >= self.turn_remaining:
            self.theta         += self.turn_remaining * self.turn_direction
            self.theta          = self._wrap_angle(self.theta)
            self.turning        = False
            self.walk_time      = 0.0
            self.walk_duration  = random.uniform(2.0, 5.0)
            self.steer_angle    = 0.0
            self.v_target       = self.LINEAR_SPEED
        else:
            self.theta         += step * self.turn_direction
            self.theta          = self._wrap_angle(self.theta)
            self.turn_remaining -= step

    def _do_walk(self, dt):
        self.v_target  = self._compute_target_speed()
        accel          = 1.5
        self.v_current = self._approach(self.v_current, self.v_target, accel * dt)

        if self._lookahead_blocked():
            self._start_turn()
            return

        left_dist  = self._obstacle_dist_at(
            self.x + self.SLOW_DIST * math.cos(self.theta + math.pi / 4),
            self.y + self.SLOW_DIST * math.sin(self.theta + math.pi / 4)
        )
        right_dist = self._obstacle_dist_at(
            self.x + self.SLOW_DIST * math.cos(self.theta - math.pi / 4),
            self.y + self.SLOW_DIST * math.sin(self.theta - math.pi / 4)
        )

        steer_bias   = (left_dist - right_dist) / max(left_dist + right_dist, 0.1)
        target_steer = steer_bias * self.MAX_STEER_ANGLE
        self.steer_angle = self._approach(self.steer_angle, target_steer, math.radians(60) * dt)
        self.steer_angle = max(-self.MAX_STEER_ANGLE, min(self.MAX_STEER_ANGLE, self.steer_angle))

        if abs(self.steer_angle) > 1e-4:
            turning_radius = self.WHEELBASE / math.tan(self.steer_angle)
            omega          = self.v_current / turning_radius
        else:
            omega = 0.0

        new_x     = self.x     + self.v_current * math.cos(self.theta) * dt
        new_y     = self.y     + self.v_current * math.sin(self.theta) * dt
        new_theta = self._wrap_angle(self.theta + omega * dt)

        if not self.is_collision(new_x, new_y, new_theta):
            self.x     = new_x
            self.y     = new_y
            self.theta = new_theta

        self.walk_time += dt
        if self.walk_time >= self.walk_duration:
            self.theta         = self._wrap_angle(self.theta + random.uniform(-self.RANDOM_TURN_MAX, self.RANDOM_TURN_MAX))
            self.walk_time     = 0.0
            self.walk_duration = random.uniform(2.0, 5.0)

    def _start_turn(self):
        self.turning        = True
        self.turn_remaining = self.TURN_ANGLE + random.uniform(0, math.radians(30))
        self.v_target       = 0.0
        self.v_current      = 0.0
        self.steer_angle    = 0.0

        left_blocked  = self.is_collision(
            self.x + self.LOOKAHEAD_DIST * math.cos(self.theta + math.pi / 2),
            self.y + self.LOOKAHEAD_DIST * math.sin(self.theta + math.pi / 2),
            self.theta
        )
        right_blocked = self.is_collision(
            self.x + self.LOOKAHEAD_DIST * math.cos(self.theta - math.pi / 2),
            self.y + self.LOOKAHEAD_DIST * math.sin(self.theta - math.pi / 2),
            self.theta
        )

        if left_blocked and not right_blocked:
            self.turn_direction = -1.0
        elif right_blocked and not left_blocked:
            self.turn_direction =  1.0
        else:
            self.turn_direction = random.choice([-1.0, 1.0])

    def publish_all(self):
        self.publish_tf()
        self.publish_markers()
        self.publish_true_pose()
        self.publish_landmarks()
        self.publish_landmark_readings()

    def publish_true_pose(self):
        msg                    = PoseStamped()
        msg.header.frame_id    = "map"
        msg.header.stamp       = self.get_clock().now().to_msg()
        msg.pose.position.x    = self.x
        msg.pose.position.y    = self.y
        msg.pose.orientation.z = math.sin(self.theta / 2.0)
        msg.pose.orientation.w = math.cos(self.theta / 2.0)
        self.pose_pub.publish(msg)

        # Publish path
        self.robot_path.header.stamp = self.get_clock().now().to_msg()
        self.robot_path.poses.append(msg)
        if len(self.robot_path.poses) > 500:
            self.robot_path.poses.pop(0)
        self.path_pub.publish(self.robot_path)


    def publish_tf(self):
        t                         = TransformStamped()
        t.header.stamp            = self.get_clock().now().to_msg()
        t.header.frame_id         = 'map'
        t.child_frame_id          = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation.z    = math.sin(self.theta / 2.0)
        t.transform.rotation.w    = math.cos(self.theta / 2.0)
        self.tf_broadcaster.sendTransform(t)

    def publish_markers(self):
        stamp = self.get_clock().now().to_msg()
        array = MarkerArray()

        ct = math.cos(self.theta)
        st = math.sin(self.theta)
        hw = self.ROBOT_WIDTH  / 2.0
        hl = self.ROBOT_LENGTH / 2.0

        def local_to_world(lx, ly):
            p   = Point()
            p.x = self.x + ct * lx - st * ly
            p.y = self.y + st * lx + ct * ly
            p.z = self.ROBOT_HEIGHT / 2.0
            return p

        # Rectangle body drawn as 2 triangles (TRIANGLE_LIST needs sets of 3)
        front_left  = local_to_world( hl,  hw)
        front_right = local_to_world( hl, -hw)
        rear_left   = local_to_world(-hl,  hw)
        rear_right  = local_to_world(-hl, -hw)
        # Nose tip — extends slightly beyond the front edge to show direction
        nose        = local_to_world( hl + hw * 0.6, 0.0)

        speed_t     = (self.v_current - self.MIN_SPEED) / max(self.MAX_SPEED - self.MIN_SPEED, 1e-3)
        speed_t     = max(0.0, min(1.0, speed_t))
        fast_color  = ColorRGBA(r=0.0, g=1.0, b=0.8, a=1.0)   # cyan when fast
        slow_color  = ColorRGBA(r=1.0, g=0.4, b=0.0, a=1.0)   # orange when slow
        body_color  = ColorRGBA(
            r=slow_color.r + speed_t * (fast_color.r - slow_color.r),
            g=slow_color.g + speed_t * (fast_color.g - slow_color.g),
            b=slow_color.b + speed_t * (fast_color.b - slow_color.b),
            a=1.0
        )
        dark_color  = ColorRGBA(r=0.0, g=0.0, b=0.35, a=1.0)

        body        = Marker()
        body.header.frame_id = "map"
        body.header.stamp = stamp
        body.ns       = "robot"
        body.id       = 0
        body.type     = Marker.TRIANGLE_LIST
        body.action   = Marker.ADD
        body.scale.x  = 1.0
        body.scale.y  = 1.0
        body.scale.z  = 1.0

        # Triangle 1: rear rectangle (two tris)
        body.points = [
            front_left,  rear_left,  rear_right,   # tri 1
            front_left,  rear_right, front_right,  # tri 2
            front_left,  nose,       front_right,  # nose triangle
        ]
        body.colors = [
            body_color, dark_color, dark_color,
            body_color, dark_color, body_color,
            body_color, fast_color, body_color,
        ]
        array.markers.append(body)

        # Thin ground shadow cylinder
        base                    = Marker()
        base.header.frame_id    = "map"
        base.header.stamp       = stamp
        base.ns                 = "robot"
        base.id                 = 1
        base.type               = Marker.CYLINDER
        base.action             = Marker.ADD
        base.pose.position.x    = self.x
        base.pose.position.y    = self.y
        base.pose.position.z    = 0.01
        base.pose.orientation.w = 1.0
        base.scale.x            = self.ROBOT_WIDTH
        base.scale.y            = self.ROBOT_LENGTH
        base.scale.z            = 0.02
        base.color.a            = 0.25
        base.color.r            = body_color.r
        base.color.g            = body_color.g
        base.color.b            = body_color.b
        array.markers.append(base)

        self.marker_pub.publish(array)

    # ── Landmark helpers ─────────────────────────────────────────────────
    def publish_landmarks(self):
        """Publish landmark positions (once) and RViz markers."""
        if not self._landmarks_published:
            # Publish positions as flat array [x0, y0, x1, y1, ...]
            msg = Float32MultiArray()
            for lx, ly in self.LANDMARKS:
                msg.data.append(float(lx))
                msg.data.append(float(ly))
            self.landmark_pos_pub.publish(msg)

            # Publish RViz markers
            stamp = self.get_clock().now().to_msg()
            array = MarkerArray()
            for idx, (lx, ly) in enumerate(self.LANDMARKS):
                m = Marker()
                m.header.frame_id = 'map'
                m.header.stamp = stamp
                m.ns = 'landmarks'
                m.id = idx
                m.type = Marker.CYLINDER
                m.action = Marker.ADD
                m.pose.position.x = float(lx)
                m.pose.position.y = float(ly)
                m.pose.position.z = 0.15
                m.pose.orientation.w = 1.0
                m.scale.x = 0.3
                m.scale.y = 0.3
                m.scale.z = 0.3
                m.color.r = 1.0
                m.color.g = 0.85
                m.color.b = 0.0
                m.color.a = 1.0
                array.markers.append(m)
            self.landmark_marker_pub.publish(array)
            self._landmarks_published = True

    def publish_landmark_readings(self):
        """Simulate range/bearing sensor: for each landmark within range,
        compute noisy (range, bearing) and publish.
        Format: [landmark_index, range, bearing, ...] triples."""
        msg = Float32MultiArray()
        for idx, (lx, ly) in enumerate(self.LANDMARKS):
            dx = lx - self.x
            dy = ly - self.y
            r = math.sqrt(dx * dx + dy * dy)
            if r > self.MAX_SENSOR_RANGE:
                continue
            phi = math.atan2(dy, dx) - self.theta
            # wrap bearing into (-pi, pi]
            phi = math.atan2(math.sin(phi), math.cos(phi))
            # add Gaussian noise
            r_noisy = r + random.gauss(0, self.SENSOR_SIGMA_R)
            phi_noisy = phi + random.gauss(0, self.SENSOR_SIGMA_PHI)
            phi_noisy = math.atan2(math.sin(phi_noisy), math.cos(phi_noisy))
            msg.data.extend([float(idx), float(r_noisy), float(phi_noisy)])
        self.landmark_reading_pub.publish(msg)

    @staticmethod
    def _wrap_angle(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def _approach(current, target, max_step):
        diff = target - current
        if abs(diff) <= max_step:
            return target
        return current + math.copysign(max_step, diff)


def main(args=None):
    rclpy.init(args=args)
    node = SimpleRobot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
