#!/usr/bin/env python3
import rospy, math, heapq, numpy as np, tf
from geometry_msgs.msg import Twist
from sensor_msgs.msg  import LaserScan
from nav_msgs.msg      import Odometry, OccupancyGrid
from enum              import Enum, auto

class State(Enum):
    FORWARD = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    AVOID_OBSTACLE = auto()
    DONE = auto()

class CleanerBot:
    def __init__(self):
        rospy.init_node('cleaner_bot')

        # cmd_vel publisher
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # odom + scan subscribers
        self.odom = None
        rospy.Subscriber('/odom', Odometry, self.update_odom)
        self.dist_front = self.dist_flank = float('inf')
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        # internal rate
        self.rate = rospy.Rate(10)

        # safety & speed
        self.thresh_front = 0.30
        self.forward_speed = 0.10

        # wait for the map once
        rospy.loginfo("⏳ Waiting for /map…")
        map_msg = rospy.wait_for_message('/map', OccupancyGrid)
        self.process_map(map_msg)

        # coverage publisher
        self.coverage_pub = rospy.Publisher('/coverage', OccupancyGrid, queue_size=1)
        self.publish_coverage()
        rospy.loginfo("✅ Map & coverage grid ready.")

    def process_map(self, msg):
        # extract metadata
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution   = msg.info.resolution
        self.map_origin_x   = msg.info.origin.position.x
        self.map_origin_y   = msg.info.origin.position.y

        # reshape the flat data into a 2D array
        data = np.array(msg.data).reshape((self.map_height, self.map_width))

        # build coverage grid: 0 = unexplored, 2 = obstacle
        self.coverage_grid = np.zeros_like(data, dtype=np.uint8)
        self.coverage_grid[data != 0] = 2

    def publish_coverage(self):
        cov = OccupancyGrid()
        cov.header.stamp = rospy.Time.now()
        cov.header.frame_id = 'map'
        cov.info.resolution = self.map_resolution
        cov.info.width      = self.map_width
        cov.info.height     = self.map_height
        cov.info.origin.position.x = self.map_origin_x
        cov.info.origin.position.y = self.map_origin_y
        cov.data = list(self.coverage_grid.flatten())
        self.coverage_pub.publish(cov)

    def update_odom(self, msg):
        self.odom = msg

    def scan_callback(self, msg):
        r = np.array(msg.ranges)
        r = np.where(np.isfinite(r) & (r>0), r, np.inf)
        L = len(r)
        n45 = int(L*45/360)
        n90 = int(L*90/360)
        front = np.concatenate((r[:n45], r[-n45:]))
        flank = np.concatenate((r[n45:n90], r[-n90:-n45]))
        self.dist_front = np.min(front)
        self.dist_flank = np.min(flank)

    def get_yaw(self):
        if not self.odom:
            self.odom = rospy.wait_for_message('/odom', Odometry)
        q = self.odom.pose.pose.orientation
        return tf.transformations.euler_from_quaternion([q.x,q.y,q.z,q.w])[2]

    def turn_precise(self, angle_deg):
        target = (self.get_yaw() + math.radians(angle_deg)) % (2*math.pi)
        cmd = Twist()
        while not rospy.is_shutdown():
            yaw = self.get_yaw()
            diff = (target - yaw + math.pi) % (2*math.pi) - math.pi
            if abs(diff) < math.radians(1): break
            cmd.angular.z = math.copysign(max(0.15, abs(diff)), diff)
            self.pub.publish(cmd)
            self.rate.sleep()
        rospy.loginfo(f"✅ Turned {angle_deg}°")

    def move_to_cell(self, tx, ty, tol=0.05):
        cmd = Twist()
        while not rospy.is_shutdown():
            if not self.odom:
                self.rate.sleep()
                continue
            px = self.odom.pose.pose.position.x
            py = self.odom.pose.pose.position.y
            dx, dy = tx-px, ty-py
            dist = math.hypot(dx, dy)
            if dist < tol: break

            yaw = self.get_yaw()
            ang = math.atan2(dy, dx)
            err = (ang - yaw + math.pi) % (2*math.pi) - math.pi

            cmd.linear.x  = self.forward_speed
            cmd.angular.z = 0.5*err

            if self.dist_front < self.thresh_front:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.4

            self.pub.publish(cmd)
            self.rate.sleep()

        self.pub.publish(Twist())
        rospy.loginfo(f"✅ Reached ({tx:.2f},{ty:.2f})")

    def a_star(self, grid, start, goal):
        rows, cols = grid.shape
        open_set = [(0, start)]
        came_from = {}
        g_score   = {start: 0}

        def neighbors(p):
            i, j = p
            for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < rows and 0 <= nj < cols and grid[ni, nj] != 2:
                    yield (ni, nj)

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            for n in neighbors(current):
                tentative = g_score[current] + 1
                if n not in g_score or tentative < g_score[n]:
                    came_from[n] = current
                    g_score[n]   = tentative
                    f = tentative + abs(n[0]-goal[0]) + abs(n[1]-goal[1])
                    heapq.heappush(open_set, (f, n))
        return None

    def find_nearest_unvisited(self, grid, start):
        from collections import deque
        visited = {start}
        queue   = deque([start])
        while queue:
            cell = queue.popleft()
            if grid[cell[0], cell[1]] == 0:
                return cell
            for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
                n = (cell[0]+di, cell[1]+dj)
                if (0 <= n[0] < grid.shape[0] and 0 <= n[1] < grid.shape[1]
                    and n not in visited and grid[n[0], n[1]] != 2):
                    visited.add(n)
                    queue.append(n)
        return None

    def follow_path(self, path):
        for cell in path:
            x, y = self.map_origin_x + (cell[1]+0.5)*self.map_resolution, \
                   self.map_origin_y + (cell[0]+0.5)*self.map_resolution
            self.move_to_cell(x, y)
            self.coverage_grid[cell] = 1

    def is_cleaning_complete(self):
        return not np.any(self.coverage_grid == 0)

    def ccpp_clean(self):
        for row in range(self.map_height):
            cols = range(self.map_width) if row % 2 == 0 else reversed(range(self.map_width))
            for col in cols:
                if self.coverage_grid[row, col] == 0:
                    x = self.map_origin_x + (col + 0.5)*self.map_resolution
                    y = self.map_origin_y + (row + 0.5)*self.map_resolution
                    rospy.loginfo(f"🧭 Moving to ({row},{col}) → ({x:.2f},{y:.2f})")
                    try:
                        self.move_to_cell(x, y)
                        self.coverage_grid[row, col] = 1
                    except Exception as e:
                        rospy.logwarn(f"⚠️ move_to_cell failed ({e}) — A* fallback")
                        start = self.get_robot_cell()
                        nxt   = self.find_nearest_unvisited(self.coverage_grid, start)
                        if nxt:
                            path = self.a_star(self.coverage_grid, start, nxt)
                            if path:
                                self.follow_path(path)
            rospy.loginfo(f"✅ Finished row {row}")

    def get_robot_cell(self):
        p = self.odom.pose.pose.position
        i = int((p.y - self.map_origin_y) / self.map_resolution)
        j = int((p.x - self.map_origin_x) / self.map_resolution)
        return (i, j)

    def run(self):
        rospy.sleep(1.0)
        self.turn_precise(30)
        rospy.sleep(1.0)

        # Keep sweeping until everything is non-zero
        while not rospy.is_shutdown() and not self.is_cleaning_complete():
            rospy.loginfo("🔄 Starting new cleaning pass…")
            self.ccpp_clean()
            rospy.loginfo("🔄 Cleaning pass complete; checking for remaining cells…")

        rospy.loginfo("🧼 Cleaning complete! Stopping robot.")
        self.pub.publish(Twist())
        
if __name__ == '__main__':
    try:
        bot = CleanerBot()
        bot.run()
    except rospy.ROSInterruptException:
        pass
