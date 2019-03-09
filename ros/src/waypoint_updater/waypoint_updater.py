#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
import numpy as np
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.waypoint_2d = None
        self.kdtree = None
        self.pose = None

        self.loop()
        rospy.spin()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.kdtree:
                closest_point_index = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_point_index)
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.position.x
        y = self.pose.position.y
        closest_index =  self.kdtree.query([x,y],1)[1]

        closest_coord = self.waypoint_2d[closest_index]
        pre_coord = self.waypoint_2d[closest_index-1]

        cl_vect = np.array(closest_coord)
        pr_vect = np.array(pre_coord)
        cur_vect = np.array([x,y])

        val = np.dot(cl_vect-pr_vect,cl_vect-cur_vect)
        if val<0:
            closest_index = (closest_index+1)%len(self.waypoint_2d)
        return closest_index
    def publish_waypoints(self,index):
        lane = Lane()
        lane.header = self.waypoint_list.header
        lane.waypoints = self.waypoint_list.waypoints[index:index+LOOKAHEAD_WPS]
        self.final_waypoints_pub.publish(lane)




    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg.pose
        pass

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.waypoint_list = waypoints
        if not self.waypoint_2d:
            self.waypoint_2d = [[waypoint.pose.pose.position.x,waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.kdtree = KDTree(self.waypoint_2d)
        pass

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
