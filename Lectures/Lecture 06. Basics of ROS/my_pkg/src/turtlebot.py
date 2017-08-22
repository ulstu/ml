#!/usr/bin/env python

# TurtleBot must have minimal.launch & amcl_demo.launch
# running prior to starting this script
# For simulation: launch gazebo world & amcl_demo prior to run this script

import rospy
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose, Point, Quaternion
from tf import TransformListener
from geometry_msgs.msg import PoseWithCovarianceStamped

class GoToPose():
    goals = [] 
    cur_pose = None
    def __init__(self):
	    self.goal_sent = False
	    self.tf = TransformListener()

		# What to do if shut down (e.g. Ctrl-C or failure)
		rospy.on_shutdown(self.shutdown)
		
		# Tell the action client that we want to spin a thread by default
		self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
		rospy.loginfo("Wait for the action server to come up")

		# Allow up to 15 seconds for the action server to come up
		self.move_base.wait_for_server(rospy.Duration(15))
	    rospy.Subscriber("amcl_pose", PoseWithCovarianceStamped, self.pose_callback)

    def pose_callback(self, data):
        self.cur_pose = data
#        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data)

    def get_position(self):
        print self.cur_pose

    def goto(self, pos, quat):

        # Send a goal
        self.goal_sent = True
		goal = MoveBaseGoal()
		goal.target_pose.header.frame_id = 'base_link' #'map'
		goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = Pose(Point(pos['x'], pos['y'], 0.000),
                                     Quaternion(quat['r1'], quat['r2'], quat['r3'], quat['r4']))
		# Start moving
        self.move_base.send_goal(goal)
        rospy.loginfo('goal is sended')

		# Allow TurtleBot up to 60 seconds to complete task
		success = self.move_base.wait_for_result(rospy.Duration(60)) 

        state = self.move_base.get_state()
        result = False
        rospy.loginfo('result is received')

        if success and state == GoalStatus.SUCCEEDED:
            # We made it!
            result = True
        else:
            self.move_base.cancel_goal()

        self.goal_sent = False
        return result

    def addgoal(self, x, y):
        position = {'x': x, 'y' : y}
        quaternion = {'r1' : 0.000, 'r2' : 0.000, 'r3' : 0.000, 'r4' : 1.000}
        self.goals.append({'pos': position, 'quat': quaternion})
    
    def move(self):
        rospy.loginfo('move method is invoked')

        self.get_position()
        self.addgoal(1.22, 2.55)
        self.addgoal(-1.22, -2.55)
        self.addgoal(1.22, 2.55)
        self.addgoal(-1.22, -2.55)

        for goal in self.goals:
            rospy.loginfo("Go to (%s, %s) pose", goal['pos']['x'], goal['pos']['y'])
            success = self.goto(goal['pos'], goal['quat'])
            if success:
                rospy.loginfo("Hooray, reached the desired pose")
            else:
                rospy.loginfo("The base failed to reach the desired pose")
            self.get_position()
            rospy.sleep(1)
 

    def shutdown(self):
        if self.goal_sent:
            self.move_base.cancel_goal()
        rospy.loginfo("Stop")
        rospy.sleep(1)

if __name__ == '__main__':
    try:
        rospy.init_node('nav_test', anonymous=False)
#        navigator = GoToPose()
#        navigator.move()
        rospy.sleep(1)

    except rospy.ROSInterruptException:
        rospy.loginfo("Ctrl-C caught. Quitting")

