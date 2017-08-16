#!/usr/bin/env python

import rospy
import sys
from geometry_msgs.msg import Twist

class GoForward():
    def __init__(self):
        rospy.init_node('my_pkg', anonymous=False)
    	rospy.loginfo("To stop Turtle CTRL + C")

        rospy.on_shutdown(self.shutdown)
        self.cmd_vel = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
     
        r = rospy.Rate(1)
        i = 1
        while not rospy.is_shutdown():
            rospy.loginfo(i)

            move_cmd = Twist()
            move_cmd.linear.x = 1
            move_cmd.linear.y = 0
            move_cmd.angular.z = 10 / i
            self.cmd_vel.publish(move_cmd)
            r.sleep() 
            i += 1                       
        
    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop TurtleBot")
	    # a default Twist has linear.x of 0 and angular.z of 0.  So it'll stop TurtleBot
        self.cmd_vel.publish(Twist())
	    # sleep just makes sure TurtleBot receives the stop command prior to shutting down the script
        rospy.sleep(1)
 
if __name__ == '__main__':
    try:
        GoForward()
    except:
        rospy.loginfo("GoForward node terminated.")
        rospy.loginfo(sys.exc_info()[:2])

