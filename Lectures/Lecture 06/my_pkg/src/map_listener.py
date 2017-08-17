#!/usr/bin/env python
import rospy
import math
import random
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


FORWARD_SPEED = 0.5
MIN_SCAN_ANGLE = -30.0 / 180 * 3.14
MAX_SCAN_ANGLE = +30.0 / 180 * 3.14
MIN_DIST_FROM_OBSTACLE = 1.5 # Should be smaller than sensor_msgs::LaserScan::range_max
keepMoving = True

def stop():
    cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)
    move_cmd = Twist()
    move_cmd.linear.x = 0
    move_cmd.linear.y = 0
    move_cmd.angular.z = 0
    cmd_vel.publish(move_cmd)

def turn(angle, speed):
    rospy.loginfo("turn!")
    global keepMoving
    cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)

    angular_speed = speed * 2 * 3.14 / 360
    relative_angle = angle * 2 * 3.14 / 360

    vel_msg = Twist()
    vel_msg.linear.x = 0
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = -abs(angular_speed)

    t0 = rospy.Time.now().to_sec()
    current_angle = 0
    r = rospy.Rate(10)
    while(current_angle < relative_angle):
        cmd_vel.publish(vel_msg)
        t1 = rospy.Time.now().to_sec()
        current_angle = angular_speed * (t1 - t0)
        r.sleep() 

    keepMoving = True
    stop()
    move_forward()


def move_forward():
    rospy.loginfo("move!")
    global keepMoving
    cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)
    r = rospy.Rate(3)
    while not rospy.is_shutdown() and keepMoving:
        move_cmd = Twist()
        move_cmd.linear.x = 0.5
        move_cmd.linear.y = 0
        move_cmd.angular.z = 0
        cmd_vel.publish(move_cmd)
        r.sleep() 
    stop()
    turn(random.randint(120, 240), 50)

def callback(scan):
    global keepMoving
    #rospy.loginfo(rospy.get_caller_id() + "I heard %s", scan)

    isObstacleInFront = False
    # Find the closest range between the defined minimum and maximum angles
    minIndex = math.ceil((MIN_SCAN_ANGLE - scan.angle_min) / scan.angle_increment)
    maxIndex = math.floor((MAX_SCAN_ANGLE - scan.angle_min) / scan.angle_increment)
    res = ""
    for currIndex in range(int(minIndex + 1), int(maxIndex)):
        if scan.ranges[currIndex] < MIN_DIST_FROM_OBSTACLE:
            isObstacleInFront = True
            res += " " + str(scan.ranges[currIndex])
            break
    rospy.loginfo(res)
    if isObstacleInFront:
        rospy.loginfo("Stop!");
        keepMoving = False
    
def listener():
    rospy.init_node('map_listener', anonymous=True)
    rospy.Rate(10).sleep()
    rospy.Subscriber("scan", LaserScan, callback)
    move_forward()
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
