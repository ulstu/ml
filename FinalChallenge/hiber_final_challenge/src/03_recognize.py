#!/usr/bin/env python

from __future__ import print_function
import rospy
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose, Point, Quaternion
from tf import TransformListener
from geometry_msgs.msg import PoseWithCovarianceStamped
import cv2
import numpy as np
import sys
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import urllib2
import datetime


class GoToPose():
    goals = [] 
    cur_pose = None
    not_reached_goal = True

    def send_goal(self, nickname, x, y):
        tmpl = "http://summerschool.simcase.ru/challenge/standings/?name={}&x={}&y={}"
        surl = tmpl.format(nickname, x, y)
        res = urllib2.urlopen(surl).read()
        rospy.loginfo(res)

    def calc_pos(self, object):
        # put here calculations of object positions
        x = 0
        y = 0
        #send_goal('hiber', x, y)

    def __init__(self):
        # What to do if shut down (e.g. Ctrl-C or failure)
        rospy.on_shutdown(self.shutdown)
        
        self.bridge = CvBridge()
        self.image_received = False

        # Connect image topic
        img_topic = "/camera/rgb/image_raw"
        self.image_sub = rospy.Subscriber(img_topic, Image, self.image_callback)
        r = rospy.Rate(1)
        while not rospy.is_shutdown():
            r.sleep()

        rospy.sleep(1)

    def image_callback(self, data):
        rospy.loginfo('image received')
        try:
            # Convert image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
            self.image_received = True
            self.image = cv_image
            self.image = cv2.medianBlur(self.image, 5)
            cimg = self.image #cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            cv2.imwrite('/home/fist/catkin_ws/src/hiber_final_challenge/images/img_{}.png'.format(datetime.datetime.now()), cimg)
            circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
            if not circles is None:
                cv2.waitKey(1)
                rospy.loginfo('circles found: {}'.format(len(circles[0,:])))
                circles = np.uint16(np.around(circles))
                for i in circles[0,:]:
                    # draw the outer circle
                    cv2.circle(cimg,(i[0], i[1]), i[2],(0,255,0),2)
                    # draw the center of the circle
                    cv2.circle(cimg,(i[0], i[1]),2,(0,0,255),3)
                    #go only to first sphere
                    circle_radius = 0.5
                    visual_radius = i[2]
                    rospy.loginfo('!!!CIRCLE RADIUS: {}'.format(visual_radius))
                    self.calc_pos(i)
                    break
                cv2.waitKey(1)
                cv2.imshow('circles',cimg)
                cv2.imwrite('/home/fist/catkin_ws/src/hiber_final_challenge/images/image.jpg', cimg)
                cv2.waitKey(4)
            else:
                cv2.waitKey(1)
                cv2.imshow('circles',self.image)
                cv2.imwrite('/home/fist/catkin_ws/src/hiber_final_challenge/images/image.jpg', self.image)
                cv2.waitKey(4)


        except CvBridgeError as e:
            print(e)

    def take_picture(self, img_title):
        if self.image_received:
            # Save an image
            cv2.imwrite(img_title, self.image)
            return True
        else:
            return False

    def shutdown(self):
        cv2.destroyAllWindows()
        rospy.loginfo("Stop")
        rospy.sleep(1)

if __name__ == '__main__':
    try:
        rospy.init_node('hiber_final_challenge_opencv', anonymous=False)
        navigator = GoToPose()
        rospy.sleep(1)

    except rospy.ROSInterruptException:
        rospy.loginfo("Ctrl-C caught. Quitting")

