#!/usr/bin/env python

# before running do:
# (T1) $ rosrun baxter_tools enable_robot.py -e
# (T1) $ rosrun baxter_tools tuck_arms.py -u
# (T2) $ rosrun baxter_interface joint_trajectory_action_server.py --mode position
#
# Copyright (c) 2013-2015, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Baxter RSDK Joint Trajectory Action Client Example
"""
import argparse
import sys
import subprocess
import struct
import numpy as np
from copy import copy
import matplotlib.pyplot as plt

import cv2
import cv_bridge

import rospy

import actionlib

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint,
)

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

from std_msgs.msg import (
    Header,
    Empty,
)

from sensor_msgs.msg import (
    Image,
)

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

import baxter_interface

from baxter_interface import CHECK_VERSION

class Draw():
    def __init__(self):
        self.limb = 'left'
        ns = "ExternalTools/" + self.limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)
        self._verbose = True

    def ik_request(self, pose):
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
        try:
            resp = self._iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False
        # Check if result valid, and type of seed ultimately used to get solution
        # convert rospy's string representation of uint8[]'s to int's
        resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
        limb_joints = {}
        if (resp_seeds[0] != resp.RESULT_INVALID):
            seed_str = {
                        ikreq.SEED_USER: 'User Provided Seed',
                        ikreq.SEED_CURRENT: 'Current Joint Angles',
                        ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                       }.get(resp_seeds[0], 'None')
            if self._verbose:
                print("IK Solution SUCCESS - Valid Joint Solution Found from Seed Type: {0}".format(
                         (seed_str)))
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            if self._verbose:
                print("IK Joint Solution:\n{0}".format(limb_joints))
                print("------------------")
        else:
            rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
            return False
        return limb_joints

    def draw_line(self, x1, y1, x2, y2):
        # draw a line from (x1,y1) to (x2,y2)

        limb = 'left'
        traj = Trajectory(limb)
        rospy.on_shutdown(traj.stop)
        # Command Current Joint Positions first
        limb_interface = baxter_interface.limb.Limb(limb)
        current_angles = [limb_interface.joint_angle(joint) for joint in limb_interface.joint_names()]
        traj.add_point(current_angles, 5.0)

        p1 = [-0.08000397926829805, -0.9999781166910306, -1.189968899785275, 1.9400238130755056, 0.6699952259595108,
              1.030009435085784, -0.4999997247485215]

        #    p1 = positions[limb]
        traj.add_point(p1, 10.0)

        overhead_orientation = Quaternion(
            x=0.0,
            y=1.0,
            z=0.0,
            w=0.0)

        t = 20

        # go from above the first point to the paper

        min_z = -0.005
        max_z = 0.1
        z_step = 0.01

        for z_k in np.arange(max_z, min_z, -1 * z_step):
            pose = Pose(position=Point(x=x1, y=y1, z=z_k),
                        orientation=overhead_orientation)
            joint_angles = self.ik_request(pose)
            print ('z_k', z_k, joint_angles)
            if joint_angles:
                p = [joint_angles['left_s0'], joint_angles['left_s1'], joint_angles['left_e0'], joint_angles['left_e1'],
                     joint_angles['left_w0'], joint_angles['left_w1'], joint_angles['left_w2']]
                traj.add_point(p, t)
                t += 0.3

        traj.start()
        print ('t', t)
        traj.wait(30)
        traj.stop()
        traj.clear('left')

        # draw the line on the paper

        t = 10.0
        Dx = x2 - x1
        Dy = y2 - y1

        step_size = 0.01

        dx = step_size * Dx / np.sqrt(Dx ** 2 + Dy ** 2)
        dy = step_size * Dy / np.sqrt(Dx ** 2 + Dy ** 2)

        N_step = int(np.floor(max(Dx / dx, Dy / dy)))

        t += 1.0

        for k in range(N_step):
            pose = Pose(position=Point(x=x1 + k * dx, y=y1 + k * dy, z=min_z),
                        orientation=overhead_orientation)
            joint_angles = self.ik_request(pose)
            print ('k', k, joint_angles)
            if joint_angles:
                p = [joint_angles['left_s0'], joint_angles['left_s1'], joint_angles['left_e0'], joint_angles['left_e1'],
                     joint_angles['left_w0'], joint_angles['left_w1'], joint_angles['left_w2']]
                traj.add_point(p, t)
                t += 0.4

        traj.start()
        print ('t', t)
        traj.wait(30)
        traj.stop()
        traj.clear('left')

        # lift the pencil up

        t = 10.0
        t += 1.0

        for z_k in np.arange(min_z, max_z, z_step):
            pose = Pose(position=Point(x=x2, y=y2, z=z_k),
                        orientation=overhead_orientation)
            joint_angles = self.ik_request(pose)
            print ('z_k', z_k, joint_angles)
            if joint_angles:
                p = [joint_angles['left_s0'], joint_angles['left_s1'], joint_angles['left_e0'], joint_angles['left_e1'],
                     joint_angles['left_w0'], joint_angles['left_w1'], joint_angles['left_w2']]
                traj.add_point(p, t)
                t += 0.2

        traj.start()
        print ('t', t)
        traj.wait(30)
        traj.stop()
        traj.clear('left')

    def draw_curve(self, x_vec, y_vec):
        # draw a curve that passes through x_vec and y_vec coordinates
        limb = 'left'
        traj = Trajectory(limb)
        rospy.on_shutdown(traj.stop)
        # Command Current Joint Positions first
        limb_interface = baxter_interface.limb.Limb(limb)
        current_angles = [limb_interface.joint_angle(joint) for joint in limb_interface.joint_names()]
        traj.add_point(current_angles, 5.0)

        p1 = [-0.08000397926829805, -0.9999781166910306, -1.189968899785275, 1.9400238130755056, 0.6699952259595108,
              1.030009435085784, -0.4999997247485215]

        #    p1 = positions[limb]
        traj.add_point(p1, 10.0)

        overhead_orientation = Quaternion(
            x=0.0,
            y=1.0,
            z=0.0,
            w=0.0)

        t = 20

        # go from above the first point to the paper

        min_z = -0.009
        max_z = 0.1
        z_step = 0.01

        for z_k in np.arange(max_z, min_z, -1 * z_step):
            pose = Pose(position=Point(x=x_vec[0], y=y_vec[0], z=z_k),
                        orientation=overhead_orientation)
            joint_angles = self.ik_request(pose)
            print ('z_k', z_k, joint_angles)
            if joint_angles:
                p = [joint_angles['left_s0'], joint_angles['left_s1'], joint_angles['left_e0'], joint_angles['left_e1'],
                     joint_angles['left_w0'], joint_angles['left_w1'], joint_angles['left_w2']]
                traj.add_point(p, t)
                t += 0.3

        traj.start()
        print ('t', t)
        traj.wait(30)
        traj.stop()
        traj.clear('left')

        # draw the line on the paper
        t = 10.0

        for p_k in range(len(x_vec)-1):

            x1 = x_vec[p_k]
            y1 = y_vec[p_k]
            x2 = x_vec[p_k+1]
            y2 = y_vec[p_k+1]
            Dx = x2 - x1
            Dy = y2 - y1

            step_size = 0.01

            if Dx != 0.0 and Dy != 0.0:
                dx = step_size * Dx / np.sqrt(Dx ** 2 + Dy ** 2)
                dy = step_size * Dy / np.sqrt(Dx ** 2 + Dy ** 2)
                N_step = max(1,int(np.floor(max(Dx / dx, Dy / dy))))
            elif Dx == 0 and Dy != 0.0:
                dx = 0.0
                dy = step_size * np.sign(Dy)
                N_step = max(1, int(np.floor(np.abs(Dy / dy))))
            elif Dx != 0 and Dy == 0.0:
                dx = step_size * np.sign(Dx)
                dy = 0.0
                N_step = max(1, int(np.floor(np.abs(Dx / dx))))
            else:
                N_step = 0

            # t += 3.0
            t +=0.5
            print N_step
            for k in range(N_step):
                pose = Pose(position=Point(x=x1 + k * dx, y=y1 + k * dy, z=min_z),
                            orientation=overhead_orientation)
                joint_angles = self.ik_request(pose)
                #print ('k', k, joint_angles)
                if joint_angles:
                    p = [joint_angles['left_s0'], joint_angles['left_s1'], joint_angles['left_e0'], joint_angles['left_e1'],
                         joint_angles['left_w0'], joint_angles['left_w1'], joint_angles['left_w2']]
                    traj.add_point(p, t)
                    print('p_k',p_k,'k',k,'x',x1 + k * dx, 'y', y1 + k * dy, 't',t)
                    # t += 0.4
                    t += 0.1

        traj.start()
        print ('t', t)
        traj.wait(t*1.5)
        traj.stop()
        traj.clear('left')

        # lift the pencil up

        t = 10.0
        t += 1.0

        for z_k in np.arange(min_z, max_z, z_step):
            pose = Pose(position=Point(x=x2, y=y2, z=z_k),
                        orientation=overhead_orientation)
            joint_angles = self.ik_request(pose)
            print ('z_k', z_k, joint_angles)
            if joint_angles:
                p = [joint_angles['left_s0'], joint_angles['left_s1'], joint_angles['left_e0'],
                     joint_angles['left_e1'],
                     joint_angles['left_w0'], joint_angles['left_w1'], joint_angles['left_w2']]
                traj.add_point(p, t)
                t += 0.2

        traj.start()
        print ('t', t)
        traj.wait(t*1.50)
        traj.stop()
        traj.clear('left')

    # def move_to_neutral(self, timeout=15.0):
    #     angles = dict(zip(self.joint_names(), [0.0, -0.55, 0.0, 0.75, 0.0, 1.26, 0.0]))
    #     return self.move_to_joint_positions(angles, timeout)



class Trajectory(object):
    def __init__(self, limb):
        ns = 'robot/limb/' + limb + '/'
        self._client = actionlib.SimpleActionClient(
            ns + "follow_joint_trajectory",
            FollowJointTrajectoryAction,
        )
        self._goal = FollowJointTrajectoryGoal()
        self._goal_time_tolerance = rospy.Time(0.1)
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        server_up = self._client.wait_for_server(timeout=rospy.Duration(10.0))
        if not server_up:
            rospy.logerr("Timed out waiting for Joint Trajectory"
                         " Action Server to connect. Start the action server"
                         " before running example.")
            rospy.signal_shutdown("Timed out waiting for Action Server")
            sys.exit(1)
        self.clear(limb)

    def add_point(self, positions, time):
        point = JointTrajectoryPoint()
        point.positions = copy(positions)
        point.time_from_start = rospy.Duration(time)
        self._goal.trajectory.points.append(point)

    def start(self):
        self._goal.trajectory.header.stamp = rospy.Time.now()
        self._client.send_goal(self._goal)

    def stop(self):
        self._client.cancel_goal()

    def wait(self, timeout=15.0):
        self._client.wait_for_result(timeout=rospy.Duration(timeout))

    def result(self):
        return self._client.get_result()

    def clear(self, limb):
        self._goal = FollowJointTrajectoryGoal()
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        self._goal.trajectory.joint_names = [limb + '_' + joint for joint in \
            ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]

def send_image(path):
    """
    Send the image located at the specified path to the head
    display on Baxter.

    @param path: path to the image file to load and send
    """
    img = cv2.imread(path)
    msg = cv_bridge.CvBridge().cv2_to_imgmsg(img, encoding="bgr8")
    pub = rospy.Publisher('/robot/xdisplay', Image, latch=True, queue_size=1)
    pub.publish(msg)
    # Sleep to allow for image to be published.
    rospy.sleep(1)

def track_ball():
    # cap = cv2.VideoCapture('green_ball.avi')
    cap = cv2.VideoCapture('output.avi')
    # cap = cv2.VideoCapture(0)
    # 121
    # 64.43
    # 50.39


    greenLower = (29, 86, 6)
    greenUpper = (90, 255, 255)

    thefile = open('test.txt', 'w')

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cv2.imshow('mask', mask)

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            print center

            # only proceed if the radius meets a minimum size
            if radius > 3:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                thefile.write(str(center[0]) + ',' + str(center[1]) + '\n')
            cv2.imshow("Frame", frame)

    cap.release()
    cv2.destroyAllWindows()

def center_to_cord():
    x_vec  = []
    y_vec = []
    with open('test.txt') as f:
        lines = f.read().splitlines()
    for line in lines:
        cord = line.split(',')
        x_vec.append(float(cord[0]))
        y_vec.append(float(cord[1]))

    x_max = np.max(x_vec)
    x_min = np.min(x_vec)
    x_center = 0.5*(x_max+x_min)
    x_d = x_max - x_min
    y_max = np.max(y_vec)
    y_min = np.min(y_vec)
    y_center = 0.5*(y_max+y_min)
    y_d = y_max - y_min

    x_vec_n = []
    y_vec_n = []
    for n in range(len(x_vec)):
        x_vec_n.append(0.1*(x_vec[n]-x_center)/x_d+0.5+0.15)
        y_vec_n.append(0.1*(y_vec[n]-y_center)/y_d)
    return x_vec_n, y_vec_n



def main():
    """RSDK Joint Trajectory Example: Simple Action Client

    Creates a client of the Joint Trajectory Action Server
    to send commands of standard action type,
    control_msgs/FollowJointTrajectoryAction.

    Make sure to start the joint_trajectory_action_server.py
    first. Then run this example on a specified limb to
    command a short series of trajectory points for the arm
    to follow.
    """
    # arg_fmt = argparse.RawDescriptionHelpFormatter
    # parser = argparse.ArgumentParser(formatter_class=arg_fmt,
    #                                  description=main.__doc__)
    # required = parser.add_argument_group('required arguments')
    # required.add_argument(
    #     '-l', '--limb', required=True, choices=['left', 'right'],
    #     help='send joint trajectory to which limb'
    # )
    # args = parser.parse_args(rospy.myargv()[1:])



    dr = Draw()

    limb = 'left'

    print("Initializing node... ")
    rospy.init_node("rsdk_joint_trajectory_client_%s" % (limb,))
  #  rospy.init_node('rsdk_xdisplay_image', anonymous=True)

  #   plt.figure(1)
  #   plt.plot(range(1, 10))
    # plt.show()
    # plt.savefig('foo.png')
    # send_image('foo.png')



    print("Getting robot state... ")
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    print("Enabling robot... ")
    rs.enable()
    #limb_interface = baxter_interface.limb.Limb(limb)
    #limb_interface = baxter_interface.limb.Limb(limb)
    #print limb_interface.move_to_neutral()
    print("Running. Ctrl-c to quit")
    send_image('Face_normal.png')

    subprocess.call(['rosrun','baxter_tools', 'camera_control.py', '-c', 'right_hand_camera'])
    subprocess.call(['rosrun', 'baxter_tools', 'camera_control.py', '-o', 'head_camera', '-r', '1280x800'])

    send_image('Face_ready.png')
    raw_input("Press Enter to continue...")
    process = subprocess.Popen(["rosrun", "image_view", "video_recorder", "image:=/cameras/head_camera/image"])

    # send_image('Face_record_5.png')
    # plt.pause(1)
    # send_image('Face_record_4.png')
    # plt.pause(1)
    send_image('Face_record_3.png')
    plt.pause(1)
    send_image('Face_record_2.png')
    plt.pause(1)
    send_image('Face_record_1.png')
    plt.pause(1)
    send_image('Face_record_0.png')
    plt.pause(1)
    process.kill()
    print('Done Recording')
    send_image('Face_normal.png')

    track_ball()

    x_vec_n, y_vec_n = center_to_cord()

    plt.figure()
    plt.plot(x_vec_n, y_vec_n)
    plt.pause(1)
    raw_input("Press Enter to continue...")

    dr.draw_curve(x_vec_n, y_vec_n)


if __name__ == "__main__":
    main()
