#!/usr/bin/env python

import os
import time
import math
import numpy as np
import configparser
from datetime import datetime
import receiver
import sender
import statistics
import queue

def joint_distance_calculator(kps_to_render,kp):
    distance = []
    normalizer = math.sqrt(pow((kp[22]-kp[24]),2) + pow((kp[23]-kp[25]),2)) #spalle

    for i in range(len(kps_to_render)):
        # print("number of arms: {}".format(len(kps_to_render)))

        segment = kps_to_render[i]
        #  x1, y1, x2, y2
        xa =kp[segment[0]]
        ya = kp[segment[1]]
        xb = kp[segment[2]]
        yb = kp[segment[3]]
       #((kp[segment[0]], kp[segment[1]]), (kp[segment[2]], kp[segment[3]])
        distance.append((math.sqrt(pow((xa-xb),2) + pow((ya-yb),2)))/normalizer)
    return distance


def no_ex_cycle_control(string_from_tcp_ID,ex_string):
    while ex_string == "":
        ex_string = TCP_listen_check_4_string(string_from_tcp_ID, ex_string)
        # print("listen to port for command...")
    return ex_string


def ex_string_to_ID(ex_string):
    # read from ex_info
    config = configparser.ConfigParser()
    config.read('exercise_info.ini')
    sections = config.sections()
    # print("sections are : {}".format(sections))
    ex_string=ex_string.rstrip("\n")
    for exercise in sections:

        if exercise == ex_string:
            # config.get("test", "foo")

            ID = int(config.get(exercise, 'ID'))
            return ID

    print("no exercise found")
    ID = 0
    return ID


def ID_to_ex_string(ID):
    # read from ex_info
    config = configparser.ConfigParser()
    config.read('exercise_info.ini')
    sections = config.sections()
    # print("sections are : {}".format(sections))

    for exercise in sections:
        ID_num = int(config[exercise]["ID"])

        if ID_num == ID:
            ex_string = exercise

            return ex_string

    print("no exercise found")
    ex_string = "0"
    return ex_string


def KP_to_render_from_config_file(dictionary):
    config_geometrical = configparser.ConfigParser()

    config_geometrical.read('config.ini')
    KPS_to_render = []
    # print(type(dictionary["segments"]))
    # print(type(dictionary["eva_range"]))
    for limb in dictionary["segments"]:
        # print("analizing arto: {}".format(arto))

        kps = config_geometrical["ALIAS"][limb]

        kps = [int(x) for x in kps.split(",")]
        KPS_to_render.append(kps)
    dictionary["KPS_to_render"] = KPS_to_render

    # print(dictionary)
    return dictionary


def ex_string_to_config_param(ex_string):
    # read from ex_info
    config = configparser.ConfigParser()
    config.read('exercise_info.ini')
    sections = config.sections()

    # print("sections are : {}".format(sections))

    for exercise in sections:

        if exercise == ex_string:
            # config.get("test", "foo")

            segments = config.get(exercise, 'segments_to_render')
            segments = segments.split(',')
            eva_range = config.get(exercise, 'evaluation_range')
            eva_range = [int(x) for x in eva_range.split(",")]

            dictionary = {
                "segments": segments,
                "eva_range": eva_range,
                "KPS_to_render": []
            }
            # print("dictionary : {}".format(dictionary))

            dictionary = KP_to_render_from_config_file(dictionary)

            return dictionary

    print("no exercise found")
    dictionary = {}
    return dictionary


def wait_for_keypoints(queuekp):
    keypoints = []

    while not keypoints:
        
        try:
            keypoints = queuekp.get(False)
        except queue.Empty:
            #print("no KP data aviable: queue empty")
            pass

        else:

            if not keypoints:
                print("no valid kp")
            else:
                return keypoints


def check_for_string_in_memory(multiprocessing_value_slot):
    ex_ID = multiprocessing_value_slot
    ex_string_selected = ID_to_ex_string(ex_ID)

    return ex_string_selected


def TCP_listen_check_4_string(string_from_tcp_ID,ex_string_recived):

    if string_from_tcp_ID.value == 0:
        ex_string_recived = "stop"
        return ex_string_recived
    elif string_from_tcp_ID.value == 10:
        ex_string_recived = "pause"
        return ex_string_recived
    elif string_from_tcp_ID.value == 100:
        ex_string_recived = "start"
        return ex_string_recived
    else:
        ex_string_recived = ID_to_ex_string(string_from_tcp_ID.value)

        return ex_string_recived


def write_ex_string_in_shared_memory(ex_string_recived):
    # qui associo alla stringa ricevuta l'ID dell esertcizio

    multiprocessing_value_slot = ex_string_to_ID(ex_string_recived)

    return multiprocessing_value_slot


def findAngle(p1, p2, p3):
    # ci vuole una routine che controlli geometricamente e
    # a livello di tipo che i valori di kp siano reali
    # Get the landmarks
    (x1, y1) = p1
    (x2, y2) = p2
    (x3, y3) = p3

    # Calculate the Angle
    deg = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                       math.atan2(y1 - y2, x1 - x2))

    angle = np.abs(deg)
    if angle > 180:
        angle = 360 - angle

    # print(angle)

    return angle



def kp_geometry_analisys(kp, count, stage, per, dictionary):
    if kp == []:
        print("no KP aviable")
    else:

        eva_range = dictionary["eva_range"]
        kps_to_render = dictionary["KPS_to_render"]
        angle = []

        per = []



        if len(kps_to_render) != 0:
                # print("dictionary: {}".format(dictionary))
            if len(kps_to_render[0]) == 4:

                distance = joint_distance_calculator(kps_to_render,kp)
                print("distance is :", distance)
                print("eva range", eva_range)

                for val in range(len(distance)):


                    # print("angle from EVA : {}".format(angle))
                    p = np.interp(distance[val], (eva_range[1], eva_range[0]), (100, 0))
                    per.append(p)
                    # Check for the dumbbell curls
                    # print("eva range 1 : {}".format(eva_range[1]))
                    # print(a)
                    # print("stage control: {}".format(stage))
                    print("dist_eva: ", distance[val], (eva_range[1])/100)

                    if distance[val] > (eva_range[1])/100:

                        stage[val] = "down"

                    if distance[val] < (eva_range[0])/100 and stage[val] == "down":
                        stage[val] = "up"
                        count[val] += 1

                    print("COUNTING routine :  {} ".format(count))

                return count, stage
                # distance





            elif len(kps_to_render[0]) == 6:
                print("angle")

            # angle:

                for i in range(len(kps_to_render)):
                    # print("number of arms: {}".format(len(kps_to_render)))

                    segment = kps_to_render[i]
                    # print(kp[segment[5]])
                    a = findAngle((kp[segment[0]], kp[segment[1]]), (kp[segment[2]], kp[segment[3]]),
                                  (kp[segment[4]], kp[segment[5]]))
                    angle.append(a)
                    # print("angle from EVA : {}".format(angle))
                    p = np.interp(a, (10, 160), (100, 0))
                    per.append(p)
                    # Check for the dumbbell curls
                    # print("eva range 1 : {}".format(eva_range[1]))
                    # print(a)
                    # print("stage control: {}".format(stage))

                    if a > eva_range[1]:
                        stage[i] = "down"

                    if a < eva_range[0] and stage[i] == "down":
                        stage[i] = "up"
                        count[i] += 1

                    print("COUNTING routine :  {} ".format(count))
                    # print("percentage : {} %".format(per))
                # print("angle : {}".format(angle))
                # print("stage : {}".format(stage))

                return count, stage , per
        else:
            print("no kps to render dictionary error, dictionary: {}".format(dictionary))


def evaluator(EX_global, q,string_from_tcp_ID):
    # printing process id
    print("ID of process running evaluator: {}".format(os.getpid()))

    #time.sleep(3)
    kp = []

    stage = ["", ""]
    per = [0,0]

    count = [0, 0]
    ex_string_from_TCP = ""  # comando arrivato dalla tcp
    ex_string = ""  # comando letto dalla memoria

    while True:

        #time.sleep(0.5)
        ex_string_from_TCP = TCP_listen_check_4_string(string_from_tcp_ID,ex_string_from_TCP)
        #print("TCP ex ID : ", string_from_tcp_ID.value)

        if ex_string_from_TCP == "":

            ex_string_from_TCP = no_ex_cycle_control(string_from_tcp_ID,ex_string_from_TCP)
            #print("count = {}".format(count))

        elif ex_string_from_TCP == "stop":
            #print("stop command detected")
            # refreshing parameter of exercise
            ex_string_from_TCP = ""
            count = [0, 0]
            stage = ["", ""]
            EX_global.value = 0
            ex_string = ""
            #print("count = {}".format(count))
            ex_string_from_TCP = no_ex_cycle_control(string_from_tcp_ID,ex_string_from_TCP)

        elif ex_string_from_TCP == "pause":
            #print("pause command detected")
            # rimane comunque l esercizio in memoria
            ex_string_from_TCP = ""
            #print("count(pause) = {}".format(count))
            ex_string_from_TCP = no_ex_cycle_control(string_from_tcp_ID,ex_string_from_TCP)

        elif ex_string_from_TCP == "start":
            if EX_global.value != 0:
                ex_string = check_for_string_in_memory(EX_global.value)
                #print("an exercise is under evaluation, starting...", ex_string)
                #print("count = {}".format(count))

            else:
                #print("start command error: no exercise selected, waiting for a selection before start")
                ex_string_from_TCP = ""
                #print("count = {}".format(count))
                ex_string_from_TCP = no_ex_cycle_control(string_from_tcp_ID,ex_string_from_TCP)  # da togliere

        else:

            EX_global.value = write_ex_string_in_shared_memory(ex_string_from_TCP)
            #print("{} string wrote in memory".format(ex_string_from_TCP))
            #print("EX_global.value", EX_global.value)

        if EX_global.value != 0:
            if ex_string != "":
                # controllo di aver letto con successo la memoria dopo il comando di start

                #print("read from memory: {}".format(ex_string))

                kp = wait_for_keypoints(q)

                config_param_dictionary = ex_string_to_config_param(ex_string)

                count, stage , per = kp_geometry_analisys(kp, count, stage,per, config_param_dictionary)
                #print("count. ",count)

                packet = str(max(count)) + "," + str(int(max(per)))   
                print("packet", packet)             
                
                sender.send_status(21011, packet,'192.168.10.2')
