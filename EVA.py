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
import csv

STORY = 5

def write_data_csv(data):
    """
    write data 2 CSV

    :param data: write to a csv file input data (append to the end)
    
    :return: nothing
    """

    f = open('data.csv', 'a')
    writer = csv.writer(f)
        
    writer.writerow(data)

def joint_distance_calculator(kps_to_render,kp):
    """
    calculate distance btw 2 2d point (from all KP select only the segment 2 render)


    :param kps_to_render: ID of the joints under distance evaluation 
    :param kp: actual frame joints detected with their coordinates
    :return: distance between the couple of joints
    """
    distance = []
    normalizer = math.sqrt(pow((kp[2]-kp[4]),2) + pow((kp[3]-kp[5]),2)) #spalle

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
    """
    loop to wait command from coordinator

    :param string_from_tcp_ID: ID of the string coming from the listener process (ports listening)
    :param ex_string: formal string containing the name of the exercise under eval
    :return: ex_string
    """
    while ex_string == "":
        ex_string = TCP_listen_check_4_string(string_from_tcp_ID, ex_string)
        # print("listen to port for command...")
    return ex_string


def ex_string_to_ID(ex_string):
    """
    read from ex_info string and convert to integer ID of exercise

    :param ex_string: string of the name of the exercise
    :param param2: none
    :return: the ID corresponding the exercise string
    """
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

    #print("no exercise found")
    ID = 0
    return ID


def ID_to_ex_string(ID):
    """
    function description

    :param param1: description of param1
    :param param2: description of param2
    :return: description of the function output
    """
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

    #print("no exercise found")
    ex_string = "0"
    return ex_string


def KP_to_render_from_config_file(dictionary):
    """
    function description

    :param param1: description of param1
    :param param2: description of param2
    :return: description of the function output
    """
#extract the Kps of the exercise fron the ini file 
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
    """
    function description

    :param param1: description of param1
    :param param2: description of param2
    :return: description of the function output
    """
#take all info of the exercise from the config files, write on a dictionary
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
            joints_target = config.get(exercise, 'joints_target')
            joints_target = [int(x) for x in joints_target.split(",")]
            threshold = int(config.get(exercise, 'threshold'))
            threshold_count = int(config.get(exercise, 'threshold_count'))
            
            
            
            #print("joints and target : ", joints_target)

            dictionary = {
                "segments": segments,
                "eva_range": eva_range,
                "KPS_to_render": [],
                "joints_target" : joints_target,
                "threshold" : threshold,
                "threshold_count" : threshold_count
                
            }
            #print("dictionary : {}".format(dictionary))

            dictionary = KP_to_render_from_config_file(dictionary)

            return dictionary

    #print("no exercise found")
    dictionary = {}
    return dictionary


def wait_for_keypoints(queuekp):
    """
    function description

    :param param1: description of param1
    :param param2: description of param2
    :return: description of the function output
    """
#loop 4 waiting the queue of KP from mediapipe
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
    """
    function description

    :param param1: description of param1
    :param param2: description of param2
    :return: description of the function output
    """
#return the exercise written in the shared memory (if prresent)
    ex_ID = multiprocessing_value_slot
    ex_string_selected = ID_to_ex_string(ex_ID)

    return ex_string_selected


def TCP_listen_check_4_string(string_from_tcp_ID,ex_string_recived):
    """
    function description

    :param param1: description of param1
    :param param2: description of param2
    :return: description of the function output
    """
#convert the shared memory id to a string identifing the exercise 
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
    """
    function description

    :param param1: description of param1
    :param param2: description of param2
    :return: description of the function output
    """
# qui associo alla stringa ricevuta l'ID dell esertcizio e salvo in shared memory

    multiprocessing_value_slot = ex_string_to_ID(ex_string_recived)

    return multiprocessing_value_slot


def findAngle(p1, p2, p3):
    # da 3 punti 2d ricavo l angolo interno
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

def kp_geometry_analisys_v2(eval_data, kp_history, count, dictionary, stage):
    """
    compute distance from specific  joints of the ex and the compatible target,
    calculate it's velocity over a moving avarege of the distances and
    interpret the stage of th movement based on the sign of the velocity

    :param eval_data: last distance and velocity of the joint target points
    :param kp_history: last n (10) set of keypoint in the past
    :param count: current count of the ex repetitions (permanent in memory)
    :param dictionary: python dict with all info about the current exercise
    (interprete from config files)
    :param stage: the velocity trigger for counting 
    :return: eval_data, count, phase the first two from memory are here fulfilled,
    phase is the velocity descriptor for retroacting motors 
    """
    phase = [0,0]
    to_count = [False,False]
    
    
    w = 0.5
    threshold = dictionary["threshold"]
    threshold_count = dictionary["threshold_count"]
    printV = False
    if printV == True:
        print("kp_history:", kp_history)
        print("eval_data:", eval_data)
        print("count:", count)
        print("dictionary:", dictionary)
        print("dictionary:__ LENGHT", len(dictionary["joints_target"]))
    if len(kp_history) > STORY - 1:

        side = int(len(dictionary["joints_target"])/4) #0-1, 8 valori (4 punti)
        old_value = []

        for hand in range(side):
            IDx_m = dictionary["joints_target"][4*hand]
            IDy_m = dictionary["joints_target"][4*hand+1]

            xa = kp_history[-1][IDx_m]
            xt = ((dictionary["joints_target"][4*hand +2] )/100 )*480
            ya = kp_history[-1][IDy_m]
            yt = ((dictionary["joints_target"][4*hand +3] )/100 )*640
            #D_sx = math.sqrt(pow((xa - xt), 2) + pow((ya - yt), 2))
            D_time = []

            for i in range(len(kp_history)):
                XM_i = kp_history[i][IDx_m]
                YM_i = kp_history[i][IDy_m]
                D_i = pow((XM_i - xt), 2) + pow((YM_i - yt), 2)
                D_time.append(D_i)
            D_m = np.median(D_time)
            D_p_m = eval_data[hand]
            
            V_p = eval_data[hand +2]
            #calcolo V istantanea guardando la media dei valori precedenti e la velocità del frame precedente
            Vx = V_p * w + (D_m - D_p_m) * (1 - w)
            #trascrivo i valori di Dmedia e Vi sui dati di valutazione per la prox iterazione
            eval_data[hand] = D_m
            eval_data[hand+2] = Vx
           
            old_value.append(V_p)
            #print("CHNNNDNT:",  hand)

            if Vx <= -threshold:
                phase[hand] = 1
                if Vx <= threshold_count:
                    stage[hand] = "load"
            elif Vx < threshold and Vx > -threshold:
                #stage[hand] = "pause"
                phase[hand] = 0
            elif Vx >= threshold:
                phase[hand] = -1
                if Vx >= threshold_count:
                    if stage[hand] == "load":
                        #count[hand] +=1
                        to_count[hand] = True
                        
                        stage[hand]= "release"
                        #print("O-O-O-LD__!!! : ",-threshold, stage[hand], old_value)

        if stage[0]  == stage [1]:
            
            if to_count[0] == True or to_count[1] == True:
                
                count[0] +=1
                count[1] +=1
                print("phase : ", count)
                print("COUNTED!!! : " , stage , eval_data[2], eval_data[3])

            
            
        #print("||||__________|||||:",eval_data[2], eval_data[3], count, stage)
        

    else:
        print("need story data to tune evaluator")

    return eval_data, count, phase, stage
    
    



def kp_geometry_analisys(kp, count, stage, per, dictionary):
    #funzione per il conteggio esercizi
    #confronta le posizioni dei giunti con le soglie stabilite dai file di configurazione
    #differenzia tra confronto di angoli e di distanze
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
                #distanza

                distance = joint_distance_calculator(kps_to_render,kp)
                #print("distance is :", distance)
                #print("eva range", eva_range)

                for val in range(len(distance)):


                    # print("angle from EVA : {}".format(angle))
                    p = np.interp(distance[val], (eva_range[1], eva_range[0]), (100, 0))
                    per.append(p)
                    # Check for the dumbbell curls
                    # print("eva range 1 : {}".format(eva_range[1]))
                    # print(a)
                    # print("stage control: {}".format(stage))
                    #print("dist_eva: ", distance[val], (eva_range[1])/100)

                    if distance[val] > (eva_range[1])/100:

                        stage[val] = "down"

                    if distance[val] < (eva_range[0])/100 and stage[val] == "down":
                        stage[val] = "up"
                        count[val] += 1

                    #print("COUNTING routine :  {} ".format(count))

                return count, stage, per
                # distance





            elif len(kps_to_render[0]) == 6:
                #confronto angolare
                #print("KPS to render:", kps_to_render)

            # angle:

                for i in range(len(kps_to_render)):
                    # print("number of arms: {}".format(len(kps_to_render)))

                    segment = kps_to_render[i]
                    # print(kp[segment[5]])
                    a = findAngle((kp[segment[0]], kp[segment[1]]), (kp[segment[2]], kp[segment[3]]),
                                  (kp[segment[4]], kp[segment[5]]))
                    #write_data_csv([int(a)])
                    
                    angle.append(a)
                    # print("angle from EVA : {}".format(angle))
                    p = np.interp(a, (eva_range[0], eva_range[1]), (100, 0))
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

                    #print("COUNTING routine :  {} ".format(count))
                    # print("percentage : {} %".format(per))
                # print("angle : {}".format(angle))
                # print("stage : {}".format(stage))

                return count, stage , per
        else:
            print("no kps to render dictionary error, dictionary: {}".format(dictionary))


def evaluator(EX_global, q,string_from_tcp_ID):
    """
    funzione main per l evaluator, gestisce le sub funzioni e lavora come una macchina a stati


    :param EX_global: ID of the exercise saved in the multiprocessing shared memory
    :param q: the queue shared by the process to exchange the keypoits detected, produced by the skel and consumed by the evaluator
    :param string_from_tcp_ID: ID of the string recived from the ports communicator
    :return: none
    """
    #funzione main per l evaluator, gestisce le sub funzioni e lavora come una macchina a stati
    # printing process id
    print("ID of process running evaluator: {}".format(os.getpid()))

    #time.sleep(3)
    kp = []
    kp_story = []
    eval_data = [0,0,0,0]
    stage_v2 = [0,0]

    stage = ["", ""]
    per = [0,0]

    count = [0, 0]
    ex_string_from_TCP = ""  # comando arrivato dalla tcp
    ex_string = ""  # comando letto dalla memoria

    while True:

        #time.sleep(0.5)
        #verifico l arrivo di comandi da coordinator
        ex_string_from_TCP = TCP_listen_check_4_string(string_from_tcp_ID,ex_string_from_TCP)
        #print("TCP ex ID : ", string_from_tcp_ID.value)

        #time.sleep(0.05)
        if ex_string_from_TCP == "":

            ex_string_from_TCP = no_ex_cycle_control(string_from_tcp_ID,ex_string_from_TCP)
            #print("count = {}".format(count))

        elif ex_string_from_TCP == "stop":
            #print("stop command detected")
            # refreshing parameter of exercise
            ex_string_from_TCP = ""
            count = [0, 0]
            count_v2 =  [0, 0]
            stage = ["", ""]
            stage_v2 = [0, 0]
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
                #______ASSEGNO LA ex_string per l inizio della valutazione
                ex_string = check_for_string_in_memory(EX_global.value)
                #print("an exercise is under evaluation, starting...", ex_string)
                #print("count = {}".format(count))

            else:
                #print("start command error: no exercise selected, waiting for a selection before start")
                ex_string_from_TCP = ""
                #print("count = {}".format(count))
                ex_string_from_TCP = no_ex_cycle_control(string_from_tcp_ID,ex_string_from_TCP)  # da togliere

        else:
            #arrivato un nuovo esercizio non ancora scritto

            EX_global.value = write_ex_string_in_shared_memory(ex_string_from_TCP)
            #print("{} string wrote in memory".format(ex_string_from_TCP))
            #print("EX_global.value", EX_global.value)

        if EX_global.value != 0:
            if ex_string != "":
                # controllo di aver letto con successo la memoria dopo il comando di start

                #print("read from memory: {}".format(ex_string))
                config_param_dictionary = ex_string_to_config_param(ex_string)



                kp = wait_for_keypoints(q)
                kp_story.append(kp)
                while (len(kp_story) > STORY):
                    kp_story.pop(0)
                #print("len kp:",len(kp_story))

                eval_data,count_v2,stage_v2, stage = kp_geometry_analisys_v2(eval_data, kp_story, count_v2, config_param_dictionary,stage)


                #count, stage , per = kp_geometry_analisys(kp, count, stage,per, config_param_dictionary)
                #print("count. ",count)
                if stage_v2[0] == 0 or stage_v2[1] == 0:
                    stg = 0
                else:
                    stg = np.median(stage_v2)
                    
                

                ##packet = str(max(count)) + "," + str(int(max(per)))
                #packet = str(max(count_v2)) + "," + str(int(stg))
                #packet = [float(count_v2[0]),float(stg)]
                #packet = [0,0]
            
                print("packet", stg)
                #print("stg is : ", stage_v2)
                
                sender.send_status(21011, count_v2[0],stg,'localhost')
                #sender.send_status(21011, 5,0,'localhost')

 
