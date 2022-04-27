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


default_cooldown = 50

def default_dictionary_control(parameter,descriptor):
    """
    check missing parameters and subsitute the 

    :param parameter: the quantity checked by the function, if empty, the default one is chosen
    :param descriptor: the dictionary key that identify the parameter under evaluation, it is used to access
    it from the dictionary

    :return: the parameter (if empty substituted with default)
    """
    if parameter is None or parameter == '':
        config = configparser.ConfigParser()
        config.read('exercise_info.ini')
        parameter = config.get('default', descriptor)
        print("missing parameter :", descriptor)
        print("automaticcally selected the default one")
        return parameter
    return parameter


def write_data_csv(data):
    """
    write data 2 CSV

    :param data: write to a csv file input data (append to the end)

    :return: nothing
    """

    f = open('data.csv', 'a')
    writer = csv.writer(f)

    writer.writerow(data)


def joint_distance_calculator(kps_to_render, kp):
    """
    calculate distance btw 2 2d point (from all KP select only the segment 2 render)


    :param kps_to_render: ID of the joints under distance evaluation
    :param kp: actual frame joints detected with their coordinates
    :return: distance between the couple of joints
    """
    distance = []
    normalizer = math.sqrt(pow((kp[2] - kp[4]), 2) + pow((kp[3] - kp[5]), 2))  # spalle

    for i in range(len(kps_to_render)):
        # print("number of arms: {}".format(len(kps_to_render)))

        segment = kps_to_render[i]
        #  x1, y1, x2, y2
        xa = kp[segment[0]]
        ya = kp[segment[1]]
        xb = kp[segment[2]]
        yb = kp[segment[3]]
        # ((kp[segment[0]], kp[segment[1]]), (kp[segment[2]], kp[segment[3]])
        distance.append((math.sqrt(pow((xa - xb), 2) + pow((ya - yb), 2))) / normalizer)
    return distance


def no_ex_cycle_control(string_from_tcp_ID, ex_string):
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
    ex_string = ex_string.rstrip("\n")
    for exercise in sections:

        if exercise == ex_string:
            # config.get("test", "foo")

            ID = config.get(exercise, 'ID')
            ID = int(default_dictionary_control(ID,"ID")) #controllo che non sia vuoto
            #se ID vuoto associo quello di default
            return ID
        
    ex_string = "default" #se durante il ciclo for non c e match
    ID = ex_string_to_ID(ex_string) #se ex_string non riconosciuta prendo quela di default
    print("not recognised exercise- switching to default...")
    return ID


def ID_to_ex_string(ID):
    """
    convert an ID from multiproc value slot to the ex_string corresponding

    :param ID: an int, the ID of the current exercise
    :return: dthe ex string associated to the ID, if not fount return empty ex_string
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

    # print("no exercise found")
    ex_string = "0"
    return ex_string

'''
def KP_to_render_from_config_file(dictionary):
    """
    function description

    :param param1: description of param1
    :param param2: description of param2
    :return: description of the function output
    """
    # extract the Kps of the exercise fron the ini file
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

'''

def dictionary_string_2_machine_param(value):
    """
    this function translate human understandable parameter for the ex.
    evaluation to the machine lanmguage for the cinematic analisys

    :param value: the human interpreted parameter
    :return: the machine understandable parameters
    """
    # extract the Kps of the exercise fron the ini file
    config_geometrical = configparser.ConfigParser()

    config_geometrical.read('config.ini')
    string2value = []


    for bodyframe in value:
        kps = config_geometrical["ALIAS"][bodyframe.replace(" ", "")]

        kps = [int(x) for x in kps.split(",")]
        string2value.append(kps)


    #print(string2value)
    return string2value



def ex_string_to_config_param(ex_string):
    print("string", ex_string)
    """
    convert the ex string to a dictionary with all the config instrunctions
    check also if there are missing paramers, and subsitute it with default value

    :param ex_string: the exercise string identifing the exercise
    :return: the dictionary associated to the exercise
    """
    # take all info of the exercise from the config files, write on a dictionary
    # read from ex_info
    config = configparser.ConfigParser()
    config.read('exercise_info.ini')
    sections = config.sections()

    # print("sections are : {}".format(sections))



    for exercise in sections:

        if exercise == ex_string:
            # config.get("test", "foo")

            segments_to_render = config.get(exercise, 'segments_to_render')
            segments_to_render = default_dictionary_control(segments_to_render, 'segments_to_render')
            segments_to_render = segments_to_render.split(',')

            joints_to_evaluate = config.get(exercise, 'joints_to_evaluate')
            joints_to_evaluate = default_dictionary_control(joints_to_evaluate, 'joints_to_evaluate')
            joints_to_evaluate = joints_to_evaluate.split(',')

            evaluation_range = config.get(exercise, 'evaluation_range')
            evaluation_range = default_dictionary_control(evaluation_range, 'evaluation_range')
            evaluation_range = [int(x) for x in evaluation_range.split(",")]

            ID = config.get(exercise, 'ID')
            ID = int(default_dictionary_control(ID, 'ID'))

            joints_with_ropes = config.get(exercise, 'joints_with_ropes')
            joints_with_ropes = default_dictionary_control(joints_with_ropes, 'joints_with_ropes')
            joints_with_ropes = joints_with_ropes.split(',')

            target_bar = config.get(exercise, 'target_bar')
            target_bar = default_dictionary_control(target_bar, 'target_bar')
            target_bar = target_bar.split(',')

            threshold = config.get(exercise, 'threshold')
            threshold = int(default_dictionary_control(threshold, 'threshold'))

            motor_history_events = config.get(exercise, 'motor_history_events')
            motor_history_events = int(default_dictionary_control(motor_history_events, 'motor_history_events'))

            threshold_count = config.get(exercise, 'threshold_count')
            threshold_count = int(default_dictionary_control(threshold_count, 'threshold_count'))

            # print("joints and target : ", joints_target)

            dictionary = {
                "segments_to_render": segments_to_render,
                "joints_to_evaluate": joints_to_evaluate,
                "evaluation_range": evaluation_range,
                "ID": ID,
                "joints_with_ropes": joints_with_ropes,
                "target_bar": target_bar,
                "threshold": threshold,
                "motor_history_events": motor_history_events,
                "threshold_count": threshold_count

            }
            # print("dictionary : {}".format(dictionary))
            for key, value in dictionary.items():
                #print(key, '\t', value)

                if isinstance(value, list):
                    if all(isinstance(s, str) for s in value):
                        num_value = dictionary_string_2_machine_param(value)
                        dictionary[key] = num_value

            return dictionary
    print("no exercise found, switch to default, string recived:", ex_string)
    dictionary = ex_string_to_config_param("default")
    return dictionary






def wait_for_keypoints(queuekp):
    """
    function in loop stucked until new keypoints are aviable in the multiprocessing queue

    :param queuekp: a multiprocessing queue filled with an array of keypoint and consumed by this function
    :return: consume and return the keypoints from the queue making aviable to this process (evaluator)
    """
    # loop 4 waiting the queue of KP from mediapipe
    keypoints = []

    while not keypoints:

        try:
            keypoints = queuekp.get(False)
        except queue.Empty:
            # print("no KP data aviable: queue empty")
            pass

        else:

            if not keypoints:
                print("no valid kp")
            else:
                return keypoints


def check_for_string_in_memory(multiprocessing_value_slot):
    """
    check the presence in the memory of an exercise ID and convert it to a tring  
    :param multiprocessing_value_slot: the memory slot where ID are saved
    :return: the string associated to the ID
    """
    # return the exercise written in the shared memory (if prresent)
    ex_ID = multiprocessing_value_slot
    ex_string_selected = ID_to_ex_string(ex_ID)
    #print("triggering this every iteration?")

    return ex_string_selected


def TCP_listen_check_4_string(string_from_tcp_ID, ex_string_recived):
    """
    function for listen the subprocess that communicate with the interface and coordiunator. 

    :param param1: description of param1
    :param param2: description of param2
    :return: description of the function output
    """
    # convert the shared memory id to a string identifing the exercise
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


def kp_geometry_analisys_v2(eval_data, kp_history, dictionary, stage):
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
    phase = [0, 0]
    to_count = [False, False]
    count =[0,0]

    w = 0.5
    threshold = dictionary["threshold"]
    threshold_count = dictionary["threshold_count"]
    STORY = dictionary["motor_history_events"]
    joints_with_ropes = dictionary["joints_with_ropes"]
    target_bar = dictionary["target_bar"]
    printV = False

    if len(kp_history) > STORY - 1:

        side = int(len(target_bar))  # 0-1, 8 valori (4 punti)
        old_value = []

        for hand in range(side): #0-1
            IDx_m = joints_with_ropes[hand][0]
            IDy_m = joints_with_ropes[hand][1] #id del giunto mano x e y

            xa = kp_history[-1][IDx_m] #ultimo valore di posizione della mano
            xt = (target_bar[hand][0])*480/100 #x del target
            ya = kp_history[-1][IDy_m]
            yt = (target_bar[hand][1])*640/100
            # D_sx = math.sqrt(pow((xa - xt), 2) + pow((ya - yt), 2))
            D_time = []

            for i in range(len(kp_history)):
                XM_i = kp_history[i][IDx_m]
                YM_i = kp_history[i][IDy_m]
                D_i = pow((XM_i - xt), 2) + pow((YM_i - yt), 2)
                D_time.append(D_i)
            D_m = np.median(D_time) #distanza media da target adesso
            D_p_m = eval_data[hand] #vecchia distanza media

            V_p = eval_data[hand + 2] #vecchia velocità
            # calcolo V istantanea guardando la media dei valori precedenti e la velocità del frame precedente
            Vx = V_p * w + (D_m - D_p_m) * (1 - w)
            # trascrivo i valori di Dmedia e Vi sui dati di valutazione per la prox iterazione
            eval_data[hand] = D_m
            eval_data[hand + 2] = Vx

            old_value.append(V_p)
            # print("CHNNNDNT:",  hand)

            if Vx >= threshold:
                phase[hand] = 1
                if Vx >= threshold_count:
                    stage[hand] = "load"
            elif Vx < threshold and Vx > -threshold:
                # stage[hand] = "pause"
                phase[hand] = 0
            elif Vx <= -threshold:
                phase[hand] = -1
                if Vx <= -threshold_count:
                    if stage[hand] == "load":
                        # count[hand] +=1
                        to_count[hand] = True

                        stage[hand] = "release"
                        # print("O-O-O-LD__!!! : ",-threshold, stage[hand], old_value)

        if stage[0] == stage[1]:

            if to_count[0] == True or to_count[1] == True:
                count[0] = 1
                count[1] = 1
               
                #print("COUNTED!!! : ", stage, eval_data[2], eval_data[3])

        # print("||||__________|||||:",eval_data[2], eval_data[3], count, stage)


    else:
        print("need story data to tune evaluator")

    return eval_data, count, phase, stage

def velocity_tracker_angle(eval_data, kp_history, dictionary, stage):
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
    phase = [0, 0]
    to_count = [False, False]
    count = [0,0]
    
    w = 0.5
    
    STORY = dictionary["motor_history_events"]
    joints_to_evaluate = dictionary["joints_to_evaluate"]
    evaluation_range = dictionary["evaluation_range"]
    threshold = evaluation_range[0]
    threshold_count = evaluation_range[1]
    

    if len(kp_history) > STORY - 1:

        side = int(len(joints_to_evaluate))  #two array one for every arm
        old_value = []

        for hand in range(side): #0-1
            #6 coordinate 3 punti 1 angolo
            IDx1 = joints_to_evaluate[hand][0]
            IDy1 = joints_to_evaluate[hand][1] #id del giunto mano x e y
            IDx2 = joints_to_evaluate[hand][2]
            IDy2 = joints_to_evaluate[hand][3]
            IDx3 = joints_to_evaluate[hand][4]
            IDy3 = joints_to_evaluate[hand][5]

            
            A_time = []

            for i in range(len(kp_history)):
                X1 = kp_history[i][IDx1]
                Y1 = kp_history[i][IDy1]
                X2 = kp_history[i][IDx2]
                Y2 = kp_history[i][IDy2]
                X3 = kp_history[i][IDx3]
                Y3 = kp_history[i][IDy3]
                A_i = findAngle((X1,Y1),(X2,Y2),(X3,Y3))
                A_time.append(A_i)
            A_m = np.median(A_time) #distanza media da target adesso
            A_p_m = eval_data[hand] #vecchia angolo media

            V_p = eval_data[hand + 2] #vecchia velocità
            # calcolo V istantanea guardando la media dei valori precedenti e la velocità del frame precedente
            Vx = V_p * w + (A_m - A_p_m) * (1 - w)
            # trascrivo i valori di Dmedia e Vi sui dati di valutazione per la prox iterazione
            eval_data[hand] = A_m
            eval_data[hand + 2] = Vx

            old_value.append(V_p)
            # print("CHNNNDNT:",  hand)

            if Vx >= threshold:
                phase[hand] = 1
                if Vx >= threshold_count:
                    stage[hand] = "load"
            elif Vx < threshold and Vx > -threshold:
                # stage[hand] = "pause"
                phase[hand] = 0
            elif Vx <= -threshold:
                phase[hand] = -1
                if Vx <= -threshold_count:
                    if stage[hand] == "load":
                        # count[hand] +=1
                        to_count[hand] = True

                        stage[hand] = "release"
                        # print("O-O-O-LD__!!! : ",-threshold, stage[hand], old_value)
        #print("ang_vel: ", Vx)
        if stage[0] == stage[1]:

            if to_count[0] == True or to_count[1] == True:
                count[0] = 1
                count[1] = 1
                
                #print("ANGLE____COUNTED!!! : ", stage, eval_data[2], eval_data[3])

        # print("||||__________|||||:",eval_data[2], eval_data[3], count, stage)


    else:
        print("need story data to tune evaluator")

    return eval_data, count, phase, stage

def specific_joints_evaluator(kp,story, dictionary):
    # funzione per il conteggio esercizi
    # confronta le posizioni dei giunti con le soglie stabilite dai file di configurazione
    # differenzia tra confronto di angoli e di distanze
    if kp == []:
        print("no KP aviable")
    else:

        evaluation_range = dictionary["evaluation_range"]
        joints_to_evaluate = dictionary["joints_to_evaluate"]
        angle = []

        per = []    


        for arto in joints_to_evaluate:
            # print("number of arms: {}".format(len(kps_to_render)))

            
            # print(kp[segment[5]])
            a = findAngle((kp[arto[0]], kp[arto[1]]), (kp[arto[2]], kp[arto[3]]),
                          (kp[arto[4]], kp[arto[5]]))
            # write_data_csv([int(a)])

            angle.append(a)
            # print("angle from EVA : {}".format(angle))
            p = np.interp(a, (evaluation_range[0], evaluation_range[1]), (100, 0))
            per.append(p)
            
        story.append(angle)
        #print("angle:", angle)

        return story, per , angle
    
    
def compared_counting(compared_count,count_target, count_angle,cooldown_frame):
    #print("calldown",cooldown_frame)
    if cooldown_frame < 1:
        if count_target[0] == 1 or count_angle[0] == 1:
            compared_count += 1
            cooldown_frame = default_cooldown
            
            print("counted: ", compared_count)
    else:
        if cooldown_frame != 0:
            cooldown_frame -= 1
               
    return compared_count, cooldown_frame


def evaluator(EX_global, q, string_from_tcp_ID):
    """
    funzione main per l evaluator, gestisce le sub funzioni e lavora come una macchina a stati


    :param EX_global: ID of the exercise saved in the multiprocessing shared memory
    :param q: the queue shared by the process to exchange the keypoits detected, produced by the skel and consumed by the evaluator
    :param string_from_tcp_ID: ID of the string recived from the ports communicator
    :return: none
    """
    # funzione main per l evaluator, gestisce le sub funzioni e lavora come una macchina a stati
    # printing process id
    print("ID of process running evaluator: {}".format(os.getpid()))

    # time.sleep(3)
    kp = []
    kp_story = []
    story_specific = []
    eval_data = [0, 0, 0, 0]
    stage_v2 = [0, 0]
    
    
    #frame to wait before next counting
    cooldown_frame =  default_cooldown
    
    #history of angle and angular velocity
    eval_data_angle = [0, 0, 0, 0]
    
    #stage phase for angle dynamics
    stage_v2_angle = [0, 0]
    
    config_param_dictionary ={}

    stage = ["", ""]
    
    #stage for angle velocity traker
    #(histeresis memorized phase for triggering counting)
    stage_angle = ["", ""]
    
    per = [0, 0]


    compared_count = 0
    #count for angle velocity traker

    
    
    ex_string_from_TCP = ""  # comando arrivato dalla tcp
    ex_string = ""  # comando letto dalla memoria

    while True:

        # time.sleep(0.5)
        # verifico l arrivo di comandi da coordinator
        ex_string_from_TCP = TCP_listen_check_4_string(string_from_tcp_ID, ex_string_from_TCP)
        # print("TCP ex ID : ", string_from_tcp_ID.value)

        # time.sleep(0.05)
        if ex_string_from_TCP == "":

            ex_string_from_TCP = no_ex_cycle_control(string_from_tcp_ID, ex_string_from_TCP)
            # print("count = {}".format(count))

        elif ex_string_from_TCP == "stop":
            # print("stop command detected")
            # refreshing parameter of exercise
            ex_string_from_TCP = ""
            count = [0, 0]
            count_angle = [0, 0]
            count_v2 = [0, 0]
            count_v2_angle = [0, 0]
            stage_angle = ["", ""]
            stage = ["", ""]
            
            #conteggio totale
            compared_count = 0

            
            stage_v2 = [0, 0]
            stage_v2_angle = [0, 0]
            EX_global.value = 0
            ex_string = ""
            config_param_dictionary = {}
            # print("count = {}".format(count))
            ex_string_from_TCP = no_ex_cycle_control(string_from_tcp_ID, ex_string_from_TCP)

        elif ex_string_from_TCP == "pause":
            # print("pause command detected")
            # rimane comunque l esercizio in memoria
            ex_string_from_TCP = ""
            # print("count(pause) = {}".format(count))
            ex_string_from_TCP = no_ex_cycle_control(string_from_tcp_ID, ex_string_from_TCP)

        elif ex_string_from_TCP == "start":
            if EX_global.value != 0:
                # ______ASSEGNO LA ex_string per l inizio della valutazione
                ex_string = check_for_string_in_memory(EX_global.value)
                # print("an exercise is under evaluation, starting...", ex_string)
                # print("count = {}".format(count))

            else:
                # print("start command error: no exercise selected, waiting for a selection before start")
                ex_string_from_TCP = ""
                # print("count = {}".format(count))
                ex_string_from_TCP = no_ex_cycle_control(string_from_tcp_ID, ex_string_from_TCP)  # da togliere

        else:
            # arrivato un nuovo esercizio non ancora scritto

            EX_global.value = write_ex_string_in_shared_memory(ex_string_from_TCP)
            # print("{} string wrote in memory".format(ex_string_from_TCP))
            # print("EX_global.value", EX_global.value)

        if EX_global.value != 0:
            if ex_string != "":
                # controllo di aver letto con successo la memoria dopo il comando di start

                # print("read from memory: {}".format(ex_string))
                if bool(config_param_dictionary):
                    pass
                else:
                    config_param_dictionary = ex_string_to_config_param(ex_string)

                kp = wait_for_keypoints(q)
                kp_story.append(kp)
                STORY = config_param_dictionary["motor_history_events"]
                while (len(kp_story) > STORY):
                    kp_story.pop(0)
                
                while (len(story_specific) > STORY):
                    story_specific.pop(0)
                    
                    
                # print("len kp:",len(kp_story))

                eval_data, count_v2, stage_v2, stage = kp_geometry_analisys_v2(eval_data, kp_story,
                                                                               config_param_dictionary, stage)
                #story_specific, per , angle = specific_joints_evaluator(kp,story_specific, config_param_dictionary)
                
                eval_data_angle, count_v2_angle, stage_v2_angle, stage_angle = velocity_tracker_angle(eval_data_angle, kp_story,
                                                                                                     config_param_dictionary, stage_angle)
                
                
                #print("angle, trigger",count_v2_angle,count_v2) 
                compared_count, cooldown_frame  = compared_counting(compared_count,count_v2, count_v2_angle,cooldown_frame)
                # count, stage , per = kp_geometry_analisys(kp, count, stage,per, config_param_dictionary)
                # print("count. ",count)
                if len(story_specific) >5:
                    med_gradient = np.median(np.gradient(story_specific))
                    #print("gradeint:",med_gradient)
                state_value = stage_v2[0] + stage_v2[1] 
                if state_value > 0:
                    stg = 1
                if state_value == 0:
                
                    if stage_v2[0] == stage_v2[1]:
                        stg = 0
                    else:
                        stg = 1
                if state_value < 0:
                    
                    #evaluation_range = config_param_dictionary["evaluation_range"]
                    #if np.median(angle) < evaluation_range[0]*2:
                    stg = -1
                    #else:
                    #   stg = 0
                    
               

                ##packet = str(max(count)) + "," + str(int(max(per)))
                # packet = str(max(count_v2)) + "," + str(int(stg))
                # packet = [float(count_v2[0]),float(stg)]
                # packet = [0,0]

                #print("act: ", stg)
                # print("stg is : ", stage_v2)

                sender.send_status(21011, compared_count, stg, 'localhost')
                # sender.send_status(21011, 5,0,'localhost')


