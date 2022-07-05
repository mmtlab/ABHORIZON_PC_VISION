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
import logging



data_collection=True
default_cooldown = 60
logging3 = logging.getLogger('EVA')
logging3.setLevel(logging.INFO)
fh3 = logging.FileHandler('./log/EVA.log')
fh3.setLevel(logging.DEBUG)
logging3.addHandler(fh3)
logging3.info(".............................................")

logging3.info("____!!!!!!!_____starting time____!!!!!!!_____: %s",datetime.now())
logging3.info(".............................................")

def load_all_exercise_in_RAM():
    """
    load_all_exercise_in_RAM

    :param nothing:

    :return: all_exercise: a dictionary of dictionary with all exercise
    """

    #open the ini file
    config = configparser.ConfigParser()
    config.read('exercise_info.ini')
    sections = config.sections()
    all_exercise = {}
    for exercise in sections:

        #start_time = time.time()


        try:
            segments_to_render = config.get(exercise, 'segments_to_render')
        except:
            logging3.warning("missing line of config, switch to default segments_to_render")
            segments_to_render = config.get("default", 'segments_to_render')
        segments_to_render = default_dictionary_control(segments_to_render, 'segments_to_render',config)
        segments_to_render = segments_to_render.split(',')

        try:
            joints_to_evaluate = config.get(exercise, 'joints_to_evaluate')
        except:
            logging3.warning("missing line of config, switch to default joints_to_evaluate")
            joints_to_evaluate = config.get("default", 'joints_to_evaluate')
        joints_to_evaluate = default_dictionary_control(joints_to_evaluate, 'joints_to_evaluate',config)
        joints_to_evaluate = joints_to_evaluate.split(',')

        try:
            evaluation_range = config.get(exercise, 'evaluation_range')
        except:
            logging3.warning("missing line of config, switch to default evaluation_range")
            evaluation_range = config.get("default", 'evaluation_range')
        evaluation_range = default_dictionary_control(evaluation_range, 'evaluation_range',config)
        evaluation_range = [int(x) for x in evaluation_range.split(",")]

        try:
            ID = config.get(exercise, 'ID')
        except:
            logging3.warning("missing line of config, switch to default ID")
            ID = config.get("default", 'ID')
        ID = int(default_dictionary_control(ID, 'ID',config))

        try:
            joints_with_ropes = config.get(exercise, 'joints_with_ropes')
        except:
            logging3.warning("missing line of config, switch to default joints_with_ropes")
            joints_with_ropes = config.get("default", 'joints_with_ropes')
        joints_with_ropes = default_dictionary_control(joints_with_ropes, 'joints_with_ropes',config)
        joints_with_ropes = joints_with_ropes.split(',')

        try:
            target_bar = config.get(exercise, 'target_bar')
        except:
            logging3.warning("missing line of config, switch to default target_bar")
            target_bar = config.get("default", 'target_bar')
        target_bar = default_dictionary_control(target_bar, 'target_bar',config)
        target_bar = target_bar.split(',')

        try:
            threshold = config.get(exercise, 'threshold')
        except:
            logging3.warning("missing line of config, switch to default threshold")
            threshold = config.get("default", 'threshold')
        threshold = int(default_dictionary_control(threshold, 'threshold',config))

        try:
            motor_history_events = config.get(exercise, 'motor_history_events')
        except:
            logging3.warning("missing line of config, switch to default motor_history_events")
            motor_history_events = config.get("default", 'motor_history_events')
        motor_history_events = int(default_dictionary_control(motor_history_events, 'motor_history_events',config))

        try:
            threshold_count = config.get(exercise, 'threshold_count')
        except:
            logging3.warning("missing line of config, switch to default threshold_count")
            threshold_count = config.get("default", 'threshold_count')
        threshold_count = int(default_dictionary_control(threshold_count, 'threshold_count',config))

        try:
            HISTERESYS = config.get(exercise, 'histeresys')
        except:
            logging3.warning("missing line of config, switch to default histeresys")
            HISTERESYS = config.get("default", 'histeresys')
        HISTERESYS = (int(default_dictionary_control(HISTERESYS, 'histeresys',config))) / 100
        # print("joints and target : ", joints_target)

        try:
            camera = config.get(exercise, 'camera')
        except:
            logging3.warning("missing line of config, switch to default camera")
            camera = config.get("default", 'camera')  # changes done
        camera = int(default_dictionary_control(camera, 'camera',config))


        try:
            motor = config.get(exercise, 'motor')

        except:
            logging3.warning("missing line of config, switch to default motor")
            motor = config.get("default", 'motor')  # changes done

        motor = int(default_dictionary_control(motor, 'motor',config))


        dictionary = {
            "segments_to_render": segments_to_render,
            "joints_to_evaluate": joints_to_evaluate,
            "evaluation_range": evaluation_range,
            "ID": ID,
            "joints_with_ropes": joints_with_ropes,
            "target_bar": target_bar,
            "threshold": threshold,
            "motor_history_events": motor_history_events,
            "threshold_count": threshold_count,
            "histeresys": HISTERESYS,
            "camera": camera,
            "motor": motor

        }


        # print("dictionary : {}".format(dictionary))
        for key, value in dictionary.items():
            # print(key, '\t', value)

            if isinstance(value, list):
                if all(isinstance(s, str) for s in value):
                    num_value = dictionary_string_2_machine_param(value)
                    dictionary[key] = num_value

        all_exercise[exercise] = dictionary


        #print("--- %s seconds ---" % (time.time() - start_time), exercise)




    return all_exercise

def writeCSVdata(data):
    """
    write data 2 CSV

    :param data: write to a csv file input data (append to the end)

    :return: nothing
    """
    # scrive su un file csv i dati estratti dalla rete Neurale
    file = open('./data/ex_data.csv', 'a')
    writer = csv.writer(file)
    now = datetime.now()
    time = now.strftime("%d/%m/%Y %H:%M:%S")

    writer.writerow(data)
    file.close()



def default_dictionary_control(parameter,descriptor, configurator):
    """
    check missing parameters and subsitute the 

    :param parameter: the quantity checked by the function, if empty, the default one is chosen
    :param descriptor: the dictionary key that identify the parameter under evaluation, it is used to access
    it from the dictionary

    :return: the parameter (if empty substituted with default)
    """
    if parameter is None or parameter == '':
        config = configparser.ConfigParser()
        #start_time = time.time()
        #config.read('exercise_info.ini')
        #print("--- %s seconds ---" % (time.time() - start_time), descriptor)
        parameter = configurator.get('default', descriptor)
        logging3.warning("missing parameter automaticcally selected the default one:%s", descriptor)

        return parameter
    return parameter


def write_data_csv(exercise,time,data):
    """
    write data 2 CSV,auto start and close when an exercise is done

    :param exercise: the name of the exercise for the title formatting
    :param time: timestamp of the start point of the excercise for the title of the excercise
    :param data: write to a csv file input data (the evaluation data of the exercise)

    :return: nothing
    """
    #print("filename_execution")
    filename = "./data/" + exercise + "_" + time.strftime("%m-%d-%Y_%H:%M:%S") +".csv"
    #print("filename:",filename)
    f = open(filename, 'a')
    writer = csv.writer(f)


    writer.writerow(data)
    f.close()




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
    if id is missing or incompletethe code will take the default one and pass directly
    to the dictionary, that will complete it with other defoul param without checkin
    the existence of other parameters

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

            try:
                ID = config.get(exercise, 'ID')
            except:
                logging3.warning("missing line of config, switch to default ID")
                ID = config.get("default", 'ID')
            ID = int(default_dictionary_control(ID, 'ID',config))
            
            return ID
        
    ex_string = "default" #se durante il ciclo for non c e match
    ID = ex_string_to_ID(ex_string) #se ex_string non riconosciuta prendo quela di default
    logging3.warning("not recognised exercise- switching to default...")
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
        #togli gli spazi
        kps = config_geometrical["ALIAS"][bodyframe.replace(" ", "")]

        kps = [int(x) for x in kps.split(",")]
        string2value.append(kps)


    #print(string2value)
    return string2value

def ex_string_to_config_param(ex_string, all_exe_dict):
    """
    convert the ex string to a dictionary with all the config instrunctions
    check also if there are missing paramers, and subsitute it with default value
    if ID is missing the default one are choosen and also other parameters so you will lose all
    of the other section (the incomplete but also the filled ones)
    :param ex_string: the exercise string identifing the exercise
    :return: the dictionary associated to the exercise
    """
    logging3.info("string to claim dictionary: %s", ex_string)

    dictionarey_from_ex_string = all_exe_dict[ex_string]
    #print("new dict; ", dictionarey_from_ex_string)
    return dictionarey_from_ex_string





"""

def ex_string_to_config_param(ex_string):
    logging3.info("string: %s", ex_string)

    # take all info of the exercise from the config files, write on a dictionary
    # read from ex_info
    config = configparser.ConfigParser()
    config.read('exercise_info.ini')
    sections = config.sections()

    # print("sections are : {}".format(sections))




    for exercise in sections:

        if exercise == ex_string:
            # config.get("test", "foo")
            
            try:
                segments_to_render = config.get(exercise, 'segments_to_render')
            except:
                logging3.warning("missing line of config, switch to default segments_to_render")
                segments_to_render = config.get("default", 'segments_to_render')    
            segments_to_render = default_dictionary_control(segments_to_render, 'segments_to_render',config)
            segments_to_render = segments_to_render.split(',')
            
            try:
                joints_to_evaluate = config.get(exercise, 'joints_to_evaluate')
            except:
                logging3.warning("missing line of config, switch to default joints_to_evaluate")
                joints_to_evaluate = config.get("default", 'joints_to_evaluate')
            joints_to_evaluate = default_dictionary_control(joints_to_evaluate, 'joints_to_evaluate',config)
            joints_to_evaluate = joints_to_evaluate.split(',')
            
            try:
                evaluation_range = config.get(exercise, 'evaluation_range')
            except:
                logging3.warning("missing line of config, switch to default evaluation_range")    
                evaluation_range = config.get("default", 'evaluation_range')
            evaluation_range = default_dictionary_control(evaluation_range, 'evaluation_range',config)
            evaluation_range = [int(x) for x in evaluation_range.split(",")]
            
            try:
                ID = config.get(exercise, 'ID')
            except:
                logging3.warning("missing line of config, switch to default ID")
                ID = config.get("default", 'ID')
            ID = int(default_dictionary_control(ID, 'ID',config))
            
            try:          
                joints_with_ropes = config.get(exercise, 'joints_with_ropes')
            except:
                logging3.warning("missing line of config, switch to default joints_with_ropes")
                joints_with_ropes = config.get("default", 'joints_with_ropes')
            joints_with_ropes = default_dictionary_control(joints_with_ropes, 'joints_with_ropes',config)
            joints_with_ropes = joints_with_ropes.split(',')
            
            try:
                target_bar = config.get(exercise, 'target_bar')
            except:
                logging3.warning("missing line of config, switch to default target_bar")
                target_bar = config.get("default", 'target_bar')
            target_bar = default_dictionary_control(target_bar, 'target_bar',config)
            target_bar = target_bar.split(',')
            
            try:           
                threshold = config.get(exercise, 'threshold')
            except:
                logging3.warning("missing line of config, switch to default threshold")
                threshold = config.get("default", 'threshold')
            threshold = int(default_dictionary_control(threshold, 'threshold',config))

            try:
                motor_history_events = config.get(exercise, 'motor_history_events')
            except:
                logging3.warning("missing line of config, switch to default motor_history_events")
                motor_history_events = config.get("default", 'motor_history_events')
            motor_history_events = int(default_dictionary_control(motor_history_events, 'motor_history_events',config))
            
            try:
                threshold_count = config.get(exercise, 'threshold_count')
            except:
                logging3.warning("missing line of config, switch to default threshold_count")
                threshold_count = config.get("default", 'threshold_count')
            threshold_count = int(default_dictionary_control(threshold_count, 'threshold_count',config))
       
            try:
                HISTERESYS = config.get(exercise, 'histeresys')
            except:
                logging3.warning("missing line of config, switch to default histeresys")
                HISTERESYS = config.get("default", 'histeresys')
            HISTERESYS = (int(default_dictionary_control(HISTERESYS, 'histeresys',config)))/100
            # print("joints and target : ", joints_target)

            try:
                camera = config.get(exercise, 'camera')
            except:
                logging3.warning("missing line of config, switch to default camera")
                camera = config.get("default", 'camera') #changes done
            camera = int(default_dictionary_control(camera, 'camera',config))

            try:
                motor = config.get(exercise, 'motor')
            except:
                logging3.warning("missing line of config, switch to default motor")
                motor = config.get("default", 'motor') #changes done
            motor = int(default_dictionary_control(motor, 'motor',config))

            dictionary = {
                "segments_to_render": segments_to_render,
                "joints_to_evaluate": joints_to_evaluate,
                "evaluation_range": evaluation_range,
                "ID": ID,
                "joints_with_ropes": joints_with_ropes,
                "target_bar": target_bar,
                "threshold": threshold,
                "motor_history_events": motor_history_events,
                "threshold_count": threshold_count,
                "histeresys": HISTERESYS,
                "camera": camera,
                "motor": motor

            }
            # print("dictionary : {}".format(dictionary))
            for key, value in dictionary.items():
                #print(key, '\t', value)

                if isinstance(value, list):
                    if all(isinstance(s, str) for s in value):
                        num_value = dictionary_string_2_machine_param(value)
                        dictionary[key] = num_value

            return dictionary
    logging3.warning("no exercise found, switch to default, string recived:", ex_string)
    dictionary = ex_string_to_config_param("default")
    return dictionary





"""
def wait_for_keypoints(queuekp):
    """
    function in loop stucked until new keypoints are aviable in the multiprocessing queue

    :param queuekp: a multiprocessing queue filled with an array of keypoint and consumed by this function
    :return: consume and return the keypoints from the queue making aviable to this process (evaluator)
    """
    # loop 4 waiting the queue of KP from mediapipe
    keypoints = []
    presence = False

    while not keypoints:

        try:
            keypoints = queuekp.get(False)
        except queue.Empty:
            # print("no KP data aviable: queue empty")
            presence = False
            #trigger of missing image and kp
            #print("q empty")
            pass

        else:

            if not keypoints:
                presence = False
                #print("not kp")
                #logging3.error("no valid kp")
            else:
                presence = True
                return keypoints, presence


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
    elif string_from_tcp_ID.value == 1000:
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


def kp_geometry_analisys_v2(eval_data, kp_history, dictionary, stage,retro_filter_state):
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
    :param retro_filter_state: the history state for controlling the jitter
    in the retroacting variable phase. present also in the return, work as a phase memory
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
    HISTERESYS = dictionary["histeresys"]
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
            D_m = np.mean(D_time) #distanza media da target adesso
            D_p_m = eval_data[hand] #vecchia distanza media

            V_p = eval_data[hand + 2] #vecchia velocità
            # calcolo V istantanea guardando la media dei valori precedenti e la velocità del frame precedente
            Vx = V_p * w + (D_m - D_p_m) * (1 - w)
            # trascrivo i valori di Dmedia e Vi sui dati di valutazione per la prox iterazione
            eval_data[hand] = D_m
            eval_data[hand + 2] = Vx

            old_value.append(V_p)
            # print("CHNNNDNT:",  hand)
            if retro_filter_state[hand] == "retro":
                histeresys = HISTERESYS*threshold
            else:
                histeresys = 0
            '''    
            if Vx >= threshold +histeresys :
                if retro_filter_state[hand] != "retro":
                    phase[hand] = 1
                    retro_filter_state[hand] = "traction"
                else:
                    phase[hand] = 0
                    retro_filter_state[hand] = "neutral"
                    logging3.debug("new traction smoothed in neutral")
                if Vx >= threshold_count:
                    stage[hand] = "load"
            elif Vx < threshold +histeresys and Vx > -threshold +histeresys:
                # stage[hand]  = "pause"
                phase[hand] = 0
                retro_filter_state[hand] = "neutral"
            elif Vx <= -threshold +histeresys:
                if retro_filter_state[hand] != "traction":
                    phase[hand] = -1
                    retro_filter_state[hand] = "retro"
                else:
                    phase[hand] = 0
                    logging3.debug("new retro smoothed in neutral")
                    retro_filter_state[hand] = "neutral"
                if Vx <= -threshold_count:
                    if stage[hand] == "load":
                        # count[hand] +=1
                        to_count[hand] = True
                        stage[hand] = "release"
            '''
            #retroaction

            if Vx >= threshold + histeresys:
                if retro_filter_state[hand] != "retro":
                    phase[hand] = 1
                    retro_filter_state[hand] = "traction"
                else:
                    phase[hand] = 0
                    retro_filter_state[hand] = "neutral"
                    logging3.debug("new traction smoothed in neutral")
            elif Vx < threshold + histeresys and Vx > -threshold + histeresys:
                    # stage[hand]  = "pause"
                    phase[hand] = 0
                    retro_filter_state[hand] = "neutral"
            elif Vx <= -threshold + histeresys:
                if retro_filter_state[hand] != "traction":
                    phase[hand] = -1
                    retro_filter_state[hand] = "retro"
                else:
                    phase[hand] = 0
                    logging3.debug("new retro smoothed in neutral")
                    retro_filter_state[hand] = "neutral"

            #counting

            if Vx >= threshold_count:
                stage[hand] = "load"

            elif Vx <= -threshold_count:
                if stage[hand] == "load":
                    # count[hand] +=1
                    print("to release")
                    stage[hand] = "release"
            elif Vx <= threshold_count/2 and Vx >= -threshold_count/2:
                if stage[hand] == "release":
                    print("to count...")
                    # count[hand] +=1
                    to_count[hand] = True
                    stage[hand] = "terminated_count"







                        # print("O-O-O-LD__!!! : ",-threshold, stage[hand], old_value)
        #print("eval", eval_data[0])
        #print('stages:', stage, eval_data)
        if stage[0] == stage[1]:

            if to_count[0] == True or to_count[1] == True:
                count[0] = 1
                count[1] = 1

               
                print("COUNTED!!! : ", stage, eval_data[2], eval_data[3])

        # print("||||__________|||||:",eval_data[2], eval_data[3], count, stage)


    else:
        logging3.warning("need story data to tune evaluator")

    return eval_data, count, phase, stage, retro_filter_state

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
            A_m = np.mean(A_time) #distanza media da target adesso
            A_p_m = eval_data[hand] #vecchia angolo media

            V_p = eval_data[hand + 2] #vecchia velocità
            # calcolo V istantanea guardando la media dei valori precedenti e la velocità del frame precedente
            Vx = V_p * w + (A_m - A_p_m) * (1 - w)
            
            # trascrivo i valori di Dmedia e Vi sui dati di valutazione per la prox iterazione
            eval_data[hand] = A_m
            eval_data[hand + 2] = Vx

            old_value.append(V_p)
            # print("CHNNNDNT:",  hand)
            ''' retroazione + cnteggio angolare 
            if threshold > 0:
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
            else:
                if Vx <= threshold:
                    phase[hand] = 1
                    if Vx <= threshold_count:
                        stage[hand] = "load"
                        
                elif Vx > threshold and Vx < -threshold:
                    # stage[hand] = "pause"
                    phase[hand] = 0
                elif Vx >= -threshold:
                    phase[hand] = -1
                    if Vx >= -threshold_count:
                        if stage[hand] == "load":
                            # count[hand] +=1
                            to_count[hand] = True
                            stage[hand] = "release"
                            
            '''
            #solo conteggio
            #threshold positiva
            if threshold_count > 0:
                if Vx >= threshold_count:
                    stage[hand] = "load"

                elif Vx < threshold_count/2 and Vx > -threshold_count/2:
                    if stage[hand] == "release":
                        to_count[hand] = True

                        stage[hand] = "terminated_count"
                elif Vx <= -threshold_count:

                    if stage[hand] == "load":
                        stage[hand] = "release"
            #threshold negativa
            else:
                if Vx <= threshold_count:
                    stage[hand] = "load"

                elif Vx > threshold_count/2 and Vx < -threshold_count/2:
                    if stage[hand] == "release":
                        to_count[hand] = True

                        stage[hand] = "terminated_count"
                elif Vx >= -threshold_count:

                    if stage[hand] == "load":
                        stage[hand] = "release"

        
                
                        # print("O-O-O-LD__!!! : ",-threshold, stage[hand], old_value)
        #print("ang_vel: ", Vx)
        #print("vel: ",eval_data[3])
        if stage[0] == stage[1]:

            if to_count[0] == True or to_count[1] == True:
                count[0] = 1
                count[1] = 1
                
                #print("ANGLE____COUNTED!!! : ", stage, eval_data[2], eval_data[3])

        # print("||||__________|||||:",eval_data[2], eval_data[3], count, stage)


    else:
        logging3.warning("need story data to tune evaluator")

    return eval_data, count, phase, stage

def specific_joints_evaluator(kp,story, dictionary):
    # funzione per il conteggio esercizi
    # confronta le posizioni dei giunti con le soglie stabilite dai file di configurazione
    # differenzia tra confronto di angoli e di distanze
    if kp == []:
        logging3.error("no KP aviable")
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
        #if count_angle[0] == 1:
            compared_count += 1
            cooldown_frame = default_cooldown
            if count_target[0] == 1:
                logging3.info("TARGET trigger count")
            else:
                logging3.info("ANGLE trigger count")
            
            logging3.info("counted: %s", compared_count)
    else:
        if cooldown_frame != 0:
            cooldown_frame -= 1
               
    return compared_count, cooldown_frame


def evaluator(EX_global, q, string_from_tcp_ID,user_id):
    """
    funzione main per l evaluator, gestisce le sub funzioni e lavora come una macchina a stati


    :param EX_global: ID of the exercise saved in the multiprocessing shared memory
    :param q: the queue shared by the process to exchange the keypoits detected, produced by the skel and consumed by the evaluator
    :param string_from_tcp_ID: ID of the string recived from the ports communicator
    :return: none
    """
    # funzione main per l evaluator, gestisce le sub funzioni e lavora come una macchina a stati
    # printing process id
    logging3.info("ID of process running evaluator: {}".format(os.getpid()))


    #carico in ram tutti gli esercizi
    all_exercise = load_all_exercise_in_RAM()
    # the user for data saving
    user = user_id.value

    # time.sleep(3)
    kp = []
    kp_story = []
    story_specific = []
    eval_data = [0, 0, 0, 0]
    stage_v2 = [0, 0]
    
    #param for save data in csv
    time_csv = ""
    exercise_csv = ""
    inizialize_csv_file = False
    
    #variable for story of stg
    retro_filter_state = ["neutral","neutral"]
    
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
            
            #stop reset the inizialization state of the csv data file  
            time_csv = ""
            exercise_csv = ""
            inizialize_csv_file = False
            
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
                if data_collection == True:
                    if inizialize_csv_file == False:
                        time_csv = datetime.now()
                        exercise_csv = "user_"+ str(user) + "_" + ex_string
                        header = ["count","D2_r","D2_l","V_r","V_l","A_r","A_l","VA_r","VA_l", "retro_param"]
                        write_data_csv(exercise_csv,time_csv,header)
                        inizialize_csv_file = True
                        logging3.debug("exercise string and time saved for csv writing")
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
                    config_param_dictionary = ex_string_to_config_param(ex_string,all_exercise)

                kp, presence = wait_for_keypoints(q)
                #print("people is presence?:",presence)
                kp_story.append(kp)
                STORY = config_param_dictionary["motor_history_events"]
                while (len(kp_story) > STORY):
                    kp_story.pop(0)
                
                while (len(story_specific) > STORY):
                    story_specific.pop(0)
                    
                    
                # print("len kp:",len(kp_story))

                eval_data, count_v2, stage_v2, stage, retro_filter_state = kp_geometry_analisys_v2(eval_data, kp_story,
                                                                               config_param_dictionary, stage,retro_filter_state)
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
                    
            
                if data_collection == True:
                    #print(exercise_csv,time_csv)
                    if exercise_csv != "" and time_csv != "":
                    
                        data = [compared_count,eval_data[0],eval_data[1],eval_data[2],eval_data[3], eval_data_angle[0],eval_data_angle[1],eval_data_angle[2],eval_data_angle[3], stg]
                        write_data_csv(exercise_csv,time_csv,data)
                ##packet = str(max(count)) + "," + str(int(max(per)))
                # packet = str(max(count_v2)) + "," + str(int(stg))
                # packet = [float(count_v2[0]),float(stg)]
                # packet = [0,0]

                logging3.debug("act: %s", stg)
                # print("stg is : ", stage_v2)

                sender.send_status(21011, compared_count, stg, 'localhost')
                # sender.send_status(21011, 5,0,'localhost')


