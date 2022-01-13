# importing the multiprocessing module
import multiprocessing
import os
import time
import receiver


import sender
import configparser
import EVA
import SKEL
import ctypes








def init():
    config = configparser.RawConfigParser()
    config.read('/home/bernardo/PycharmProjects/AB_POSE_MACHINE/config.ini')

    #PORTS_INFO = dict(config.items('PORTS'))
    #print("Configuration file : {}".format(PORTS_INFO))
    #EX_INFO = dict(config.items('ALIAS'))#SEGMENT
    #print("Configuration file : {}".format(EX_INFO))



    #return PORTS_INFO



if __name__ == "__main__":
    stage = ""


    exercise_string = multiprocessing.Value("i",0)#(id esercizio)
    ex_count = multiprocessing.Value("i", 0)

    #deprecated: abbiamo la queue per i kp
    KeyPoints = multiprocessing.Array("i", 66) #dep


    string_from_tcp_ID = multiprocessing.Value("i", 0)

    #ports_info = init()
    # printing main program process id
    print("ID of main process: {}".format(os.getpid()))
    #A variant of Queue that retrieves most recently added entriesfirst(last in, firstout).
    time.sleep(5)
    q = multiprocessing.Queue(maxsize=1)


    # creating processes
    #LIFO queue 1, gli interessa solo dell ultimo elemento prodotto dallo skeletonizzatore

    p1 = multiprocessing.Process(target=SKEL.skeletonizer, args=(KeyPoints, exercise_string,q))
    p2 = multiprocessing.Process(target=EVA.evaluator, args=(exercise_string,q,string_from_tcp_ID))
    p3 = multiprocessing.Process(target=receiver.listen_for_TCP_string, args=(string_from_tcp_ID,))



    # starting processes
    p1.start()
    p2.start()
    p3.start()


    # process IDs
    print("ID of process p1: {}".format(p1.pid))
    print("ID of process p2: {}".format(p2.pid))
    print("ID of process p2: {}".format(p3.pid))

    # wait until processes are finished
    p1.join()
    p2.join()
    p3.join()

    # both processes finished
    print("Both processes finished execution!")

    # check if processes are alive
    print("Process p1 is alive: {}".format(p1.is_alive()))
    print("Process p2 is alive: {}".format(p2.is_alive()))
    print("Process p2 is alive: {}".format(p3.is_alive()))
