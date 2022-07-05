#!/usr/bin/env python

from __future__ import division
import cv2
import numpy as np
import socket
import struct
import EVA
import multiprocessing
from datetime import datetime

MAX_DGRAM = 2 ** 16

def search_for_user_info(string_from_tcp):
    #es: user_0_triceps_push_down_06-24-2022_09:34:36.csv
    codification_string = "setuser"
    splitted = string_from_tcp.split('_')
    if splitted[0] == codification_string:
        user = splitted[1]
        EVA.logging3.critical("new user communication detected : %s", user)
        return user
    else:
        return None




def dump_buffer(s):
    """ Emptying buffer frame """
    while True:
        seg, addr = s.recvfrom(MAX_DGRAM)
        EVA.logging3.critical("::RECEIVER:: %s",seg[0])
        if struct.unpack("B", seg[0:1])[0] == 1:
            EVA.logging3.critical("::RECEIVER::finish emptying buffer")
            break


def listen_for_TCP_string(string_from_tcp_ID,user_id):

    #port = 1025
    ip = '127.0.0.1'
    port = 21001
    #ip = '192.168.10.1' #mettere localhost per togliere errore '127.0.0.1'

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    s.bind((ip, port))
    EVA.logging3.critical("::RECEIVER::__reciving config: IP = {}, PORT = {}. ".format(ip, port))
    user_id.value = 0

    while True:

        seg, addr = s.recvfrom(MAX_DGRAM)  # here was the error lo da sempre e non legge mai la stringa inviata sara
        # perche deve stare in ascolto costante per poter beccare l esatto momento in cui arriva il pacchetto ,
        # unica proposta crea un nuovo processo parallelo che continua ad ascoltare e quando c e un  nuovo messaggio aggiorna il value di multiprocessing comune


        seg = seg.decode('utf-8')
        seg = str(seg)
        #taglip l ultima lettera perche manuel me le inviava con lo spazio dopo
        
        seg = seg[:-1]
        
        EVA.logging3.critical("::RECEIVER::__FROM TCP SEGB: %s",seg)

        user_research = search_for_user_info(seg)
        if user_research != None:
            EVA.logging3.critical("WELCOME new user! : %s", user_research)
            user_id.value = int(user_research)
        else:
            pass


        if seg == "stop":
            string_from_tcp_ID.value = 0
        elif seg == "pause":
            string_from_tcp_ID.value = 10
        elif seg == "start":
            string_from_tcp_ID.value = 1000
        else:
            string_from_tcp_ID.value = EVA.ex_string_to_ID(seg)
    s.close()


def main():
    """ Getting image udp frame &
    concate before decode and output image """
    EVA.logging3.critical("::RECEIVER::receiver listening")

    # Set up socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #Raddr = '127.0.0.1'
    #Rport = 5005
    Raddr = '192.168.10.2'
    Rport = 21001
    s.bind((Raddr, Rport))
    EVA.logging3.critical("::RECEIVER::receiving config: IP = {}, PORT = {}. ".format(Raddr, Rport))

    dump_buffer(s)

    while True:
        seg, addr = s.recvfrom(MAX_DGRAM)
        if struct.unpack("B", seg[0:1])[0] > 1:
            dat += seg[1:]
        else:
            dat += seg[1:]
            img = cv2.imdecode(np.fromstring(dat, dtype=np.uint8), 1)

            # cv2.imshow('frame', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            dat = b''

    # cap.release()
    cv2.destroyAllWindows()
    s.close()


if __name__ == "__main__":
    main()
