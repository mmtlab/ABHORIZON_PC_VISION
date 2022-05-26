#!/usr/bin/env python

from __future__ import division
import cv2
import numpy as np
import socket
import struct

MAX_DGRAM = 2 ** 16




def main():
    ip='127.0.0.1'
    port=1200

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)



    s.bind((ip, port))

    print("reciving config: IP = {}, PORT = {}. ".format(ip, port))


    while True:

        seg, addr = s.recvfrom(MAX_DGRAM)
        seg = seg.decode('utf-8')
        print(seg)
        #if not seg:
            #print("no data recived")


        print('Received :', format(seg))

    s.close()








if __name__ == "__main__":
    main()
