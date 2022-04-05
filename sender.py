#!/usr/bin/env python

from __future__ import division
import cv2
import numpy as np
import socket
import struct
import math
import multiprocessing


class FrameSegment(object):
    """ 
    Object to break down image frame segment
    if the size of image exceed maximum datagram size 
    """
    MAX_DGRAM = 2**16
    MAX_IMAGE_DGRAM = MAX_DGRAM - 64 # extract 64 bytes in case UDP frame overflown
    def __init__(self, sock, port, addr='127.0.0.1'):
        self.s = sock
        self.port = port
        self.addr = addr

    def udp_frame(self, img):
        """ 
        Compress image and Break down
        into data segments 
        """

        """ resize image before send
        scale_percentage=.50
        width=int(img.shape[1]*scale_percentage)
        height=int(img.shape[0]*scale_percentage)

        resized_img=cv2.resize(img,(width,height))
        compress_img = cv2.imencode('.jpg', resized_img)[1]
        """
        compress_img = cv2.imencode('.jpg', img)[1]
        dat = compress_img.tostring()
        size = len(dat)
        num_of_segments = math.ceil(size/(self.MAX_IMAGE_DGRAM))
        array_pos_start = 0
        while num_of_segments:
            array_pos_end = min(size, array_pos_start + self.MAX_IMAGE_DGRAM)
            self.s.sendto(struct.pack("B", num_of_segments) +
                dat[array_pos_start:array_pos_end], 
                (self.addr, self.port)
                )
            
            array_pos_start = array_pos_end
            num_of_segments -= 1


def send_status(port, status1,status2, ip='localhost'):
    #if __name__ == "__main__":
        BUFFER_SIZE = 1024

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        host = socket.gethostname()  # Get the local machine name
        # print("i am : {}".format(host))
        s.connect((ip, port))
        status =[float(status1),float(status2)]
       
        try:
            tmp = struct.pack(f'{len(status)}d',*status)
        except:
            disp(status)
        
        try:
            s.send(tmp)
        except:
            disp(tmp)
        #print(status)
        #s.send((str(status)+'\n').encode())
        
        #s.send(struct.pack(f'{len(status)}d',*status))
        #print("sended : {}".format(status))
        # data = s.recv(BUFFER_SIZE)
        s.close()
        #print("s closd _||")
        # print("received data: {}".format(data))


def stream(img):
    #if __name__ == "__main__":
        """ Top level main function """

        #print("sender working")

        # Set up UDP socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        port = 5005

        fs = FrameSegment(s, port)

        fs.udp_frame(img)

        s.close()


