#!/usr/bin/env python

from __future__ import division
import cv2
import numpy as np
import socket
import struct
import math

import time


def main():





    ip = '127.0.0.1'

    status = ["lat_pull_down_supinated_grip ", "start "]
    #status = ["lat_pull_down_supinated_grip ", "start ", "stop ", "pulls_apart ", "start ", "stop "]
    port = 21001
    print("sending config: IP = {}, PORT = {}. ".format(ip, port))

    

    for i in range(len(status)):
        ex_string_recived = status[i]
        time.sleep(12)

        BUFFER_SIZE = 1024

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        host = socket.gethostname()  # Get the local machine name

        s.connect((ip, port))
        s.send(ex_string_recived.encode())
        print("sended: {}".format(ex_string_recived))

        s.close()




if __name__ == "__main__":
    main()
