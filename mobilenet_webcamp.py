#!/usr/bin/python3

'''
Test program for MobileNet v1 1.0 224.
File 'graph' should be placed in ./model/ subfolder.
'''

from mvnc import mvncapi as mvnc
import sys
import numpy
import cv2
import os
import datetime
import time
from threading import Thread
from queue import Queue

print("Initialization started:", datetime.datetime.now())

path_to_networks = './model/'
graph_filename = 'graph'

NR_FRAMES = 300

mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 2)
devices = mvnc.enumerate_devices()
if len(devices) == 0:
    print('No devices found')
    quit()

NR_NCS = len(devices)
print("Number of NCSs is", NR_NCS)

#Load graph
with open(path_to_networks + graph_filename, mode='rb') as f:
    graphFileBuff = f.read()

devices = mvnc.enumerate_devices()

devs = []
for d in devices:
    devs.append(mvnc.Device(d))

for d in devs:
    d.open()

graphs = [mvnc.Graph('graph') for i in range(NR_NCS)]

fifoIns = [None] * NR_NCS
fifoOuts = [None] * NR_NCS
for i in range(NR_NCS):
    fifoIns[i], fifoOuts[i] = graphs[i].allocate_with_fifos(devs[i], graphFileBuff)

vc = cv2.VideoCapture(0)

while not vc.isOpened():
    pass

reqsize = 224

q = Queue(NR_FRAMES + NR_NCS)

def thread_func(i, q, graph, fifoIn, fifoOut):
    while True:
        frame = q.get()
        if frame is None:
            print("Finished", i)
            return
        graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, frame, 'user object')
        output, userobj = fifoOut.read_elem()
        top_inds = output.argsort()[::-1][:1]
        print(i, top_inds[0])

ts = [Thread(target=thread_func, args=(i, q, graphs[i], fifoIns[i], fifoOuts[i])) for i in range(NR_NCS)]

for t in ts:
    t.start()

time0 = time.time()
print("NN started:", datetime.datetime.now())

for i in range(NR_FRAMES):
    rval, frame = vc.read()
    frame = frame.astype(numpy.float32)
    frame = cv2.resize(frame, (reqsize, reqsize))
    q.put(frame)

for i in range(NR_NCS):
    q.put(None)

for t in ts:
    t.join()

time1 = time.time()
print("NN finished:", datetime.datetime.now())

print("FPS =", NR_FRAMES / (time1 - time0))

vc.release()

print(''.join(['*' for i in range(79)]))

for i in range(NR_NCS):
    fifoIns[i].destroy()
    fifoOuts[i].destroy()
    graphs[i].destroy()
    devs[i].close()
print('Finished')
