'''
Name: real_time_histogram.py
Description: Uses OpenCV to display video from a camera or file
    and matplotlib to display and update either a grayscale or
    RGB histogram of the video in real time. For usage, type:
    > python real_time_histogram.py -h
Author: Najam Syed (github.com/nrsyed)
Created: 2018-Feb-07
'''

import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import time

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
    help='Path to video file (if not using camera)')
parser.add_argument('-c', '--color', type=str, default='gray',
    help='Color space: "gray" (default), "rgb", or "lab"')
parser.add_argument('-b', '--bins', type=int, default=16,
    help='Number of bins per channel (default 16)')
parser.add_argument('-w', '--width', type=int, default=0,
    help='Resize video to specified width in pixels (maintains aspect)')
args = vars(parser.parse_args())

# Configure VideoCapture class instance for using camera or file input.
if not args.get('file', False):
    capture = cv2.VideoCapture(1)
else:
    capture = cv2.VideoCapture(args['file'])

color = args['color']
bins = args['bins']
resizeWidth = args['width']

# Initialize plot.
fig, ax = plt.subplots(1,2, figsize=(10, 5))
if color == 'rgb':
    ax[0].set_title('Histogram (RGB)')
    ax[1].set_title('Histogram (RGB) CLAHE')
elif color == 'lab':
    ax[0].set_title('Histogram (L*a*b*)')
else:
    ax[0].set_title('Histogram (grayscale)')
ax[0].set_xlabel('Bin')
ax[0].set_ylabel('Frequency')

# Initialize plot line object(s). Turn on interactive plotting and show plot.
lw = 3
alpha = 0.5

lineR, = ax[0].plot(np.arange(bins), np.zeros((bins,)), c='r', lw=lw, alpha=alpha, label='Red')
lineG, = ax[0].plot(np.arange(bins), np.zeros((bins,)), c='g', lw=lw, alpha=alpha, label='Green')
lineB, = ax[0].plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha, label='Blue')

ax[0].set_xlim(0, bins-1)
ax[0].set_ylim(0, 1)
ax[0].legend()
plt.ion()
plt.show()

lineR1, = ax[1].plot(np.arange(bins), np.zeros((bins,)), c='r', lw=lw, alpha=alpha, label='Red')
lineG1, = ax[1].plot(np.arange(bins), np.zeros((bins,)), c='g', lw=lw, alpha=alpha, label='Green')
lineB1, = ax[1].plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha, label='Blue')

ax[1].set_xlim(0, bins-1)
ax[1].set_ylim(0, 1)
time_text = ax[1].text(1.6, 0.95, '', transform=ax[0].transAxes, ha='center', va='center', color='black')

plt.ion()
plt.show()

# Definindo o CLHAE:
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Grab, process, and display video frames. Update plot line object(s).
i = 0
while True:
    (grabbed, frame) = capture.read()

    if not grabbed:
        break

    # Resize frame to width, if specified.
    if resizeWidth > 0:
        (height, width) = frame.shape[:2]
        resizeHeight = int(float(resizeWidth / width) * height)
        frame = cv2.resize(frame, (resizeWidth, resizeHeight),
            interpolation=cv2.INTER_AREA)
        
    if i%15 == 0:

        # Normalize histograms based on number of pixels per frame.
        numPixels = np.prod(frame.shape[:2])
        
        cv2.imshow('RGB', frame)
        (b, g, r) = cv2.split(frame)
        # Processing:
        sc = time.time()
        b_clahe, g_clahe, r_clahe = clahe.apply(b), clahe.apply(g), clahe.apply(r)
        ec = time.time()
        
        time_text.set_text(f'Tempo de processamento: {(ec - sc)*1000:.4} ms')
        cv2.imshow('CLAHE', np.stack((b_clahe, g_clahe, r_clahe), axis=2))

       
       
        histogramR = cv2.calcHist([r], [0], None, [bins], [0, 256]) / numPixels
        histogramG = cv2.calcHist([g], [0], None, [bins], [0, 256]) / numPixels
        histogramB = cv2.calcHist([b], [0], None, [bins], [0, 256]) / numPixels
        lineR.set_ydata(histogramR)
        lineG.set_ydata(histogramG)
        lineB.set_ydata(histogramB)
       
        histogramR1 = cv2.calcHist([r_clahe], [0], None, [bins], [0, 256]) / numPixels
        histogramG1 = cv2.calcHist([g_clahe], [0], None, [bins], [0, 256]) / numPixels
        histogramB1 = cv2.calcHist([b_clahe], [0], None, [bins], [0, 256]) / numPixels
        lineR1.set_ydata(histogramR1)
        lineG1.set_ydata(histogramG1)
        lineB1.set_ydata(histogramB1)
    
        fig.canvas.draw()
        plt.pause(0.001)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    i+=1

capture.release()
cv2.destroyAllWindows()