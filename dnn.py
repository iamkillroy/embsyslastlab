
#!/usr/bin/env python
# Imports
import collections
import math
import os
import sys
import time

import cv2
import numpy as np
import serial
import tensorflow as tf

try:
    ser1 = serial.Serial("/dev/ttyAMA1", 115200)
except:  # can't connect with the first let's go to the second
    try:
        ser1 = serial.Serial("/dev/ttyAMA2", 115200)
    except:
        raise RuntimeError(
            "Cannot find serial device to send data to. Intended target machine to run code is RPi 5"
        )


def send_to_hisense(message: str, byteObj: bytes = bytes(0x00)):
    """Sends a serial string in UTF-8 or a series of bytes objects (access with kwarg byteArray]"""
    if byteObj != 0: #check if it's the initial kwarg object
            ser1.write(byteObj)
    else:
        ser1.write(bytes(message, "utf-8"))

# Set environment variables for TensorFlow threading
def set_tf_config(ncpu):
    os.environ["OMP_NUM_THREADS"] = str(ncpu)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(ncpu)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(ncpu)
    tf.config.threading.set_inter_op_parallelism_threads(ncpu)
    tf.config.threading.set_intra_op_parallelism_threads(ncpu)
    tf.config.set_soft_device_placement(True)


# Radian <-> Degree conversion functions
def deg2rad(deg):
    return deg * math.pi / 180.0


def rad2deg(rad):
    return 180.0 * rad / math.pi


# Get the number of cores to be used by TensorFlow
if len(sys.argv) > 1:
    NCPU = int(sys.argv[1])
else:
    NCPU = 1

send_to_hisense("serial connection established")


batch_size = max(16, NCPU * 4)  # Larger batch size to better utilize multiple CPUs

print(f"Using batch size: {batch_size}")

# Set up TensorFlow configuration
physical_devices = tf.config.list_physical_devices("CPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_visible_devices(physical_devices[0], "CPU")
    print(f"Using CPU device: {physical_devices[0]}")

print(f"Trying to use {NCPU} CPUs")
set_tf_config(NCPU)

# Import the model
from model import create_model

# Load the model
model = create_model(input_shape=(66, 200, 3))
model.load_weights("model/model.h5")

#    and the number of frames already processed
NFRAMES = 1000
curFrame = 0

# Periodic task options
period = 50
is_periodic = True
# Create lists for tracking operation timings
cap_time_list = []
prep_time_list = []
pred_time_list = []
tot_time_list = []

print("---------- Processing video for epoch 1 ----------")
# Open the video file
vid_path = "epoch-1.avi"
assert os.path.isfile(vid_path)
cap = cv2.VideoCapture(vid_path)

# Process the video while recording the operation execution times
print("Performing inference...")
time_start = time.time()
first_frame = True
count: int = 0
while curFrame < NFRAMES:
    batch_frames = []
    batch_times = []

    # Collect frames for a batch
    for _ in range(batch_size):
        if curFrame >= NFRAMES:
            break

        cam_start = time.time()
        ret, img = cap.read()
        if not ret:
            break

        prep_start = time.time()

        # Preprocess the input frame
        img = cv2.resize(img, (200, 66))
        img = img / 255.0

        batch_frames.append(img)
        batch_times.append((cam_start, prep_start))

    if not batch_frames:
        break

    # Convert list to numpy array for batch prediction
    batch_input = np.array(batch_frames)

    # Perform batch prediction
    pred_start = time.time()
    predictions = model.predict(batch_input, verbose=1)
    pred_end = time.time()

    # Process prediction results
    for i, prediction in enumerate(predictions):
        if i == 0 and first_frame:
            first_frame = False
            continue

        rad = prediction[0]
        deg = rad2deg(rad)

        cam_start, prep_start = batch_times[i]

        # Calculate the timings for each step
        cam_time = (prep_start - cam_start) * 1000
        prep_time = (pred_start - prep_start) * 1000

        # Distribute prediction time proportionally
        pred_time_per_frame = (pred_end - pred_start) * 1000 / len(batch_frames)
        pred_time = pred_time_per_frame

        # Total time includes capture, preprocessing, and a portion of prediction
        tot_time = cam_time + prep_time + pred_time

        print(
            f"pred: {deg:0.2f} deg. took: {tot_time:0.2f} ms | cam={cam_time:0.2f} prep={prep_time:0.2f} pred={pred_time:0.2f}"
        )

        # Add timings to lists
        if not (i == 0 and first_frame):
            tot_time_list.append(tot_time)
            curFrame += 1
        if count % 4 == 0:  # send every four frames
            deg_bytes = deg.to_bytes(2, byteorder='big', signed=True) #convert to signed big endian
            send_to_hisense("", bytes([0x06]) + deg_bytes + bytes([0x0A]))
            # Wait for next period (only for the last frame in batch)
        if i == len(predictions) - 1 and is_periodic:
            wait_time = (period - tot_time) / 1000
            if wait_time > 0:
                time.sleep(wait_time)
        count += 1
cap.release()

# Calculate and output FPS/frequency
fps = curFrame / (time.time() - time_start)
print(
    "completed inference, total frames: {}, average fps: {} Hz".format(
        curFrame, round(fps, 1)
    )
)

# Calculate and display statistics of the total inferencing times
print("count: {}".format(len(tot_time_list)))
print("mean: {}".format(np.mean(tot_time_list)))
print("99.999pct: {}".format(np.percentile(tot_time_list, 99.999)))
print("99.99pct: {}".format(np.percentile(tot_time_list, 99.99)))
print("99.9pct: {}".format(np.percentile(tot_time_list, 99.9)))
print("99pct: {}".format(np.percentile(tot_time_list, 99)))
