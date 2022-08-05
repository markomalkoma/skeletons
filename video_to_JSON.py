#LOADING and INSPECTING VIDEO
import sys
if len(sys.argv) < 2:
    raise ValueError('VIDEO PATH MUST BE PROVIDED WITH PYTHON FILE CALL ARGUMENT!')

import cv2
cap = cv2.VideoCapture(sys.argv[1])

N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cols, rows = int(cap.get(3)), int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

if N_frames == 0:
    raise ValueError('PATH IS EMPTY!')

print(f'Number of frames: {N_frames}')
print(f'Frame size: {rows} rows, {cols} cols')
print(f'Number of frames per second: {fps}')


#IMPORTS
print('Importing libraries..')

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json

print('Loading detection model..')
model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']


#CREATING JSON
data = {'n_frames':N_frames, 'frame_size':(cols,rows), 'fps':fps, 'keypoints': []}

print('Building JSON..')

for i in range(N_frames):
    
    # Loading frames
    ret, frame = cap.read()# ret is False for last frame
    
    # Rehaping to 4D
    rows, cols, colors = frame.shape
    image = frame.reshape(1, rows, cols, colors)# numpy to Tensorflow
    
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    scaled_rows = int((((rows/cols)*256)//32)*32)
    image = tf.cast(tf.image.resize_with_pad(image, scaled_rows, 256), dtype=tf.int32)
    
    # Detection
    outputs = movenet(image)
    
    # Bringing back to Numpy Array
    keypoints = outputs['output_0']# Output is a [1, 6, 56] tensor for each iteration
    # can detect up to 6 people
    # 51 values are x,y,score for 17 body points
    # 51-56 are box points
    # fltering first 51, after tf.tensor -> np.array
    keypoints = outputs['output_0'].numpy()[0,:,:51]# (1,6,51)
    
    # array with 6 people and 17 arrays for each keypoint
    keypoints_with_scores = keypoints.reshape((6,17,3))

    data['keypoints'].append(keypoints_with_scores.tolist())
    
    if (i+1)%100==0:
        print(f'{i+1:>{len(str(N_frames))}}/{N_frames} frames processed')
                
cap.release()

with open('data.json', 'w') as f:
    json.dump(data, f)

print('JSON is ready!')
