#CREATING VIDEO CONTAINER FROM JSON

import sys
if len(sys.argv) < 2:
    raise ValueError('VIDEO PATH MUST BE PROVIDED WITH PYTHON FILE CALL ARGUMENT!')

import json 
try:
    with open('data.json') as f:
        data = json.load(f)
except ValueError:
    print('JSON DOES NOT EXIST YET!')
    
N_frames = data['n_frames']
size = tuple(data['frame_size'])#Y x X (cols x rows)
path = sys.argv[1]
fps = data['fps']

import cv2
import numpy as np
result = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)


#HERE CHANGE SKELETON COLOR (Blue,Green,Red) format

COLOR = (50, 0, 200) #LINES COLOR
SIZE = 6 #LINE SIZE
KYP_COLOR = (200,0,100) #KEYPOINTS COLOR    
KYP_SIZE = 4 #JOINT SIZE



#WHERE TO DRAW LINES BETWEEN KEYPOINTS
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


# FOR EACH PERSON OUT OF 6 (THOSE WITH LOW SCORES DRAWN OUT OF RANGE OF IMAGE..)
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for i, person in enumerate(keypoints_with_scores):
        draw_connections(frame, person, edges, confidence_threshold, i)
        draw_keypoints(frame, person, confidence_threshold)


# WRITING LINES        
def draw_connections(frame, keypoints, edges, confidence_threshold, cl):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLOR, SIZE)

    
# DRAW KEYPOINTS
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), KYP_SIZE, KYP_COLOR)


# WRITING SKELETON VIDEO FRAME BY FRAME
print('Writing skeleton video..')
for i in range(N_frames):
    keypoints = np.array(data['keypoints'][i])
    blanco = (np.ones((size[1], size[0], 3))*255).astype(np.uint8)
    loop_through_people(blanco, keypoints, EDGES, 0.2)
    result.write(blanco)
    if (i+1)%100==0:
        print(f'{i+1:>{len(str(N_frames))}}/{N_frames} frames written')
        
result.release()
print('Video is ready!')
