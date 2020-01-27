import cv2, time
print(cv2.__version__)
vidcap = cv2.VideoCapture('gaitreco.mp4')
success,image = vidcap.read()
count = 0
success = True
video_frames = []
while count < 900:
  #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
    if count % 3 == 0 and count > 300:
        image = image[0:240, 0:240]
        image = cv2.resize(image, (299,299), interpolation = cv2.INTER_AREA)
        cv2.imwrite("gaitreco/frame%d.jpg" % count, image)
        video_frames.append(image)
    success,image = vidcap.read()
    print ('Read a new frame: ', success)
    count += 1

#video_frames = []
from human_pose_nn import *
from gait_nn import *
import tensorflow as tf
start = time.time()
# Initialize computational graphs of both sub-networks
net_pose = HumanPoseIRNetwork()
net_gait = GaitNetwork(recurrent_unit = 'GRU', rnn_layers = 2)

# Load pre-trained models
net_pose.restore('./models/Human3.6m.ckpt')
net_gait.restore('./models/H3.6m-GRU-1.ckpt')

# Create features from input frames in shape (TIME, HEIGHT, WIDTH, CHANNELS) 
#video_frames = tf.convert_to_tensor(video_frames, dtype=tf.float32)
spatial_features = net_pose.feed_forward_features(video_frames)

# Process spatial features and generate identification vector 
identification_vector = net_gait.feed_forward(spatial_features)
end = time.time()
print((end-start))