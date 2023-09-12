import os
import boto3
import cv2
import credentials
import pprint

# creating the directories
os.makedirs("output_dir")
os.makedirs(os.path.join("output_dir","anns"))
os.makedirs(os.path.join("output_dir","imgs"))

# getting the path of the needed directories
output_dir = "output_dir"
output_dir_anns = os.path.join(output_dir,"anns")
output_dir_imgs = os.path.join(output_dir,"imgs")

# create AWS Reko client
reko_client = boto3.client('rekognition',
                           aws_access_key_id=credentials.access_key,
                           aws_secret_access_key=credentials.secret_access_key,
                           region_name='us-west-2')

# set the target class
target_class = 'Zebra'

# load video
cap = cv2.VideoCapture("zebras.mp4")

# read frames
frame_num = 0
ret = True
while ret:
    ret, frame = cap.read()
    if ret:
        frame_num +=1
        frame = cv2.resize(frame, (854,480))
        H, W, _ = frame.shape

        # convert frame to jpg
        _, buffer = cv2.imencode(".jpg", frame)

        # convert buffer to bytes
        image_bytes = buffer.tobytes()

        # detect objects
        response = reko_client.detect_labels(Image={'Bytes': image_bytes},
                                            MinConfidence=50)

        with open(os.path.join(output_dir_anns, "format_{}.txt".format(str(frame_num).zfill(6))), "w") as f:
            for label in response['Labels']:
                if label['Name'] == target_class:
                    for instancenmr in range(len(label['Instances'])):
                        bbox = label['Instances'][instancenmr]['BoundingBox']
                        x1 = bbox['Left']
                        y1 = bbox['Top']
                        height = bbox['Height']
                        width = bbox['Width']

                        f.write("{} {} {} {} {}\n".format(0,
                                                          x1 + width/2,
                                                          y1 + height/2,
                                                          width,
                                                          height))
        cv2.imwrite(os.path.join(output_dir_imgs, "format_{}.jpg".format(frame_num)), frame)