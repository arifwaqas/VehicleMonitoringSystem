import os
import sys


def get_parent_dir(n=1):
    """ returns the n-th parent dicrectory of the current
    working directory """
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


src_path = os.path.join(get_parent_dir(1), "2_Training", "src")
utils_path = os.path.join(get_parent_dir(1), "Utils")
import cv2
sys.path.append(src_path)
sys.path.append(utils_path)
import pytesseract
import argparse
from keras_yolo3.yolo import YOLO, detect_video
from PIL import Image
from timeit import default_timer as timer
from utils import load_extractor_model, load_features, parse_input, detect_object
import test
import utils
import pandas as pd
import numpy as np
from Get_File_Paths import GetFileList
import random
import json
import requests
# import pyrebase
import datetime
'''config = {
  "apiKey": "AIzaSyDaYILIFsZWzCl54rbQQMrGT5ET3o8Yj6U",
  "authDomain": "vechiledetection",
  "databaseURL": "https://vechiledetection.firebaseio.com/",
  "storageBucket": "gs://vechiledetection.appspot.com",
  "serviceAccount": "C:/Users/ARIF/Desktop/LicensePlateWithYOLO/db/serviceAccountCredentials.json"
}
firebase = pyrebase.initialize_app(config)

db = firebase.database()

currentDTKey = datetime.datetime.now() #For retreiving the key from The system date

DTKey = currentDTKey.strftime("%d%m%Y")

db.child("Entry").set(DTKey)
#End OF Config

#Start-RTO

def RTO(country,plate_no,text):
    
    currentDT = datetime.datetime.now()
    
    login_data["r1[]"]=country
    login_data["r2"]=plate_no
    r = requests.post("https://rtovehicle.info/batman.php",login_data)
    response = r.content
    my_json = response.decode('utf8').replace("'", '"')
    #print(my_json)
 
    # Load the JSON to a Python list & dump it back out as formatted JSON
    data = json.loads(my_json)
    s = json.dumps(data, indent=4, sort_keys=True)
    #print(s)
    vehicleOwner = data.get('owner_name')
    vehicleName = data.get('vehicle_name')
    vehicleRegion = data.get('regn_auth')
    vehicleClass = data.get('vh_class')
    try:
        print('Vehicle ' + str(i+1) + ': \n' + vehicleOwner + '\n' + vehicleName + '\n' + vehicleClass + '\n' + vehicleRegion + '\n' + resultsPlate)
    except:
        print("Data not found")
#Start-of-Firebase-Operations
     #for retrieving the time from the system 
    vehicleTime = currentDT.strftime("%H%M%S")
    
    data = {"vno": str(text),
            "name": str(vehicleOwner),
            "make": str(vehicleName),
            "region": str(vehicleRegion),
            "vclass": str(vehicleClass)
             }
    
    db.child("Entry").child(DTKey).child(vehicleTime).set(data)'''
    
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set up folder names for default values
data_folder = os.path.join(get_parent_dir(n=1), "Data")

image_folder = os.path.join(data_folder, "Source_Images")

image_test_folder = os.path.join(image_folder, "Test_Images")

detection_results_folder = os.path.join(image_folder, "Test_Image_Detection_Results")
detection_results_file = os.path.join(detection_results_folder, "Detection_Results.csv")

model_folder = os.path.join(data_folder, "Model_Weights")

model_weights = os.path.join(model_folder, "yolo.h5")
model_classes = os.path.join(model_folder, "coco_classes.txt")

anchors_path = os.path.join(src_path, "keras_yolo3", "model_data", "yolo_anchors.txt")

FLAGS = None

if __name__ == "__main__":
    # Delete all default flags
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    """
    Command line options
    """

    parser.add_argument(
        "--input_path",
        type=str,
        default=image_test_folder,
        help="Path to image/video directory. All subdirectories will be included. Default is "
        + image_test_folder,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=detection_results_folder,
        help="Output path for detection results. Default is "
        + detection_results_folder,
    )

    parser.add_argument(
        "--no_save_img",
        default=False,
        action="store_true",
        help="Only save bounding box coordinates but do not save output images with annotated boxes. Default is False.",
    )

    parser.add_argument(
        "--file_types",
        "--names-list",
        nargs="*",
        default=[],
        help="Specify list of file types to include. Default is --file_types .jpg .jpeg .png .mp4",
    )

    parser.add_argument(
        "--yolo_model",
        type=str,
        dest="model_path",
        default=model_weights,
        help="Path to pre-trained weight files. Default is " + model_weights,
    )

    parser.add_argument(
        "--anchors",
        type=str,
        dest="anchors_path",
        default=anchors_path,
        help="Path to YOLO anchors. Default is " + anchors_path,
    )

    parser.add_argument(
        "--classes",
        type=str,
        dest="classes_path",
        default=model_classes,
        help="Path to YOLO class specifications. Default is " + model_classes,
    )

    parser.add_argument(
        "--gpu_num", type=int, default=1, help="Number of GPU to use. Default is 1"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        dest="score",
        default=0.25,
        help="Threshold for YOLO object confidence score to show predictions. Default is 0.25.",
    )

    parser.add_argument(
        "--box_file",
        type=str,
        dest="box",
        default=detection_results_file,
        help="File to save bounding box results to. Default is "
        + detection_results_file,
    )

    parser.add_argument(
        "--postfix",
        type=str,
        dest="postfix",
        default="_catface",
        help='Specify the postfix for images with bounding boxes. Default is "_catface"',
    )

    FLAGS = parser.parse_args()

    save_img = not FLAGS.no_save_img

    file_types = FLAGS.file_types

    if file_types:
        input_paths = GetFileList(FLAGS.input_path, endings=file_types)
    else:
        input_paths = GetFileList(FLAGS.input_path)

    # Split images and videos
    img_endings = (".jpg", ".jpg", ".png")
    vid_endings = (".mp4", ".mpeg", ".mpg", ".avi")

    input_image_paths = []
    input_video_paths = []
    for item in input_paths:
        if item.endswith(img_endings):
            input_image_paths.append(item)
        elif item.endswith(vid_endings):
            input_video_paths.append(item)

    output_path = FLAGS.output
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # define YOLO detector
    yolo = YOLO(
        **{
            "model_path": FLAGS.model_path,
            "anchors_path": FLAGS.anchors_path,
            "classes_path": FLAGS.classes_path,
            "score": FLAGS.score,
            "gpu_num": FLAGS.gpu_num,
            "model_image_size": (416, 416),
        }
    )

    # Make a dataframe for the prediction outputs
    out_df = pd.DataFrame(
        columns=[
            "image",
            "image_path",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "label",
            "confidence",
            "x_size",
            "y_size",
        ]
    )

    # labels to draw on images
    class_file = open(FLAGS.classes_path, "r")
    input_labels = [line.rstrip("\n") for line in class_file.readlines()]
    print("Found {} input labels: {} ...".format(len(input_labels), input_labels))

    if input_image_paths:
        print(
            "Found {} input images: {} ...".format(
                len(input_image_paths),
                [os.path.basename(f) for f in input_image_paths[:5]],
            )
        )
        start = timer()
        text_out = ""

        # This is for images
        for i, img_path in enumerate(input_image_paths):
            print(img_path)
            prediction,car_plate_prediction, image = detect_object(
                yolo,
                img_path,
                save_img=save_img,
                save_img_path=FLAGS.output,
                postfix=FLAGS.postfix,
            )
            y_size, x_size, _ = np.array(image).shape
            for single_prediction in car_plate_prediction:
                out_df = out_df.append(
                    pd.DataFrame(
                        [
                            [
                                os.path.basename(img_path.rstrip("\n")),
                                img_path.rstrip("\n"),
                            ]
                            + single_prediction
                            + [x_size, y_size]
                        ],
                        columns=[
                            "image",
                            "image_path",
                            "xmin",
                            "ymin",
                            "xmax",
                            "ymax",
                            "label",
                            "confidence",
                            "x_size",
                            "y_size",
                        ],
                    )
                )
        end = timer()
        print(
            "Processed {} images in {:.1f}sec - {:.1f}FPS".format(
                len(input_image_paths),
                end - start,
                len(input_image_paths) / (end - start),
            )
        )
        out_df.to_csv(FLAGS.box, index=False)
        df = pd.read_csv(os.path.join(detection_results_file))
        
        pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
        for i in range(df.shape[0]):
            img = cv2.imread('{}'.format(df['image_path'][i]))
            image = img[df['ymin'][i]:df['ymax'][i],df['xmin'][i]:df['xmax'][i]]
            cv2.imshow("im",image)
            img = cv2.resize(image, (333, 75))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("grayim",img_gray)
            _, img_binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            img_erode = cv2.erode(img_binary, (3,3))
            img_dilate = cv2.dilate(img_erode, (3,3))
        
            LP_WIDTH = img_dilate.shape[0]
            LP_HEIGHT = img_dilate.shape[1]
        
            # Make borders white
            img_dilate[0:3,:] = 255
            img_dilate[:,0:3] = 255
            img_dilate[72:75,:] = 255
            img_dilate[:,330:333] = 255
        
            # Estimations of character contours sizes of cropped license plates
            dimensions = [LP_WIDTH/6, LP_WIDTH/2, LP_HEIGHT/10, 2*LP_HEIGHT/3]
            cv2.imshow("im_dil",img_dilate)
            config = ('-l eng --oem 1 --psm 3')
            resultsPlate = pytesseract.image_to_string(img_dilate,config=config)
            print(resultsPlate)
            pathSavedLP = os.path.join(detection_results_folder,'savedLP.png')
            cv2.imwrite( pathSavedLP, img_dilate)
            country_code = resultsPlate[:len(resultsPlate)-4]
            plate_number = resultsPlate[len(resultsPlate)-4:]
     
            '''login_data={"r1[]":"PB22G",
                "r2":"4565",
                "auth":"Y29tLmRlbHVzaW9uYWwudmVoaWNsZWluZm8="}
     
            RTO(country_code,plate_number,resultsPlate)'''
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            
    # This is for videos
    if input_video_paths:
        print(
            "Found {} input videos: {} ...".format(
                len(input_video_paths),
                [os.path.basename(f) for f in input_video_paths[:5]],
            )
        )
        start = timer()
        for i, vid_path in enumerate(input_video_paths):
            output_path = os.path.join(
                FLAGS.output,
                os.path.basename(vid_path).replace(".", FLAGS.postfix + "."),
            )
            detect_video(yolo, vid_path, output_path=output_path)

        end = timer()
        print(
            "Processed {} videos in {:.1f}sec".format(
                len(input_video_paths), end - start
            )
        )
    # Close the current yolo session
    yolo.close_session()
