# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
import time
import colorsys
import os
from timeit import default_timer as timer
import utils
import pytesseract
from keras_yolo3.mouseclick import video_click
from keras_yolo3.yolo_sih import YOLO_Plate, detect_image_Plate
from utils import load_extractor_model, load_features, parse_input, detect_object
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import cv2
from .yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from .yolo3.utils import letterbox_image
import argparse
import sys
import pyrebase
from keras.utils import multi_gpu_model
import re
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
state_region = ['AN','AP','AR','AS','BR','CG','CH','DD','DL','DN','GA','GJ','HR','HP','JH','JK','KL','KA','LD','MH','ML','MN','MP','MZ','NL','OD','PB','PY','RJ','SK','TN','TR','TS','UK','UP','WB']
number = ['0','1','2','3','4','5','6','7','8','9']
import json
import datetime
import requests
config = {
  "apiKey": "AIzaSyDaYILIFsZWzCl54rbQQMrGT5ET3o8Yj6U",
  "authDomain": "vechiledetection",
  "databaseURL": "https://vechiledetection.firebaseio.com/",
  "storageBucket": "gs://vechiledetection.appspot.com",
  "serviceAccount": "D:/LicensePlateWithYOLO/db/vechiledetection-firebase-adminsdk-kiblg-ddb38b4b7f.json"
}
firebase = pyrebase.initialize_app(config)

db = firebase.database()

currentDTKey = datetime.datetime.now() #For retreiving the key from The system date

DTKey = currentDTKey.strftime("%d%m%Y")

db.child("Entry").set(DTKey)
login_data={"r1[]":"PB22G",
            "r2":"4565",
            "auth":"Y29tLmRlbHVzaW9uYWwudmVoaWNsZWluZm8="}
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
    '''try:
        print('Vehicle ' + str(i+1) + ': \n' + vehicleOwner + '\n' + vehicleName + '\n' + vehicleClass + '\n' + vehicleRegion + '\n' + resultsPlate)
    except:
        print("Data not found")'''
#Start-of-Firebase-Operations
     #for retrieving the time from the system 
    vehicleTime = currentDT.strftime("%H%M%S")
    
    data = {"vno": str(text),
            "name": str(vehicleOwner),
            "make": str(vehicleName),
            "region": str(vehicleRegion),
            "vclass": str(vehicleClass)
             }
    
    db.child("Entry").child(DTKey).child(vehicleTime).set(data)
def get_parent_dir(n=1):
    """ returns the n-th parent dicrectory of the current
        working directory """
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


src_path = os.path.join(get_parent_dir(1), "2_Training", "src")
utils_path = os.path.join(get_parent_dir(1), "Utils")

sys.path.append(src_path)
sys.path.append(utils_path)


data_folder = os.path.join(get_parent_dir(n=1), "Data")

image_folder = os.path.join(data_folder, "Source_Images")

image_test_folder = os.path.join(image_folder, "Test_Images")

detection_results_folder = os.path.join(image_folder, "Test_Image_Detection_Results")
detection_results_file = os.path.join(detection_results_folder, "Detection_Results.csv")

model_folder = os.path.join(data_folder, "Model_Weights")

model_weights = 'D:/LicensePlateWithYOLO/Data/Model_Weights/trained_weights_final.h5'
model_classes = 'D:/LicensePlateWithYOLO/Data/Model_Weights/data_classes.txt'

anchors_path = 'D:\\LicensePlateWithYOLO\\2_Training\\src\\keras_yolo3\\model_data\\yolo_anchors.txt'
parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)


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


class YOLO(object):
    _defaults = {
        "model_path": "model_data/yolo.h5",
        "anchors_path": "model_data/yolo_anchors.txt",
        "classes_path": "model_data/coco_classes.txt",
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }
    
    
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(",")]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith(".h5"), "Keras model or weights must be a .h5 file."

        # Load model, or construct model and load weights.
        start = timer()
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = (
                tiny_yolo_body(
                    Input(shape=(None, None, 3)), num_anchors // 2, num_classes
                )
                if is_tiny_version
                else yolo_body(
                    Input(shape=(None, None, 3)), num_anchors // 3, num_classes
                )
            )
            self.yolo_model.load_weights(
                self.model_path
            )  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == num_anchors / len(
                self.yolo_model.output
            ) * (
                num_classes + 5
            ), "Mismatch between model and given anchor and class sizes"

        end = timer()
        print(
            "{} model, anchors, and classes loaded in {:.2f}sec.".format(
                model_path, end - start
            )
        )

        # Generate colors for drawing bounding boxes.
        if len(self.class_names) == 1:
            self.colors = ["GreenYellow"]
        else:
            hsv_tuples = [
                (x / len(self.class_names), 1.0, 1.0)
                for x in range(len(self.class_names))
            ]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(
                    lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors,
                )
            )
            np.random.seed(10101)  # Fixed seed for consistent colors across runs.
            np.random.shuffle(
                self.colors
            )  # Shuffle colors to decorrelate adjacent classes.
            np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(
            self.yolo_model.output,
            self.anchors,
            len(self.class_names),
            self.input_image_shape,
            score_threshold=self.score,
            iou_threshold=self.iou,
        )
        return boxes, scores, classes
    
    

    def detect_image(self, image, show_stats=True):
        
        start = timer()
        global vehicleId
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, "Multiples of 32 required"
            assert self.model_image_size[1] % 32 == 0, "Multiples of 32 required"
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (
                image.width - (image.width % 32),
                image.height - (image.height % 32),
            )
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype="float32")
        if show_stats:
            print(image_data.shape)
        image_data /= 255.0
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0,
            },
        )
        if show_stats:
            print("Found {} boxes for {}".format(len(out_boxes), "img"))
        out_prediction = []

        font_path = os.path.join(os.path.dirname(__file__), "font/FiraMono-Medium.otf")
        font = ImageFont.truetype(
            font=font_path, size=np.floor(3e-2 * image.size[1] + 0.5).astype("int32")
        )
        thickness = (image.size[0] + image.size[1]) // 300
        
        for i, c in reversed(list(enumerate(out_classes))):
            
            if self.class_names[c]=='car' or self.class_names[c]=='motorbike' or self.class_names[c]=='truck' or self.class_names[c]=='bus' :
                
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]
                if score>0.8:
                    
                    label = "{} {:.2f}".format(predicted_class, score)
                    draw = ImageDraw.Draw(image)
                    label_size = draw.textsize(label, font)
        	    
                    top, left, bottom, right = box
                    top = max(0, np.floor(top + 0.5).astype("int32"))
                    left = max(0, np.floor(left + 0.5).astype("int32"))
                    bottom = min(image.size[1], np.floor(bottom + 0.5).astype("int32"))
                    right = min(image.size[0], np.floor(right + 0.5).astype("int32"))
        
                    # image was expanded to model_image_size: make sure it did not pick
                    # up any box outside of original image (run into this bug when
                    # lowering confidence threshold to 0.01)
                    if top > image.size[1] or right > image.size[0]:
                        continue
                    if show_stats:
                        print(label, (left, top), (right, bottom))
                    # Predicting Plate 
                   
                    
                    out_prediction.append([left, top, right, bottom, c, score])
        
                    if top - label_size[1] >= 0:
                        text_origin = np.array([left, top - label_size[1]])
                    else:
                        text_origin = np.array([left, bottom])
        
                    # My kingdom for a good redistributable image drawing library.
                   
                    
                    for i in range(thickness):
                        draw.rectangle(
                            [left + i, top + i, right - i, bottom - i], outline=self.colors[c]
                        )
                    draw.rectangle(
                        [tuple(text_origin), tuple(text_origin + label_size)],
                        fill=self.colors[c],
                    )
            
                    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                    del draw
                    
            else:
                continue
    
        end = timer()
        if show_stats:
            print("Time spent: {:.3f}sec".format(end - start))
        return out_prediction, image
    

    def close_session(self):
        self.sess.close()
        
yolo_plate = YOLO_Plate(
        **{
            "model_path": FLAGS.model_path,
            "anchors_path": FLAGS.anchors_path,
            "classes_path": FLAGS.classes_path,
            "score": FLAGS.score,
            "gpu_num": FLAGS.gpu_num,
            "model_image_size": (416, 416),
        }
    )

def plate_no_recognizer(text):
  
    
    
    s = '{}'.format(text)
    filtered_text = re.sub(r'[^\w]','',s)
    
    
    state_code = filtered_text[0:2]
    if state_code in state_region:
       
        
        
        last_digit_counter=0
        last_numbers = filtered_text[-4:]
        for i in range(len(last_numbers)):
            if last_numbers[i] in number:
                last_digit_counter = last_digit_counter+1
                
        if last_digit_counter==4:
            a = filtered_text
            return a

def detect_video(yolo, video_path, output_path=""):
    
    frame_no = 0
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = cv2.VideoWriter_fourcc(*"mp4v")  # int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (
        int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    isOutput = True if output_path != "" else False
    if isOutput:
        print(
            "Processing {} with frame size {} at {:.1f} FPS".format(
                os.path.basename(video_path), video_size, video_fps
            )
        )
        # print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
   

    
    mousePoints = video_click(video_path)
    while vid.isOpened():
        return_value, frame = vid.read()
        
        cv2.line(frame,mousePoints[0],mousePoints[1],(0,0,255),2)
     
        frame_no = frame_no+1
        if not return_value:
            break
        # opencv images are BGR, translate to RGB
        frame = frame[:, :, ::-1]
        image = Image.fromarray(frame)
        out_pred, image = yolo.detect_image(image, show_stats=False)
        x,image = detect_image_Plate(yolo_plate,image,show_stats=False)
        image = np.asarray(image)
        if len(out_pred)!=0: 
            for i in range(len(out_pred)):
                # For Entering Vehicle 
                    
                    if int((out_pred[i][3]))>=int((mousePoints[0][1]+mousePoints[1][1])/2) :
                       
                        cv2.line(image,mousePoints[0],mousePoints[1],(0,255,0),2)
                        
                      
                        if len(x)!=0:
                            for j in range(len(x)):
                                left_plate = x[j][0]
                                top_plate = x[j][1]
                                right_plate = x[j][2]
                                bottom_plate = x[j][3]
                                score_plate = x[j][5]
                                
                                
                               
                                
                              
                                
                                if score_plate>0.8:
                                    roi = frame[top_plate:bottom_plate,left_plate:right_plate]
                                    cv2.imwrite('D:/objects/savedPlate.png'.format(j),roi)
                                    test = cv2.imread('D:/objects/savedPlate.png')
                                    
                                    img = cv2.resize(test, (333, 75))
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
                                    config = ('-l eng+hin --oem 1 --psm 3')
                                    text = pytesseract.image_to_string(img_dilate,config=config)
                                    # Lets Filter the text
                                    data = set()
                                    data.add(plate_no_recognizer(text))
                                    if len(data)!=0:
                                        for i in data:
                                            if i!=None:
                                                if len(i)<=10:
                                                    country_code = i
                                                    country_code = country_code[:len(country_code)-4]
                                                    plate_number = i
                                                    plate_number = plate_number[len(plate_number)-4:]
                                                    RTO(country_code,plate_number,i)
                                else:
                                    result = np.asarray(image)
                            curr_time = timer()
                            exec_time = curr_time - prev_time
                            prev_time = curr_time
                            accum_time = accum_time + exec_time
                            curr_fps = curr_fps + 1
                            if accum_time > 1:
                                accum_time = accum_time - 1
                                fps = "FPS: " + str(curr_fps)
                                curr_fps = 0
                            cv2.putText(
                                image,
                                text=fps,
                                org=(3, 15),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.50,
                                color=(255, 0, 0),
                                thickness=2,
                            )
                        
                    else:
                        result = np.asarray(image)
            else:
                    result = np.asarray(image)
               
                    
        else:
                result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("result",1000,1000)
        cv2.imshow("result", result)
        if isOutput:
                out.write(result[:, :, ::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    vid.release()
    out.release()
    # yolo.close_session()
