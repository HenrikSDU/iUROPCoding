import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

from ultralytics import YOLO

from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
import torchvision.ops as ops

#######################################################
# YOLO
#######################################################


def benchmark_summary_YOLO(results, model, data_source, data_time):
        
    benchmark_results_YOLO = pd.DataFrame(columns=['URL',
                                                   'TIME_SG',
                                                   'DETECTION_TIME',
                                                   'CAR_COUNT', 
                                                   'CAR_AVG_CONF',
                                                   'MOTOR_CYCLE_COUNT',
                                                   'MC_AVG_CONF',
                                                   'BUS_COUNT',
                                                   'BUS_AVG_CONF',
                                                   'TRUCK_COUNT',
                                                   'TRUCK_AVG_CONF'
                                                   ])
    for i, result in enumerate(results):
        
        boxes = result.boxes
        cls_tensor = boxes.cls
        conf_tensor = boxes.conf

        # Reset counts and confidence counts
        car_count = 0
        car_conf_sum = 0
        motor_cycle_count = 0
        motor_cycle_conf_sum = 0
        bus_count = 0
        bus_conf_sum = 0
        truck_count = 0
        truck_conf_sum = 0
        # Count objects
        for j in range(len(cls_tensor)):
            
            class_index = int(cls_tensor[j].item())
            class_name = model.names[class_index]

            class_confidence = conf_tensor[j].cpu()

            if class_name == 'car':
                car_count += 1
                car_conf_sum += class_confidence
            elif class_name == 'truck':
                truck_count += 1
                truck_conf_sum += class_confidence
            elif class_name == 'bus':
                bus_count += 1
                bus_conf_sum += class_confidence
            else:
                motor_cycle_count += 1
                motor_cycle_conf_sum += class_confidence
            
            print(f"Object {j+1}: {class_name} (index {class_index})", f" with confidence:{class_confidence}")    

        
        # Evaluate average confidence for the different classes & bring them from GPU to CPU memory
    
        car_AVG_conf = car_conf_sum / car_count if car_count != 0 else 'N/A'
        car_AVG_conf = car_AVG_conf.cpu().numpy() if car_AVG_conf != 'N/A' else car_AVG_conf
        bus_AVG_conf = bus_conf_sum / bus_count if bus_count != 0 else 'N/A'
        bus_AVG_conf = bus_AVG_conf.cpu().numpy() if bus_AVG_conf != 'N/A' else bus_AVG_conf
        truck_AVG_conf = truck_conf_sum / truck_count if truck_count != 0 else 'N/A'
        truck_AVG_conf = truck_AVG_conf.cpu().numpy() if truck_AVG_conf != 'N/A' else truck_AVG_conf
        MC_AVG_conf = motor_cycle_conf_sum / motor_cycle_count if motor_cycle_count != 0 else 'N/A'
        MC_AVG_conf = MC_AVG_conf.cpu().numpy() if MC_AVG_conf != 'N/A' else MC_AVG_conf

        detection_time = result.speed  
        detection_time = detection_time['preprocess'] + detection_time['inference'] + detection_time['postprocess']
        
        row = pd.DataFrame({'URL': [data_source[i]],
                            'TIME_SG':[data_time[i]],
                            'DETECTION_TIME': [detection_time],
                            'CAR_COUNT': [car_count], 
                            'CAR_AVG_CONF': [car_AVG_conf],
                            'MOTOR_CYCLE_COUNT': [motor_cycle_count],
                            'MC_AVG_CONF': [MC_AVG_conf],
                            'BUS_COUNT': [bus_count],
                            'BUS_AVG_CONF': [bus_AVG_conf],
                            'TRUCK_COUNT': [truck_count],
                            'TRUCK_AVG_CONF': [truck_AVG_conf]})
        

        #result.show()
        benchmark_results_YOLO = pd.concat([benchmark_results_YOLO, row], ignore_index=True)

    return benchmark_results_YOLO


#######################################################
# Transformer
#######################################################

DETECTION_THRESHOLD = 0.2
NEEDED_LABELS = ["bus", "car", "truck", "motorcycle"]

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
transformer_resnet_50 = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]



def detect_object_using_transformer(image_path, visualize=False, jpeg=False):
    # Open the image
    if jpeg:
        image = Image.open(image_path)
    start_time = datetime.now()
    # Preprocessing
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Inference
    outputs = transformer_resnet_50(**inputs)

    # Postprocessing: convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = feature_extractor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
    
    detection_time = (datetime.now() - start_time).microseconds * 0.001  # Milliseconds
    
    # Extract bounding boxes, scores, and labels
    boxes = results['boxes'].detach()
    scores = results['scores'].detach()
    labels = results['labels'].detach()

    # Perform Non-Maximum Suppression (NMS)
    iou_threshold = 0.8
    keep_indices = ops.nms(boxes, scores, iou_threshold).tolist()

    # Filter results based on NMS
    results['boxes'] = [results['boxes'][i].detach().numpy() for i in keep_indices]
    results['scores'] = [results['scores'][i].detach().numpy() for i in keep_indices]
    results['labels'] = [results['labels'][i].detach().numpy() for i in keep_indices]

    if visualize:
        plt.figure(figsize=(16, 10))
        plt.imshow(image)
        ax = plt.gca()
        colors = plt.cm.hsv(torch.linspace(0, 1, len(keep_indices))).tolist()  # Generate a list of colors

        for box, score, label, color in zip(results['boxes'], results['scores'], results['labels'], colors):
            xmin, ymin, xmax, ymax = box
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=3)
            ax.add_patch(rect)
            text = f'{transformer_resnet_50.config.id2label[int(label)]}: {score:.2f}'
            ax.text(xmin, ymin - 10, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))

        plt.axis('off')
        plt.show()

    return results, detection_time










def benchmark_summary_transformer(results, model=transformer_resnet_50, detection_time="N/A", data_source="N/A", data_time="N/A", log=False): 

    benchmark_results_transformer = pd.DataFrame(columns=['URL',
                                                   'TIME_SG',
                                                   'DETECTION_TIME',
                                                   'CAR_COUNT', 
                                                   'CAR_AVG_CONF',
                                                   'CAR_BOXES',
                                                   'MOTOR_CYCLE_COUNT',
                                                   'MC_AVG_CONF',
                                                   'MC_BOXES',
                                                   'BUS_COUNT',
                                                   'BUS_AVG_CONF',
                                                   'BUS_BOXES'
                                                   'MC_BOXES',
                                                   'TRUCK_COUNT',
                                                   'TRUCK_AVG_CONF',
                                                   'TRUCK_BOXES']
                                                    )
    for i, result in enumerate(results):

        # Reset counts and confidence counts
        car_count = 0
        car_conf_sum = 0
        car_boxes = []
        motor_cycle_count = 0
        motor_cycle_conf_sum = 0
        mc_boxes = []
        bus_count = 0
        bus_conf_sum = 0
        bus_boxes = []
        truck_count = 0
        truck_conf_sum = 0                                                                             
        truck_boxes = []

        for score, label, boxlist in zip(result["scores"], result["labels"], result["boxes"]):

            label = model.config.id2label[label.item()]    
            # let's only keep detections with score > DETECTION_THRESHOLD
            if (score.item() > DETECTION_THRESHOLD and label in NEEDED_LABELS):
                box = [int(j) for j in boxlist.tolist()]
            if log:
                print(f"Detected object with label {label} and score {score}")
                print(f"Bounding box coordinates: {box}")

            if label == 'car':
                car_count += 1
                car_conf_sum += score
                car_boxes.append(box)
            elif label == 'truck':
                truck_count += 1
                truck_conf_sum += score
                truck_boxes.append(box)
            elif label == 'bus':
                bus_count += 1
                bus_conf_sum += score
                bus_boxes.append(box)
            elif label == 'motorcycle':
                motor_cycle_count += 1
                motor_cycle_conf_sum += score
                mc_boxes.append(box)

        # Evaluate average confidence for the different classes
        if log:
            print(f"Car count: {car_count}",type(car_boxes))
        car_AVG_conf = car_conf_sum / car_count if car_count != 0 else 'N/A'
        
        bus_AVG_conf = bus_conf_sum / bus_count if bus_count != 0 else 'N/A'
        
        truck_AVG_conf = truck_conf_sum / truck_count if truck_count != 0 else 'N/A'
        
        MC_AVG_conf = motor_cycle_conf_sum / motor_cycle_count if motor_cycle_count != 0 else 'N/A'
       

        row = pd.DataFrame({'URL': [data_source[i]],
                            'TIME_SG':[data_time[i]],
                            'DETECTION_TIME': [detection_time],
                            'CAR_COUNT': [car_count],
                            'CAR_AVG_CONF': [car_AVG_conf],
                            'CAR_BOXES': [car_boxes], 
                            'MOTOR_CYCLE_COUNT': [motor_cycle_count],
                            'MC_AVG_CONF': [MC_AVG_conf],
                            'MC_BOXES': [mc_boxes], 
                            'BUS_COUNT': [bus_count],
                            'BUS_AVG_CONF': [bus_AVG_conf],
                            'BUS_BOXES': [bus_boxes], 
                            'TRUCK_COUNT': [truck_count],
                            'TRUCK_AVG_CONF': [truck_AVG_conf],
                            'TRUCK_BOXES': [truck_boxes]
                            })
        
       
        benchmark_results_transformer = pd.concat([benchmark_results_transformer, row], ignore_index=True)

    return benchmark_results_transformer

def sort_by_time_of_day(df, datetime_col):
   
    # Convert the datetime column to datetime if it's not already
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Extract the time
    df['TIME_ONLY'] = df[datetime_col].dt.time

    # Sort by the 'TIME_ONLY' column
    df_sorted = df.sort_values(by='TIME_ONLY')

    # Drop the 'TIME_ONLY' column 
    df_sorted = df_sorted.drop(columns=['TIME_ONLY'])

    return df_sorted

def remove_na_from_lists(*lists):
    cleaned_lists = []
    for lst in lists:
        cleaned_list = [conf for conf in lst if conf != 'N/A']
        cleaned_lists.append(cleaned_list)
    return cleaned_lists


#######################################################
# Mask RCNN
#######################################################
ID_MAPPING = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush',
}
category_index = {k: {'id': k, 'name': ID_MAPPING[k]} for k in ID_MAPPING}


import subprocess
from collections import Counter


# Command to clone the repository
command = ["git", "clone", "https://github.com/tensorflow/tpu/"]

# Execute the command
result = subprocess.run(command, capture_output=True, text=True)

# Print the output
print(result.stdout)
print(result.stderr)

from IPython import display
from PIL import Image
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.python.lib.io import file_io
import sys
sys.path.insert(0, 'tpu/models/official')
sys.path.insert(0, 'tpu/models/official/mask_rcnn')
import coco_metric
from mask_rcnn.object_detection import visualization_utils
import gcsfs
import shutil

gcs_path = 'gs://cloud-tpu-checkpoints/mask-rcnn/1555659850'
local_path = '/tmp/mask-rcnn'
fs = gcsfs.GCSFileSystem()
fs.get(gcs_path, local_path, recursive=True)

session = tf.Session(graph=tf.Graph())

# Ensure TensorFlow handles memory growth properly
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load the saved model from local path
loaded_model = tf.saved_model.loader.load(session, ['serve'],local_path)

print("Mask RCNN loaded successfully")



def detect_object_using_mask_rcnn(image_path, visualize=False):
    
    with open(image_path, 'rb') as f:
        np_image_string = np.array([f.read()])

    image = Image.open(image_path)
    width, height = image.size
    np_image = np.array(image.getdata()).reshape(height, width, 3).astype(np.uint8)
    
    start_time = datetime.now()
    num_detections, detection_boxes, detection_classes, detection_scores, detection_masks, image_info = session.run(
    ['NumDetections:0', 'DetectionBoxes:0', 'DetectionClasses:0', 'DetectionScores:0', 'DetectionMasks:0', 'ImageInfo:0'],
    feed_dict={'Placeholder:0': np_image_string})
    detection_time = (datetime.now() - start_time).microseconds * 0.001 # Miliseconds

    num_detections = np.squeeze(num_detections.astype(np.int32), axis=(0,))
    detection_boxes = np.squeeze(detection_boxes * image_info[0, 2], axis=(0,))[0:num_detections]
    detection_scores = np.squeeze(detection_scores, axis=(0,))[0:num_detections]
    detection_classes = np.squeeze(detection_classes.astype(np.int32), axis=(0,))[0:num_detections]
    instance_masks = np.squeeze(detection_masks, axis=(0,))[0:num_detections]
    
    # Apply Non-Maximum Suppression
    selected_indices = tf.image.non_max_suppression(
        detection_boxes,
        detection_scores,
        max_output_size=num_detections,
        iou_threshold=0.8
    )
    detection_boxes = tf.gather(detection_boxes, selected_indices).numpy()
    detection_scores = tf.gather(detection_scores, selected_indices).numpy()
    detection_classes = tf.gather(detection_classes, selected_indices).numpy()
    instance_masks = tf.gather(instance_masks, selected_indices).numpy()
    
    ymin, xmin, ymax, xmax = np.split(detection_boxes, 4, axis=-1)
    processed_boxes = np.concatenate([xmin, ymin, xmax - xmin, ymax - ymin], axis=-1)
    segmentations = coco_metric.generate_segmentation_from_masks(instance_masks, processed_boxes, height, width)
    
    if visualize:
        max_boxes_to_draw = 50   #@param {type:"integer"}
        min_score_thresh = 0.1    #@param {type:"slider", min:0, max:1, step:0.01}

        image_with_detections = visualization_utils.visualize_boxes_and_labels_on_image_array(
            np_image,
            detection_boxes,
            detection_classes,
            detection_scores,
            category_index,
            instance_masks=segmentations,
            use_normalized_coordinates=False,
            max_boxes_to_draw=max_boxes_to_draw,
            min_score_thresh=min_score_thresh)
        output_image_path = 'test_results.jpg'
        Image.fromarray(image_with_detections.astype(np.uint8)).save(output_image_path)
        display.display(display.Image(output_image_path, width=1024))

    return {"num_detections": len(selected_indices),
            "detection_boxes": detection_boxes,
            "detection_scores": detection_scores,
            "detection_classes": detection_classes,
            "detection_time": detection_time}





def benchmark_summary_mask_rcnn(results, data_source, data_time): 

    benchmark_results_mask_rcnn = pd.DataFrame(columns=['URL',
                                                   'TIME_SG',
                                                   'DETECTION_TIME',
                                                   'CAR_COUNT', 
                                                   'CAR_AVG_CONF',
                                                   'CAR_BOXES',
                                                   'MOTOR_CYCLE_COUNT',
                                                   'MC_AVG_CONF',
                                                   'MC_BOXES',
                                                   'BUS_COUNT',
                                                   'BUS_AVG_CONF',
                                                   'BUS_BOXES'
                                                   'MC_BOXES',
                                                   'TRUCK_COUNT',
                                                   'TRUCK_AVG_CONF',
                                                   'TRUCK_BOXES']
                                                    )


    # Reset counts and confidence counts
    car_count = 0
    car_conf_sum = 0
    car_boxes = []
    motor_cycle_count = 0
    motor_cycle_conf_sum = 0
    mc_boxes = []
    bus_count = 0
    bus_conf_sum = 0
    bus_boxes = []
    truck_count = 0
    truck_conf_sum = 0                                                                             
    truck_boxes = []

    for i, result in enumerate(results):
        
        obj_counter = Counter(result['detection_classes'])
        car_count = obj_counter[3]
        truck_count = obj_counter[8]
        motor_cycle_count = obj_counter[4]
        bus_count = obj_counter[6]

        for j, obj in enumerate(result['detection_classes']):
            # Car
            if obj == 3:
                car_conf_sum += result['detection_scores'][j] 
                car_boxes.append(result['detection_boxes'][j]) 
           
            # Motorcycle
            if obj == 4:
                motor_cycle_conf_sum += result['detection_scores'][j] 
                mc_boxes.append(result['detection_boxes'][j]) 
           
            # Truck
            if obj == 8:
                truck_conf_sum += result['detection_scores'][j] 
                truck_boxes.append(result['detection_boxes'][j]) 
           
            # Bus
            if obj == 6:
                bus_conf_sum += result['detection_scores'][j] 
                bus_boxes.append(result['detection_boxes'][j]) 
           
    

        # Evaluate average confidence for the different classes
        
        car_AVG_conf = car_conf_sum / car_count if car_count != 0 else 'N/A'

        bus_AVG_conf = bus_conf_sum / bus_count if bus_count != 0 else 'N/A'

        truck_AVG_conf = truck_conf_sum / truck_count if truck_count != 0 else 'N/A'

        MC_AVG_conf = motor_cycle_conf_sum / motor_cycle_count if motor_cycle_count != 0 else 'N/A'


        row = pd.DataFrame({'URL': [data_source[i]],
                            'TIME_SG':[data_time[i]],
                            'DETECTION_TIME': [result['detection_time']],
                            'CAR_COUNT': [car_count],
                            'CAR_AVG_CONF': [car_AVG_conf],
                            'CAR_BOXES': [car_boxes], 
                            'MOTOR_CYCLE_COUNT': [motor_cycle_count],
                            'MC_AVG_CONF': [MC_AVG_conf],
                            'MC_BOXES': [mc_boxes], 
                            'BUS_COUNT': [bus_count],
                            'BUS_AVG_CONF': [bus_AVG_conf],
                            'BUS_BOXES': [bus_boxes], 
                            'TRUCK_COUNT': [truck_count],
                            'TRUCK_AVG_CONF': [truck_AVG_conf],
                            'TRUCK_BOXES': [truck_boxes]
                            })


        benchmark_results_mask_rcnn = pd.concat([benchmark_results_mask_rcnn, row], ignore_index=True)

    return benchmark_results_mask_rcnn

    
