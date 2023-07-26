
import cv2
from djitellopy import tello
import cvzone
import networkx as nx
import matplotlib.pyplot as plt
import time

thres = 0.50
nmsThres = 0.2

# Code for webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

classNames = []
classFile = 'ss.names' # Contains a totoal of 91 different objects which can be recognized by the code
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')


print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

graph = nx.Graph()
# """
# graph.add_node("Location A", pos=(0, 0))
# graph.add_node("Location B", pos=(1, 1))
# graph.add_node("Location C", pos=(2, 0))
# graph.add_node("Location D", pos=(1, -1))
# graph.add_node("Location E", pos=(0, 0))
#             """

graph.add_node("Location A", pos=(0, 0))
graph.add_node("Location B", pos=(-1, 0))
graph.add_node("Location C", pos=(0, 2))
graph.add_node("Location D", pos=(1, 0))
graph.add_node("Location E", pos=(0, 0))

graph.add_edge("Location A", "Location B")
graph.add_edge("Location B", "Location C")
graph.add_edge("Location C", "Location D")
graph.add_edge("Location D", "Location E")
graph.add_edge("Location E", "Location A")
nx.draw(graph, pos=nx.get_node_attributes(graph, "pos"), with_labels=True)
plt.show()

# Plan the path
# path = nx.dijkstra_path(graph, "Location A", "Location C")
path = ["Location A", "Location B", "Location C", "Location D", "Location E"]

me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamoff()
me.streamon()

me.takeoff()
#me.move_up(90)
me.go_xyz_speed(0, 0, 105, 30)


while True:
    # success, img = cap.read()
    img = me.get_frame_read().frame
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres) # To remove duplicates / declare accuracy
    try:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            objName = classNames[classId - 1].lower()
            distance = me.get_distance_tof()

            # Initialize the current position
            current_position = (0, 0)
            for location in path:
                # Get the coordinates of the current location
                target_position = graph.nodes[location]["pos"]

                # Calculate the relative distance from the current position to the target position
                relative_distance = (
                    target_position[0] - current_position[0],
                    target_position[1] - current_position[1]
                )
                # Update the current position
                current_position = target_position
                # Convert relative distance to centimeters
                relative_distance_cm = (
                    int(relative_distance[0] * 25),
                    int(relative_distance[1] * 25)
                )
                # Send commands to the Tello to navigate to the current location
                me.send_rc_control(relative_distance_cm[0], relative_distance_cm[1], 0, 0)
                print("about to sleep")
                time.sleep(5)
                print("ending sleep")
            while objName != "stop sign" or distance >= 130:
                print("check1")
                me.go_xyz_speed(60, 0, 0, 50)
                me.sleep(4)
            else:
                print("check2")
                me.send_rc_control(0, 0, 0, 0)
                #me.sleep(2)
                print("distance less than 200 cm, commencing landing")
                # Center the frame on the object
                # me.send_rc_control(0, 0, 0, 0)
                x, y, w, h = box  # extracts the coordinates of the bounding box around the image
                cx, cy = x + w // 2, y + h // 2  # calculates center point by adding 1/2 width to x and 1/2 height to y
                img_height, img_width, _ = img.shape  # returns height, width of image
                error_x = cx - img_width // 2  # caculate distance object is off-center horizontally
                error_y = cy - img_height // 2  # caculate distance object is off-center vertically
                me.send_rc_control(-int(error_y * 0.1), int(error_x * 0.1), 0, 0)  # adjusts drone using error values multiplied by a scaling factor (default .1)
                #me.sleep(2)
                print(distance)
                print("Distance is less than 5 feet. Drone is landing")
                me.land()
                me.end()
                #me.send_rc_control(0, 0, 0, 0)
                #print(distance)
                #me.land()
            cvzone.cornerRect(img, box)
            cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                        (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 255, 0), 2)
    except:
        pass

    me.send_rc_control(0, 0, 0, 0)

    cv2.imshow("Image", img)
    cv2.waitKey(1)



