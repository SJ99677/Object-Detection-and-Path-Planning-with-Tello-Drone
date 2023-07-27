"""
Written by Sam Johnson
"""

import cv2
from djitellopy import tello
import cvzone
import networkx as nx
import matplotlib.pyplot as plt
import time

accuracy = 0.50  # accuracy % (.5 default)
nmAccuracy = 0.2  # removes duplicates (.2 default)

classObj = []
# List of potential objects that could be recognized
fileList = 'ss.names'
with open(fileList, 'rt') as f:
    classObj = f.read().split('\n')

print(classObj)
weightsPath = "frozen_inference_graph.pb"
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

pic = cv2.dnn_DetectionModel(weightsPath, configPath)
pic.setInputSize(320, 320)
pic.setInputScale(1.0 / 127.5)
pic.setInputMean((127.5, 127.5, 127.5))
pic.setInputSwapRB(True)

# defines graph
graph = nx.Graph()

# coordinates may be adjusted to change grid pattern. Default positions follow a triangle pattern.
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
path = ["Location A", "Location B", "Location C", "Location D", "Location E"]

drone = tello.Tello()
drone.connect()
print(drone.get_battery())
drone.streamoff()
drone.streamon()

drone.takeoff()
# drone moves up, 105cm to have higher view of surroundings
drone.go_xyz_speed(0, 0, 105, 30)

# Loop to draw rectangle around objects
while True:
    feed = drone.get_frame_read().frame
    # To remove duplicates / declare accuracy
    objIds, flat, boundB = pic.detect(feed, confThreshold=accuracy, nmsThreshold=nmAccuracy)
    try:
        for objId, flat, rectangle in zip(objIds.flatten(), flat.flatten(), boundB):
            objName = classObj[objId - 1].lower()
            distance = drone.get_distance_tof()

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
                    #change the multiplying variable to increase distance (50cm default)
                    int(relative_distance[0] * 50),
                    int(relative_distance[1] * 50)
                )
                while objName != "stop sign" or distance >= 130:
                    # Send commands to the Tello to navigate to the current location
                    drone.send_rc_control(relative_distance_cm[0], relative_distance_cm[1], 0, 0)
                    print("about to sleep")
                    drone.sleep(2)
                    print("ending sleep")
                else:
                    print("check2")
                    drone.send_rc_control(0, 0, 0, 0)
                    x, y, w, h = rectangle  # extracts the coordinates of the bounding box around the image
                    cx, cy = x + w // 2, y + h // 2  # calculates center point by adding 1/2 width to x and 1/2 height to y
                    feed_height, feed_width, _ = feed.shape  # returns height, width of image
                    error_x = cx - feed_width // 2  # caculate distance object is off-center horizontally
                    error_y = cy - feed_height // 2  # caculate distance object is off-center vertically
                    drone.send_rc_control(-int(error_y * 0.1), int(error_x * 0.1), 0, 0)  # adjusts drone using error values multiplied by a scaling factor (default .1)
                    print(distance)
                    print("Distance is less than 5 feet. Drone is landing")
                    drone.land()
                    drone.end()
            cvzone.cornerRect(feed, rectangle)
            cv2.putText(feed, f'{classObj[objId - 1].upper()} {round(flat * 100, 2)}',
                        (rectangle[0] + 10, rectangle[1] + 30), cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 255, 0), 2)
    except:
        pass

    drone.send_rc_control(0, 0, 0, 0)

    cv2.imshow("Image", feed)
    cv2.waitKey(1)
