from ultralytics import YOLO
import cv2
import json
import matplotlib.pyplot as plt

#loading model
model = YOLO('yolov10n.pt')

# loading video
video_path = './input_video.mp4'
cap = cv2.VideoCapture(video_path)

# Definitions
frame_skip =5                      # integer defining number of frames to skip while doing calculations
frame_index=0                      # integer defining the number of the frame every loop
scale_percent = 50                 # percetage the video will be resized to
all_detections = []                # stores detected labels, box-coordinates, conf. scores to pass into JSON
diversity_per_frame = []           # stores number of objects detected per frame
freq = {}                          # stores number of times an object is detected per frame

ret = True
while ret:  # reads every frame
    ret, frame = cap.read()

    if ret:
        results = model.track(frame, persist=True)  # detects and tracks objects
        frame_ = results[0].plot()  # plots the detected objects on the image

        height, width, _ = frame_.shape  # resizing the image to fit the screen
        new_width = int(width * scale_percent / 100)
        new_height = int(height * scale_percent / 100)
        dim = (new_width, new_height)
        resized = cv2.resize(frame_, dim, interpolation=cv2.INTER_AREA)

        cv2.imshow('frame', resized)  # outputs the image in a window named 'frame'
        if cv2.waitKey(25) & 0xFF == ord('q'):  # and waits for 25 frames = 1 second
            break

        if frame_index % frame_skip == 0:  # executes every skip of frames = 5
            frame_detections = []  # stores frame information
            classes_seen = []  # stores number of classes seen in the frame

            for r in results:  # loop stores objects detected in the frame
                for box in r.boxes:
                    cls_id = int(box.cls)
                    label = model.model.names[cls_id]
                    conf = float(box.conf)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if (conf > 0.5):  # if confidence is greater than 50%, stores frame info.
                        frame_detections.append({
                            "label": label,
                            "confidence": round(conf, 4),
                            "bbox": [x1, y1, x2, y2]
                        })
                        classes_seen.append(label)

            frame_json = {  # updates frame detections for each frame
                "frame_index": frame_index,
                "detections": frame_detections
            }
            all_detections.append(frame_json)
            diversity_per_frame.append((frame_index, len(classes_seen)))

            path = "./output_images/frame_" + str(frame_index) + ".jpg"
            path = str(path)
            cv2.imwrite(path, resized)  # stores annonated frame

            for item in classes_seen:  # updates number of objects per class
                if item in freq:
                    freq[item] += 1
                else:
                    freq[item] = 1

        frame_index += 1  # updates to next frame

with open("detections.json", "w") as f: # saving detections in JSON
    json.dump(all_detections, f, indent=2)


# Determing and printing the frame with maximum diversity of objects
most_diverse_frame = max(diversity_per_frame, key=lambda x: x[1])

print("Frame with max class diversity:", most_diverse_frame[0],
      "\nClasses in this frame:",most_diverse_frame[1])


# Creating video file for every fifth frame, upto the last frame
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('./output_video/video.avi', fourcc, 30, (new_width, new_height))

for j in range(0,frame_index):
    if (j-1) % frame_skip == 0:
        path = "./output_images/frame_" + str(j-1) + ".jpg"
        path = str(path)
        if path is None:
            break
        img = cv2.imread(path)
        video.write(img)

# Releasing video windows
video.release()
cap.release()
cv2.destroyAllWindows()


# Assigning detected objects on x-axis,
# with overall frequency on y-axis of one plot and per-frame frequency on second plot
x= freq.keys()
y= freq.values()
y_list = list(y)                   # converting to list from dictionary
y_perframe = [f / len(all_detections) for f in y_list]


# Plotting the two graphs,
# one with overall frequency of objects
# and the other with frequency per frame of objects
# Creating 1 row, 2 column subplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plotting on the first subplot
ax1.bar(x, y, color='skyblue')
ax1.set_title('Plot 1: Frequency of Objects')
ax1.set_xlabel('Class')
ax1.set_ylabel('Count')

# Plotting on the second subplot
ax2.bar(x, y_perframe, color='salmon')
ax2.set_title('Plot 2: Frequency of Objects per frame')
ax2.set_xlabel('Class')
ax2.set_ylabel('Count')

plt.tight_layout()
plt.show()