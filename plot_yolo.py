import sys
import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from rl_agent import TrajectoryOracleRLAgent

class PlotYolo:
    def __init__(self, video_filepath, distance_threshold=50):
        self.yolo = YOLO("yolov8s.pt").to("cuda")
        self.video_filepath = video_filepath
        self.video_capture = cv2.VideoCapture(self.video_filepath)
        self.objects_last_frame = []
        self.objects = []
        self.frames_processed = 0
        self.__frame_size = None
        self.distance_threshold = distance_threshold
        self.colors_generated = 0
        self.used_colors = {}
        max_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.agent = TrajectoryOracleRLAgent(action_space=2, max_frames=max_frames, prediction_frames=20)
        self.set_frame_size()

    def __del__(self):
        # Zero out all data, and release the video capture,
        # and close all windows
        self.video_filepath = None
        self.objects_last_frame = None
        self.objects = None
        self.frames_processed = 0
        self.__frame_size = None
        self.distance_threshold = None
        self.video_capture.release()
        cv2.destroyAllWindows()

    @property
    def object_count(self):
        return len(self.objects)

    @property
    def frame_size(self):
        return self.__frame_size

    def set_frame_size(self):
        if self.__frame_size is None:
            ret, frame = self.video_capture.read()
            self.__frame_size = frame.shape[:2]
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def get_next_frame(self, confidence=0.6):
        try:
            frame, results = next(self.yield_next_frame_raw())
            if results is None:
                print("No results detected.")
                return None, frame

        except (TypeError, StopIteration):
            return None, None  # Ensure both objects and frame return None

        objects = []
        object_id = 0

        for result in results:
            classes_names = result.names
            for box in result.boxes:
                if box.conf[0] > confidence:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cls = int(box.cls[0])
                    class_name = classes_names[cls]

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # Put label and confidence score
                    label = f"{class_name} {box.conf[0]:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    id = f"{self.frames_processed}_{object_id}"
                    detected_object = DetectedObject(id, class_name, cls, box.conf[0], x1, y1, x2, y2)
                    objects.append(detected_object)
                    object_id += 1

                    # Debug print statement for detected object
                    print(f"Detected object: ID={detected_object.id}, Class={detected_object.class_name}, "
                        f"Confidence={detected_object.confidence:.2f}, "
                        f"Position=({detected_object.x1}, {detected_object.y1}, {detected_object.x2}, {detected_object.y2})")

        self.objects_last_frame = self.objects
        self.objects = objects
        self.correlate_objects()

        return self.objects, frame  # Return the frame along with objects


    def gen_unique_color(self):
        rgb = []
        for i in range(3):
            random = np.random.randint(0, 25) * 10
            rgb.append(random)

        if tuple(rgb) in self.used_colors:
            return self.gen_unique_color()
        else:
            self.used_colors[tuple(rgb)] = True

        self.colors_generated += 1
        if self.colors_generated > 1000:
            self.used_colors = {}
            self.colors_generated = 0

        return tuple(rgb)

    def yield_next_frame_raw(self):
        while True:
            ret, frame = self.video_capture.read()
            self.frames_processed += 1
            if not ret:
                return None, None
            results = self.yolo(frame, verbose=False)
            yield frame, results

    def save_frame(self, frame):
        cv2.imwrite(f"./test_output/frame_{self.frames_processed}.jpg", frame)

    def correlate_objects(self):
        for obj in self.objects:
            closest_match = float('inf')
            for obj_last in self.objects_last_frame:
                if obj_last.class_id != obj.class_id:
                    continue

                distance = ((obj.x - obj_last.x) ** 2 + (obj.y - obj_last.y) ** 2) ** 0.5
                if distance < closest_match:
                    closest_match = distance
                    if distance < self.distance_threshold:
                        obj.id = obj_last.id
                        obj.last_position = obj_last.position
                        break

    def spinning_bar(self, stop_event):
        spinner = ["|", "/", "-", "\\"]
        idx = 0
        while not stop_event.is_set():
            sys.stdout.write(f"\r{spinner[idx]}")
            sys.stdout.flush()
            time.sleep(0.1)
            idx = (idx + 1) % len(spinner)
        sys.stdout.write("\rDone!\n")

    def plot_video(self, spinner=False):
            if spinner:
                stop_event = threading.Event()
                spinner_thread = threading.Thread(target=self.spinning_bar, args=(stop_event,))
                spinner_thread.start()

            colors = {}

            while True:
                objects, frame = self.get_next_frame()
                if objects is None or frame is None:
                    break

                for obj in objects:
                    for interval in self.agent.time_intervals:
                        action = self.agent.choose_action(obj.class_name, objects, interval)
                        if action == 0:
                            if obj.id not in colors:
                                colors[obj.id] = self.gen_unique_color()

                            # Draw circle and bounding box
                            cv2.circle(frame, obj.position, 5, colors[obj.id], -1)
                            cv2.rectangle(frame, (obj.x1, obj.y1), (obj.x2, obj.y2), (255, 0, 0), 2)
                            label = f"{obj.class_name} {obj.confidence:.2f} ({interval}s)"
                            cv2.putText(frame, label, (obj.x1, obj.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            reward = 10
                        elif action == 1:
                            print(f"Skipping frame {self.frames_processed} - waiting for more frames.")
                            reward = -5

                        next_objects, _ = self.get_next_frame()
                        self.agent.update_q_value(obj.class_name, objects, action, reward, next_objects, interval)

                output_path = f"./test_output/frame_{self.frames_processed}.jpg"
                cv2.imwrite(output_path, frame)  # Save the frame with bounding boxes
                print(f"Saved frame {self.frames_processed} to {output_path}")

            if spinner:
                stop_event.set()
                spinner_thread.join()

            print("Finished processing video.")

class DetectedObject:
    def __init__(self, id, class_name, class_id, confidence, x1, y1, x2, y2):
        self.id = id
        self.class_name = class_name
        self.class_id = class_id
        self.confidence = confidence
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x = (x1 + x2) // 2
        self.y = (y1 + y2) // 2
        self.w = x2 - x1
        self.h = y2 - y1
        self.last_position = None

    @property
    def position(self):
        return (self.x, self.y)

    def __str__(self):
        return f"ID: {self.id}: {self.class_name}@({self.x}, {self.y})"