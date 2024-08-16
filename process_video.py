import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from plot_yolo import PlotYolo
import pickle
from rl_agent import TrajectoryOracleRLAgent
import tkinter as tk
from tkinter import filedialog, messagebox

def process_video(video_filepath, output_dir='./output_frames', distance_threshold=50, prediction_frames=20):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    plot_yolo = PlotYolo(video_filepath, distance_threshold)

    # Get the video properties to write the output video
    width = int(plot_yolo.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(plot_yolo.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = plot_yolo.video_capture.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    output_video_path = os.path.join(output_dir, "output_with_trajectory.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    all_positions = []  # To store midpoints for trajectory map
    previous_midpoint = None  # To store the previous midpoint

    max_frames = int(plot_yolo.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    plot_yolo.agent = TrajectoryOracleRLAgent(action_space=2, max_frames=max_frames, prediction_frames=prediction_frames)  # Use the user-defined prediction_frames

    while True:
        objects, frame = plot_yolo.get_next_frame()
        if objects is None or frame is None:
            break

        previous_midpoint = None  # Reset for each frame

        for obj in objects:
            # Use the frame number as the state
            action = plot_yolo.agent.choose_action(frame_count)
            
            mid_x = (obj.x1 + obj.x2) // 2
            mid_y = (obj.y1 + obj.y2) // 2  
            all_positions.append((mid_x, mid_y))

            # Predict the future position
            predicted_frame = frame_count + plot_yolo.agent.prediction_frames
            if predicted_frame < max_frames:
                predicted_mid_x, predicted_mid_y = plot_yolo.agent.predict_position_based_on_past(all_positions[-plot_yolo.agent.prediction_frames:])
                plot_yolo.agent.store_positions(predicted=(predicted_mid_x, predicted_mid_y), actual=(mid_x, mid_y))

                # Draw the predicted midpoint on the current frame
                cv2.circle(frame, (predicted_mid_x, predicted_mid_y), 5, (0, 0, 255), -1)  # Red for predicted

            # Draw the current midpoint on the frame (actual position)
            cv2.circle(frame, (mid_x, mid_y), 5, (0, 255, 0), -1)  # Green for actual

            # Draw a line from the previous midpoint to the current one on the frame
            if previous_midpoint is not None:
                cv2.line(frame, previous_midpoint, (mid_x, mid_y), (0, 255, 0), 2)
            
            # Update the previous midpoint to the current one
            previous_midpoint = (mid_x, mid_y)

            # Simulate a reward for the action (this will vary depending on your logic)
            reward = 10 if action == 0 else -5

            # Update the Q-table with the action taken
            plot_yolo.agent.update_q_value(frame_count, action, reward, frame_count + 1)

        # Write the processed frame with trajectory to the output video
        out_video.write(frame)

        # Optionally, save individual frames with the trajectory overlay
        output_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        if frame_count % plot_yolo.agent.prediction_frames == 0:
            cv2.imwrite(output_path.replace('.jpg', '_prediction.jpg'), frame)
        else:
            cv2.imwrite(output_path, frame)
            print(f"Saved frame {frame_count} to {output_path}")

        frame_count += 1

    print(f"Finished processing video: {video_filepath}")
    print(f"Saved output video with trajectory to {output_video_path}")

    # Release the video writer object
    out_video.release()

    # Save RL data
    save_rl_data(plot_yolo.agent, output_dir)

    # Visualize Q-table
    visualize_q_table(plot_yolo.agent.q_table, output_path=os.path.join(output_dir, "q_table_heatmap.png"))

    # Plot predicted vs actual positions over time
    plot_yolo.agent.plot_predicted_vs_actual()

    # Generate trajectory map
    generate_trajectory_map(all_positions, output_dir, width, height)

    # Close video display window
    cv2.destroyAllWindows()


def save_rl_data(agent, output_dir):
    q_table_path = os.path.join(output_dir, "q_table.pkl")
    with open(q_table_path, 'wb') as f:
        pickle.dump(agent.q_table, f)
    print(f"Saved RL Q-table to {q_table_path}")

def visualize_q_table(q_table, group_size=10, output_path=None):
    if isinstance(q_table, np.ndarray):
        num_states, num_actions = q_table.shape
        # Trim the Q-table to the nearest multiple of group_size
        trimmed_size = (num_states // group_size) * group_size
        trimmed_q_table = q_table[:trimmed_size, :]

        # Group frames together by averaging over 'group_size' frames
        grouped_q_table = trimmed_q_table.reshape(-1, group_size, num_actions).mean(axis=1)

        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(
            grouped_q_table, 
            annot=False,  # Turn off annotations for a cleaner look
            fmt=".2f", 
            cmap="coolwarm",  # Colormap with strong contrast
            cbar=True, 
            linewidths=0.1,  # Lighter gridlines for less clutter
            linecolor='black',  # Subtle gridline color
            vmin=-1, vmax=1  # Adjust color range for better contrast
        )
        
        # Adding a title and more context to the heatmap
        plt.title("Q-Table Heatmap: Predict Now vs. Wait", fontsize=16, weight='bold')
        plt.xlabel("Action (0 = Predict Now, 1 = Wait)", fontsize=14, weight='bold')
        plt.ylabel(f"Grouped Frame Number (Averaged over {group_size} frames)", fontsize=14, weight='bold')
        plt.xticks(ticks=[0, 1], labels=["Predict Now", "Wait"], fontsize=12)
        plt.yticks(fontsize=12)

        # Adding annotations for the color bar
        colorbar = ax.collections[0].colorbar
        colorbar.set_label('Q-Value', fontsize=12, weight='bold')
        colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        colorbar.set_ticklabels(['Very Low (Unfavorable)', 'Low', 'Neutral', 'High', 'Very High (Favorable)'])

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)  # Save the heatmap as an image file

        plt.show()  # Ensure the heatmap window opens

def generate_trajectory_map(all_positions, output_dir, width, height):
    plt.figure(figsize=(10, 8))
    all_positions = np.array(all_positions)

    if len(all_positions) > 0:
        plt.scatter(all_positions[:, 0], all_positions[:, 1], c=np.arange(len(all_positions)), cmap='viridis', marker='o')
        plt.xlim([0, width])  # Keep the x-axis consistent with video width
        plt.ylim([height, 0])  # Invert Y axis to keep it consistent with video coordinates
        plt.gca().set_aspect('auto')  # Restore the original aspect ratio behavior
        plt.title("Trajectory Map of Detected Objects")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.colorbar(label="Frame Number")
        plt.savefig(os.path.join(output_dir, "trajectory_map.png"))
        plt.show()


def select_video():
    video_filepath = filedialog.askopenfilename(title="Select Video", filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
    if video_filepath:
        output_dir = os.path.join('./output_frames', os.path.splitext(os.path.basename(video_filepath))[0])
        distance_threshold = int(distance_threshold_entry.get())
        prediction_frames = int(prediction_frames_entry.get())
        process_video(video_filepath, output_dir, distance_threshold, prediction_frames)
    else:
        messagebox.showwarning("No File Selected", "Please select a video file.")

# GUI Setup
root = tk.Tk()
root.title("Trajectory Oracle")

tk.Label(root, text="Distance Threshold:").grid(row=0, column=0, padx=10, pady=10)
distance_threshold_entry = tk.Entry(root)
distance_threshold_entry.insert(0, "50")
distance_threshold_entry.grid(row=0, column=1, padx=10, pady=10)

tk.Label(root, text="Prediction Frames:").grid(row=1, column=0, padx=10, pady=10)
prediction_frames_entry = tk.Entry(root)
prediction_frames_entry.insert(0, "20")
prediction_frames_entry.grid(row=1, column=1, padx=10, pady=10)

select_video_button = tk.Button(root, text="Select Video", command=select_video)
select_video_button.grid(row=2, column=0, columnspan=2, pady=20)

root.mainloop()
