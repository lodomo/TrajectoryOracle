import numpy as np
import matplotlib.pyplot as plt

class TrajectoryOracleRLAgent:
    def __init__(self, action_space, max_frames, prediction_frames=20):
        self.action_space = action_space
        self.max_frames = max_frames  # The maximum number of frames in the video
        self.prediction_frames = prediction_frames  # User-defined frames into the future to predict
        self.q_table = np.zeros((self.max_frames, self.action_space))  # 2D Q-table with frames as states
        self.epsilon = 0.1  # Exploration rate
        self.learning_rate = 0.01
        self.discount_factor = 0.99

        # To store predicted and actual positions for accuracy calculation
        self.predicted_positions = []
        self.actual_positions = []

    def choose_action(self, frame_number):
        # Choose action based on the frame number and a user-defined number of future frames
        if frame_number + self.prediction_frames < self.max_frames:
            state_index = frame_number + self.prediction_frames  # Predict for a future frame
        else:
            state_index = self.max_frames - 1  # Stay within the frame range

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)  # Explore
        else:
            return np.argmax(self.q_table[state_index])  # Exploit

    def update_q_value(self, frame_number, action, reward, next_frame_number):
        state_index = frame_number
        next_state_index = next_frame_number

        if next_state_index >= self.q_table.shape[0]:
            # If the next_state_index is out of bounds, handle it by not updating
            next_state_index = self.q_table.shape[0] - 1  # Set to the last valid index

        best_next_action = np.argmax(self.q_table[next_state_index])
        td_target = reward + self.discount_factor * self.q_table[next_state_index, best_next_action]
        td_error = td_target - self.q_table[state_index, action]

        self.q_table[state_index, action] += self.learning_rate * td_error


    def store_positions(self, predicted, actual):
        """ Store predicted and actual positions for accuracy calculation. """
        self.predicted_positions.append(predicted)
        self.actual_positions.append(actual)

    def calculate_accuracy(self):
        """ Calculate accuracy based on stored predicted and actual positions. """
        distances = np.linalg.norm(np.array(self.predicted_positions) - np.array(self.actual_positions), axis=1)
        max_distance = np.max(distances) if np.max(distances) != 0 else 1  # Normalize to avoid division by zero
        normalized_distances = distances / max_distance
        accuracy = 1 - normalized_distances  # Higher accuracy for lower distances

        # Introduce a sensitivity threshold to magnify small errors
        sensitivity_threshold = 0.01
        accuracy = np.clip(accuracy, sensitivity_threshold, 1)

        return accuracy

    def plot_predicted_vs_actual(self, output_path=None):
        """ Plot predicted vs. actual positions over time. """
        predicted_positions = np.array(self.predicted_positions)
        actual_positions = np.array(self.actual_positions)
        frame_range = np.arange(len(predicted_positions))

        plt.figure(figsize=(12, 6))
        plt.plot(frame_range, predicted_positions[:, 0], 'b-', label='Predicted X')
        plt.plot(frame_range, actual_positions[:, 0], 'r--', label='Actual X')
        plt.plot(frame_range, predicted_positions[:, 1], 'g-', label='Predicted Y')
        plt.plot(frame_range, actual_positions[:, 1], 'y--', label='Actual Y')

        plt.title("Predicted vs. Actual Positions Over Time", fontsize=16, weight='bold')
        plt.xlabel("Frame Number", fontsize=14, weight='bold')
        plt.ylabel("Position", fontsize=14, weight='bold')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)  # Save the plot as an image file

        plt.show()  # Ensure the plot window opens

    def predict_position_based_on_past(self, past_positions):
        """ Predict future position based on past positions using a simple linear extrapolation. """
        if len(past_positions) < 2:
            # Not enough data to predict, return the last known position
            return past_positions[-1]
        
        # Linear extrapolation
        x_positions = np.array([pos[0] for pos in past_positions])
        y_positions = np.array([pos[1] for pos in past_positions])

        x_trend = np.polyfit(range(len(x_positions)), x_positions, 1)
        y_trend = np.polyfit(range(len(y_positions)), y_positions, 1)

        # Predict the next position based on the linear trend
        predicted_x = np.polyval(x_trend, len(past_positions))
        predicted_y = np.polyval(y_trend, len(past_positions))

        return int(predicted_x), int(predicted_y)