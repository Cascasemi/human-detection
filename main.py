import cv2
import mediapipe as mp
import math
import winsound

# Initialize MediaPipe pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

# Initialize video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # List to store detected persons' landmarks
        persons = []

        # Loop through the detected persons
        for landmark in results.pose_landmarks.landmark:
            if landmark.visibility > 0.5:  # Visibility threshold to filter landmarks
                persons.append(landmark)

        # Calculate distance between detected persons
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                distance = calculate_distance(persons[i], persons[j])
                if distance < 0.5:  # Distance threshold (tune according to needs)
                    winsound.Beep(1000, 200)  # Beep when persons are close

        # Draw landmarks and connections on the frame
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame with detected persons
    cv2.imshow('Human Detection with Distance', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()