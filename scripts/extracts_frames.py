import cv2
import os

# Input directories
input_dirs = {
    "fire": "../data/firesense/fire",
    "smoke": "../data/firesense/smoke",
    "no_fire": "../data/firesense/no_fire"
}

# Output directories
output_base = "../data/frames"

os.makedirs(output_base, exist_ok=True)

for category, input_dir in input_dirs.items():
    output_dir = os.path.join(output_base, category)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing category: {category}")

    for video_name in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video_name)

        if not video_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        # Create sequence folder
        sequence_name = os.path.splitext(video_name)[0]
        sequence_folder = os.path.join(output_dir, sequence_name)
        os.makedirs(sequence_folder, exist_ok=True)

        # Read video
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frames (optional but recommended)
            frame = cv2.resize(frame, (224, 224))

            # Save frame
            frame_path = os.path.join(sequence_folder, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(frame_path, frame)

            frame_idx += 1

        cap.release()

        print(f"Extracted {frame_idx} frames from {video_name}")

print("\n✔ Frame extraction completed!")
