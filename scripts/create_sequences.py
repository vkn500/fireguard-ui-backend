import os
import cv2
import shutil

# Input: extracted frame folders
input_base = "../data/frames"

# Output: temporal sequences
output_base = "../data/sequences"
os.makedirs(output_base, exist_ok=True)

# Number of frames per sequence
SEQ_LEN = 16

categories = ["fire", "smoke", "no_fire"]

for category in categories:
    input_dir = os.path.join(input_base, category)
    output_dir = os.path.join(output_base, category)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing category: {category}")

    for sequence_folder in os.listdir(input_dir):
        seq_path = os.path.join(input_dir, sequence_folder)
        if not os.path.isdir(seq_path):
            continue

        frames = sorted(os.listdir(seq_path))
        total_frames = len(frames)

        if total_frames < SEQ_LEN:
            continue

        clip_count = 0

        for i in range(0, total_frames - SEQ_LEN + 1, SEQ_LEN):
            clip_output = os.path.join(output_dir, f"{sequence_folder}_clip_{clip_count}")
            os.makedirs(clip_output, exist_ok=True)

            # Copy 16 frames
            for j in range(i, i + SEQ_LEN):
                src = os.path.join(seq_path, frames[j])
                dst = os.path.join(clip_output, f"{j - i}.jpg")
                shutil.copy(src, dst)

            clip_count += 1

        print(f"Created {clip_count} clips from {sequence_folder}")

print("\n✔ All temporal sequences created successfully!")
