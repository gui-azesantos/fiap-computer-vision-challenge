import cv2
from deepface import DeepFace
import pandas as pd
from tqdm import tqdm


def analyze_video(input_path, output_path, excel_path):
    video = cv2.VideoCapture(input_path)
    if not video.isOpened():
        print("Error: Failed to open the video.")
        return

    # Video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    results = []
    emotion_counts = {}

    print(f"Video info: {total_frames} frames | {fps:.2f} FPS | Resolution: {width}x{height}")

    anomalies = 0

    for frame_number in tqdm(range(total_frames), desc="Analyzing frames"):
        ret, frame = video.read()
        if not ret:
            break

        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(analysis, list):
                analysis = analysis[0]

            # Extract data
            region = analysis.get("region", {})
            dominant_emotion = analysis.get("dominant_emotion", "unknown")
            emotions = analysis.get("emotion", {})

            x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{dominant_emotion} ({emotions.get(dominant_emotion, 0):.2f})"
            cv2.putText(frame, label, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Update emotion count
            if dominant_emotion != "unknown":
                emotion_counts[dominant_emotion] = emotion_counts.get(dominant_emotion, 0) + 1

            # Append result
            results.append({
                "frame": frame_number,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "dominant_emotion": dominant_emotion,
                **emotions
            })

        except Exception as e:
            anomalies += 1
            # Append minimal result
            results.append({
                "frame": frame_number,
                "error": str(e)
            })
            continue

        output_video.write(frame)

    # Cleanup
    video.release()
    output_video.release()

    # Save results to Excel
    df = pd.DataFrame(results)

    # Create summary
    summary_data = {
        "Total Frames Analyzed": [total_frames],
        "Number of Anomalies": [anomalies]
    }

    summary_df = pd.DataFrame(summary_data)

    # Emotion distribution
    emotion_dist = pd.DataFrame([
        {"Emotion": emotion, "Count": count, "Percentage": (count / (total_frames - anomalies)) * 100}
        for emotion, count in emotion_counts.items()
    ])

    # Write to Excel with multiple sheets
    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, sheet_name="Frame Analysis", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        emotion_dist.to_excel(writer, sheet_name="Emotion Distribution", index=False)

    print(f"Analysis saved to: {excel_path}")
    print(f"Annotated video saved to: {output_path}")


if __name__ == "__main__":
    input_video_path = "input.mp4"
    annotated_video_path = "output_analysis.mp4"
    excel_output_path = "video_analysis.xlsx"

    analyze_video(input_video_path, annotated_video_path, excel_output_path)
