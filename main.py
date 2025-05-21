import cv2
from deepface import DeepFace
import pandas as pd
from tqdm import tqdm


def analyze_video(input_path, output_path, excel_path, confidence_threshold=20):
    video = cv2.VideoCapture(input_path)
    if not video.isOpened():
        print("Error: Failed to open the video.")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    results = []
    anom_count = 0
    emotion_counter = {}

    print(f"Video info: {total_frames} frames | {fps:.2f} FPS | Resolution: {width}x{height}")

    for frame_number in tqdm(range(total_frames), desc="Analyzing frames"):
        ret, frame = video.read()
        if not ret:
            break

        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(analysis, list):
                analysis = analysis[0]

            region = analysis.get("region", {})
            dominant_emotion = analysis.get("dominant_emotion", None)
            emotions = analysis.get("emotion", {})

            if dominant_emotion:
                confidence = emotions.get(dominant_emotion, 0)
            else:
                confidence = 0

            if not dominant_emotion or confidence < confidence_threshold:
                anom_count += 1
                results.append({"frame": frame_number, "anomaly": True})
                continue

            emotion_counter[dominant_emotion] = emotion_counter.get(dominant_emotion, 0) + 1

            x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{dominant_emotion} ({confidence:.2f})"
            cv2.putText(frame, label, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            results.append({
                "frame": frame_number,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "dominant_emotion": dominant_emotion,
                "confidence": confidence,
                "anomaly": False,
                **emotions
            })

        except Exception as e:
            anom_count += 1
            results.append({"frame": frame_number, "anomaly": True})
            continue

        output_video.write(frame)

    video.release()
    output_video.release()

    # DataFrame
    df = pd.DataFrame(results)

    # Summary statistics
    total_frames_analyzed = len(df)
    anom_percent = (anom_count / total_frames_analyzed) * 100

    # Emotion percentages
    total_detected = sum(emotion_counter.values())
    emotion_percentages = {k: (v / total_detected) * 100 for k, v in emotion_counter.items()}

    # Summary DataFrame
    summary_data = {
        "Total Frames": [total_frames_analyzed],
        "Anomalies": [anom_count],
        "Anomaly %": [f"{anom_percent:.2f}%"]
    }

    for emo, perc in emotion_percentages.items():
        summary_data[f"{emo} %"] = [f"{perc:.2f}%"]

    summary_df = pd.DataFrame(summary_data)

    # Excel writer
    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, sheet_name="Detailed Results", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Analysis saved to: {excel_path}")
    print(f"Annotated video saved to: {output_path}")


if __name__ == "__main__":
    input_video_path = "input.mp4"
    annotated_video_path = "output_analysis.mp4"
    excel_output_path = "video_analysis.xlsx"

    analyze_video(input_video_path, annotated_video_path, excel_output_path)
