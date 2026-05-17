
def _composite_to_original(video_path: str, original_image_path: str) -> str:
    # Reads the face-crop bbox saved by facecrop.py and blends each
    # generated 512x512 frame back into the corresponding region of the
    # full-resolution original image.
    #
    # Pipeline:
    #   1. Load crop bbox from /tmp/_talkdrive_crop_bbox.json
    #   2. Skip if original is ~square (compositing adds no value)
    #   3. Per frame: resize → feather-blend → write to canvas
    #   4. Re-mux audio from generated clip into composite video
    import cv2, json, numpy as np, subprocess

    BBOX_FILE = "/tmp/_talkdrive_crop_bbox.json"

    if not os.path.exists(BBOX_FILE):
        logger.info("Crop bbox file absent — face_crop likely disabled, skipping compositor")
        return video_path

    with open(BBOX_FILE, "r") as bf:
        crop_data = json.load(bf)

    crop_x1, crop_y1, crop_x2, crop_y2 = crop_data["bbox"]
    orig_w = crop_data["img_w"]
    orig_h = crop_data["img_h"]
    os.remove(BBOX_FILE)

    # Skip compositing for near-square originals
    long_side  = max(orig_w, orig_h)
    short_side = max(min(orig_w, orig_h), 1)
    if (long_side / short_side) < 1.15:
        logger.info(f"Original {orig_w}x{orig_h} is ~square — skipping compositor")
        return video_path

    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    logger.info(
        f"COMPOSITOR: bbox ({crop_x1},{crop_y1})->({crop_x2},{crop_y2}) "
        f"crop={crop_w}x{crop_h} canvas={orig_w}x{orig_h}"
    )

    background = cv2.imread(original_image_path)
    if background is None:
        logger.warning(f"Could not read original image: {original_image_path}")
        return video_path

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    raw_out_path = video_path.replace(".mp4", "_comp_raw.mp4")

    writer = cv2.VideoWriter(
        raw_out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (orig_w, orig_h)
    )

    # Feather-blend alpha mask for natural edge transitions
    alpha = np.ones((crop_h, crop_w), dtype=np.float32)
    feather_px = max(min(crop_w, crop_h) // 12, 6)
    for px in range(feather_px):
        w = px / feather_px
        alpha[px, :]     *= w
        alpha[-(px+1), :] *= w
        alpha[:, px]     *= w
        alpha[:, -(px+1)] *= w
    alpha_3ch = alpha[:, :, np.newaxis]

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        resized = cv2.resize(frame, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)
        canvas  = background.copy()
        roi     = canvas[crop_y1:crop_y2, crop_x1:crop_x2].astype(np.float32)
        blended = (resized.astype(np.float32) * alpha_3ch + roi * (1.0 - alpha_3ch)).astype(np.uint8)
        canvas[crop_y1:crop_y2, crop_x1:crop_x2] = blended
        writer.write(canvas)
        frame_count += 1

    cap.release()
    writer.release()

    # Re-mux: take video stream from composite, audio stream from original generated clip
    final_out_path = raw_out_path.replace(".mp4", "_final.mp4")
    subprocess.run(
        ["ffmpeg", "-y",
         "-i", raw_out_path,
         "-i", video_path,
         "-c:v", "libx264", "-c:a", "aac",
         "-map", "0:v", "-map", "1:a",
         "-shortest", final_out_path],
        capture_output=True
    )

    if os.path.exists(final_out_path) and os.path.getsize(final_out_path) > 0:
        os.replace(final_out_path, video_path)
        if os.path.exists(raw_out_path):
            os.remove(raw_out_path)
        logger.info(f"Compositor: {frame_count} frames -> {orig_w}x{orig_h}")
        return video_path

    if os.path.exists(raw_out_path):
        os.replace(raw_out_path, video_path)
    return video_path
