import cloudinary
import cloudinary.uploader
import time
import os
import uuid
import cv2

cloudinary.config(
    cloud_name="dprwjya79",
    api_key="943616652546731",
    api_secret="khRZlG5lvjBiuvzJZZbmdIyf3OE"
)

def upload_media(video_path, image_frame, suspects, cam_id="CAM1"):
    # upload video
    video_result = cloudinary.uploader.upload_large(
        video_path,
        resource_type="video",
        folder="sri_recordings"
    )
    video_url = video_result["secure_url"]

    # save temporary snapshot
    img_name = f"snapshot_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(img_name, image_frame)

    image_result = cloudinary.uploader.upload(
        img_name,
        folder="sri_images"
    )
    image_url = image_result["secure_url"]
    os.remove(img_name)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # mongo compatible record
    record = {
        "suspects": list(suspects),
        "camera_id": cam_id,
        "timestamp": timestamp,
        "video_url": video_url,
        "image_url": image_url
    }

    return record
