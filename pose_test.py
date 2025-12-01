import os
import cv2
import mediapipe as mp
import shutil

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# 입력/출력 폴더
input_folder = "dataset/ir/cam1/0001"
output_folder = "test/rgb/cam1/0001"
os.makedirs(output_folder, exist_ok=True)

def get_body_orientation(landmarks):
    """
    landmarks: pose_landmarks.landmark
    반환값: front / side / back / unknown
    """
    try:
        # 필요한 landmark 가져오기
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

        # 벡터 차이
        shoulder_diff_x = abs(right_shoulder.x - left_shoulder.x)
        hip_diff_x = abs(right_hip.x - left_hip.x)
        nose_visible = nose.visibility > 0.5

        # 판단 로직
        if not nose_visible and shoulder_diff_x < 0.05 and hip_diff_x < 0.05:
            return "back"
        elif nose_visible and shoulder_diff_x < 0.2:
            return "side"
        elif nose_visible:
            return "front"
        else:
            return "unknown"

    except Exception as e:
        return "unknown"

# 이미지 순회
for fname in os.listdir(input_folder):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_folder, fname)
    image = cv2.imread(img_path)
    if image is None:
        continue
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)
    if results.pose_landmarks:
        orientation = get_body_orientation(results.pose_landmarks.landmark)
    else:
        orientation = "unknown"

    # 파일 이름에 라벨 추가
    name, ext = os.path.splitext(fname)
    new_name = f"{name}_{orientation}{ext}"
    new_path = os.path.join(output_folder, new_name)
    shutil.copy(img_path, new_path)

    print(f"{fname} -> {new_name}")

pose.close()
print("Done!")



