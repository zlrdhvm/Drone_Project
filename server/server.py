from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import cv2
import threading
import requests
import numpy as np
import time
from cachetools import TTLCache
from ultralytics import YOLO
import os
import itertools

# 설치 필요 : pip install pyrebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

from twilio.rest import Client

app = Flask(__name__)
CORS(app)

# YOLOv8 모델 로드 (학습된 weights 사용)
best_model_path = r'best.pt'
ESP32_STREAM_URL = 'http://192.168.31.176:81/stream'

video_frame = None
detected_objects = []
lock = threading.Lock()
model = YOLO(best_model_path)

# 주소 캐시 (TTLCache는 지정된 시간 동안만 캐시 유지)
address_cache = TTLCache(maxsize=100, ttl=3600)  # 1시간 동안 캐시 유지

latitude = 36.371691
longitude = 127.378694

# latitude = 36.371611
# longitude = 127.378645

location_count = 0
check_count = 0


pre_time = 0

account_sid = 'AC24e04b29a01ba69b69d7577bf672ade5'
auth_token = '94baaabb7a7be99ec3e106023ded3db4'
client = Client(account_sid, auth_token)


def capture_frames():
    global video_frame, detected_objects, pre_lati, pre_longi, check_count, location_count, pre_time
    while True:
        try:
            check = False   # 한 그리드에 객체 일정량 이상인지 파악용

            stream = requests.get(ESP32_STREAM_URL, stream=True, timeout=10)
            print("Connected to stream")
            byte_stream = b''
            for chunk in stream.iter_content(chunk_size=512):
                byte_stream += chunk
                a = byte_stream.find(b'\xff\xd8')
                b = byte_stream.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = byte_stream[a:b+2]
                    byte_stream = byte_stream[b+2:]
                    if jpg:
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            # YOLOv8 객체 인식 수행
                            results = model(frame)
                            detected_objects = []
                            if results and len(results) > 0:
                                result = results[0]
                                if result.boxes is not None and len(result.boxes) > 0:
                                    grid_size = (3, 4)  # 그리드 사이즈 (3 x 4)
                                    grid_counts = [[0 for _ in range(grid_size[1])] for _ in range(grid_size[0])]
                                    frame_height, frame_width = frame.shape[:2]
                                    row_height = frame_height / grid_size[0]
                                    col_width = frame_width / grid_size[1]

                                    for box in result.boxes:
                                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                                        cls = box.cls
                                        score = box.conf
                                        detected_objects.append({
                                            'class': int(cls),
                                            'score': float(score),
                                            'box': [x1, y1, x2, y2]
                                        })

                                        if cls == 0:  # assuming class 0 is for people
                                            start_col = int(x1 / col_width)
                                            end_col = int(x2 / col_width)
                                            start_row = int(y1 / row_height)
                                            end_row = int(y2 / row_height)
                                            for col in range(start_col, end_col + 1):
                                                for row in range(start_row, end_row + 1):
                                                    grid_counts[row][col] += 1

                                    annotated_frame = result.plot()  # 인식된 객체가 그려진 프레임

                                    # 그리드에 사람이 4명 이상인 경우 빨간색으로 표시
                                    for row in range(grid_size[0]):
                                        for col in range(grid_size[1]):
                                            #if grid_counts[row][col] >= 4:
                                            if grid_counts[row][col] >= 1:
                                                check = True

                                                start_x = int(col * col_width)
                                                start_y = int(row * row_height)
                                                end_x = int((col + 1) * col_width)
                                                end_y = int((row + 1) * row_height)
                                                cv2.rectangle(annotated_frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
                                else:
                                    annotated_frame = frame

                                if check:
                                    cur_time = time.time()
                                    # Firestore에 데이터 저장
                                    cred = credentials.Certificate('firebase_droneproject.json')
                                    try:
                                        firebase_admin.initialize_app(cred)
                                        print("Firebase app initialized successfully")
                                    except ValueError:
                                        # Firebase 앱이 이미 초기화된 경우
                                        print("Firebase app already initialized")
                                    db = firestore.client()
                                    ref = db.collection("locations")
                                    query = ref.where('latitude', '==', latitude).where('longitude',
                                                                                        '==', longitude)
                                    docs_al = query.stream()

                                    doc_list_al = []
                                    doc_list = []
                                    for doc in docs_al:
                                        doc_list_al.append(doc.id)

                                    # if doc_list_al and cur_time - pre_time >= 300:  # 일치하는 좌표가 존재한다면 5분단위로 저장
                                    if doc_list_al and cur_time - pre_time >= 5:  # 일치하는 좌표가 존재한다면 5분단위로 저장
                                        pre_time = cur_time
                                        # Firestore 문서 데이터를 리스트로 변환
                                        doc_id = doc_list_al[0]  # location0

                                        doc_ref = db.collection("locations").document(doc_id)
                                        new_count = doc_ref.get().to_dict()['count'] + 1

                                        data_set = {"latitude": latitude, "count": new_count, "longitude": longitude}

                                        if new_count == 2:
                                            api_key = 'AIzaSyA432B9qNSn19lPPXy83RsRIXe4oSO2pD8'  # Google API 키
                                            address = get_address_from_lat_lon_google(latitude, longitude, api_key)
                                            message = client.messages.create(
                                                to="+821093214328",
                                                from_="+12075604274",
                                                body=f"인파 밀집 위험 수준 10분 이상 지속되었습니다.\n위치: {address}")
                                            print("메시지 전송 완료")

                                    elif not doc_list_al:
                                        docs = ref.stream()
                                        for doc2 in docs:
                                            doc_list.append(doc2)
                                        doc_id = f"location{len(doc_list)}"
                                        data_set = {"latitude": latitude, "count": 1, "longitude": longitude}

                                    else:
                                        continue

                                    doc_ref = db.collection('locations').document(doc_id)
                                    doc_ref.set(data_set)
                                with lock:
                                    video_frame = annotated_frame
                            else:
                                annotated_frame = frame

                        else:
                            print("Error: Frame is None")
                    else:
                        print("Error: Empty JPEG buffer")

        except Exception as e:
            print(f"Error fetching stream: {e}")
            time.sleep(10)  # 스트림 재시도를 위한 대기 시간 증가



def get_address_from_lat_lon_google(lat, lon, api_key, retries=3):
    cache_key = (lat, lon)
    if cache_key in address_cache:
        return address_cache[cache_key]

    url = f'https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={api_key}&language=ko'
    print(f"Request URL: {url}")  # 요청 URL 출력

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            print(f"Response status code: {response.status_code}")  # 응답 상태 코드 출력
            if response.status_code == 200:
                data = response.json()
                print(f"Response data: {data}")  # 응답 데이터 출력
                if data['status'] == 'OK':
                    address = data['results'][0]['formatted_address']
                    address_cache[cache_key] = address
                    return address
                else:
                    print(f"Geocoding API Error: {data['status']} - {data['error_message'] if 'error_message' in data else 'No error message'}")
                    return "No address found"
            else:
                print(f"Attempt {attempt+1}: Error {response.status_code}. Retrying...")
                time.sleep(2 ** attempt)  # 지수 백오프
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1}: RequestException {e}. Retrying...")
            time.sleep(2 ** attempt)
    return "Failed to get address after retries"

@app.route('/')
def index():
    api_key = 'AIzaSyA432B9qNSn19lPPXy83RsRIXe4oSO2pD8'  # Google API 키
    address = get_address_from_lat_lon_google(latitude, longitude, api_key)
    print(address)
    return render_template('client.html', latitude=latitude, longitude=longitude, address=address)

@app.route('/video_feed')
def video_feed():
    def generate():
        global video_frame
        while True:
            with lock:
                if video_frame is None:
                    time.sleep(0.1)  # 프레임이 없는 경우 대기
                    continue
                ret, jpeg = cv2.imencode('.jpg', video_frame)
                if not ret:
                    time.sleep(0.1)  # 인코딩 실패 시 대기
                    continue
                frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)  # 프레임 사이의 짧은 대기 시간을 추가

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detected_objects')
def detected_objects_route():
    global detected_objects
    with lock:
        return jsonify(detected_objects)

@app.route('/detected_faces')
def detected_faces_route():
    global detected_objects
    faces = [{"x": obj["box"][0], "y": obj["box"][1], "w": obj["box"][2] - obj["box"][0], "h": obj["box"][3] - obj["box"][1]} for obj in detected_objects if obj["class"] == 0]  # assuming class 0 is for faces
    with lock:
        return jsonify(faces)

@app.route('/zoomed_frame')
def zoomed_frame():
    global video_frame
    x = int(request.args.get('x'))
    y = int(request.args.get('y'))
    width = int(request.args.get('width'))
    height = int(request.args.get('height'))

    with lock:
        if video_frame is None:
            return jsonify({"error": "No frame available"}), 404
        cropped_frame = video_frame[y:y + height, x:x + width]
        ret, jpeg = cv2.imencode('.jpg', cropped_frame)
        if not ret:
            return jsonify({"error": "Error encoding frame"}), 500
        return Response(jpeg.tobytes(), mimetype='image/jpeg')

# if __name__ == '__main__':
#     thread = threading.Thread(target=capture_frames)
#     thread.daemon = True
#     thread.start()
#     app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    thread = threading.Thread(target=capture_frames)
    thread.daemon = True
    thread.start()
    app.run(host='0.0.0.0', port=5000, debug=False)  # debug=False로 설정
