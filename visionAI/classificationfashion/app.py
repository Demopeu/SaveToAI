import numpy as np
import requests
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# YOLO 모델 로드
model = YOLO('./best.pt')

# 클래스 이름 설정
all_class_names = ['casual_look', 'classic_look', 'minimal_look', 'chic_look', 'sporty_look', 'street_look', 'person']

# 이미지 URL을 처리하고 예측 수행
def predict_image_from_url(image_url):
    try:
        # 이미지 다운로드
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')  # 모노 채널 설정 (필요 시)
        logging.info(f"이미지 다운로드 성공: {image_url}")

        # YOLO 모델로 예측 수행
        result = model.predict(img)
        class_probs = {class_name: 0 for class_name in all_class_names}

        # 감지된 객체 처리
        if result and result[0].boxes and len(result[0].boxes) > 0:
            for detection in result[0].boxes:
                conf = detection.conf.item()  # 신뢰도 점수
                class_idx = int(detection.cls.item())  # 클래스 인덱스
                class_name = all_class_names[class_idx]
                class_probs[class_name] = max(class_probs[class_name], conf)
            logging.info(f"객체 감지 성공: {image_url}")
            return class_probs
        else:
            logging.warning(f"객체 감지 실패: {image_url}")
            return None
    except Exception as e:
        logging.error(f"이미지 처리 오류 ({image_url}): {e}")
        return None

# 이미지에서 6D 벡터를 얻는 함수 (URL 사용)
def get_6d_vector(image_urls):
    if len(image_urls) != 2:
        return None

    main_image_url, sub_image_url = image_urls
    main_class_probs = predict_image_from_url(main_image_url)
    sub_class_probs = predict_image_from_url(sub_image_url)

    if main_class_probs is None or sub_class_probs is None:
        return None

    if main_class_probs['person'] == 0 or sub_class_probs['person'] == 0:
        return None

    # 서브 이미지의 'person' 클래스도 감지되었는지 확인
    main_filtered_probs = [main_class_probs[cls] for cls in all_class_names if cls != 'person']
    sub_filtered_probs = [sub_class_probs[cls] for cls in all_class_names if cls != 'person']

    # 70% 메인, 30% 서브 비율로 벡터 결합
    main_vector = np.array(main_filtered_probs)
    sub_vector = np.array(sub_filtered_probs)
    combined_vector = 0.7 * main_vector + 0.3 * sub_vector

    logging.info("두 이미지의 벡터 결합 완료")
    return combined_vector

# 단일 이미지의 6D 벡터를 얻는 함수
def get_single_6d_vector(image_url):
    class_probs = predict_image_from_url(image_url)
    if class_probs is None:
        return None

    # 'person' 클래스가 감지되었는지 확인
    if class_probs['person'] == 0:
        return None

    # 'person' 클래스를 제외한 클래스 확률 추출
    filtered_probs = [class_probs[cls] for cls in all_class_names if cls != 'person']
    vector = np.array(filtered_probs)

    logging.info("단일 이미지의 6D 벡터 생성 완료")
    return vector

# Flask 라우트 설정
@app.route('/process_images', methods=['POST'])
def process_images():
    try:
        # 클라이언트로부터 이미지 URL 리스트 받기
        data = request.json
        if 'image_urls' not in data or not isinstance(data['image_urls'], list):
            return jsonify({'error': '잘못된 입력입니다. image_urls는 리스트 형태로 제공되어야 합니다.'}), 400

        image_urls = data['image_urls']
        num_images = len(image_urls)

        if num_images == 1:
            # 이미지 URL이 1개인 경우
            image_url = image_urls[0]
            class_probs = predict_image_from_url(image_url)

            if class_probs is None:
                return jsonify({'error': '유효한 이미지를 찾지 못했습니다.'}), 400

            # 'person' 클래스를 제외한 가장 높은 클래스 찾기
            filtered_class_probs = {cls: prob for cls, prob in class_probs.items() if cls != 'person'}
            if not filtered_class_probs:
                return jsonify({'error': 'person 클래스를 제외한 다른 클래스가 감지되지 않았습니다.'}), 400

            top_class = max(filtered_class_probs.items(), key=lambda item: item[1])

            # 6D 벡터 생성
            vector = get_single_6d_vector(image_url)
            if vector is None:
                return jsonify({'error': '유효한 이미지를 찾지 못했거나 person이 감지되지 않았습니다.'}), 400

            # 클러스터 예측 서버로 POST 요청
            external_url = 'http://j11e104a.p.ssafy.io:5000/predict'
            try:
                response = requests.post(external_url, json={'point': vector.tolist()}, timeout=10)
                logging.info(f"외부 서버로 벡터 전송 완료: {external_url}")
            except requests.exceptions.Timeout:
                logging.error("외부 서버와의 연결 시간 초과")
                return jsonify({'error': '외부 서버와의 연결이 시간 초과되었습니다.'}), 504
            except requests.exceptions.RequestException as e:
                logging.error(f"외부 서버와의 통신 오류: {e}")
                return jsonify({'error': f'외부 서버와의 통신 오류: {e}'}), 502

            # 외부 서버에서 받은 응답 반환
            if response.status_code == 200:
                external_response = response.json()
                return jsonify({
                    'top_class': {'class': top_class[0], 'confidence': top_class[1]},
                    'response': external_response
                }), 200
            else:
                logging.error(f"외부 서버 응답 오류: {response.status_code}")
                return jsonify({'error': '외부 서버로부터 유효한 응답을 받지 못했습니다.'}), response.status_code

        elif num_images == 2:
            # 이미지 URL이 2개인 경우
            vector = get_6d_vector(image_urls)
            if vector is None:
                return jsonify({'error': '유효한 이미지를 찾지 못했거나 person이 감지되지 않았습니다.'}), 400

            # 클러스터 예측 서버로 POST 요청
            external_url = 'http://j11e104a.p.ssafy.io:5000/predict'
            try:
                response = requests.post(external_url, json={'point': vector.tolist()}, timeout=10)
                logging.info(f"외부 서버로 벡터 전송 완료: {external_url}")
            except requests.exceptions.Timeout:
                logging.error("외부 서버와의 연결 시간 초과")
                return jsonify({'error': '외부 서버와의 연결이 시간 초과되었습니다.'}), 504
            except requests.exceptions.RequestException as e:
                logging.error(f"외부 서버와의 통신 오류: {e}")
                return jsonify({'error': f'외부 서버와의 통신 오류: {e}'}), 502

            # 외부 서버에서 받은 응답 반환
            if response.status_code == 200:
                return jsonify({'response':response.json()}), 200
            else:
                logging.error(f"외부 서버 응답 오류: {response.status_code}")
                return jsonify({'error': '외부 서버로부터 유효한 응답을 받지 못했습니다.'}), response.status_code

        else:
            return jsonify({'error': '지원하지 않는 이미지 개수입니다. 1개 또는 2개의 이미지 URL을 제공해주세요.'}), 400

    except Exception as e:
        logging.error(f"서버 내부 오류: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
