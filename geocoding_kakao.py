"""
카카오 로컬 API를 사용한 Geocoding 스크립트
- 아파트 실거래가 데이터의 도로명 주소를 위도/경도로 변환
"""

import pandas as pd
import requests
import time
import os
from dotenv import load_dotenv
from pathlib import Path

# .env 파일 로드
load_dotenv()

# 카카오 REST API 키
KAKAO_API_KEY = os.getenv('KAKAO_REST_API_KEY')

def geocode_address(address: str) -> dict:
    """
    카카오 로컬 API를 사용하여 주소를 좌표로 변환

    Args:
        address: 도로명 주소 (예: "서울특별시 서초구 양재천로17길 11")

    Returns:
        dict: {'lat': 위도, 'lng': 경도} 또는 실패 시 {'lat': None, 'lng': None}
    """
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": address}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        result = response.json()

        if result['documents']:
            doc = result['documents'][0]
            return {
                'lat': float(doc['y']),
                'lng': float(doc['x'])
            }
        else:
            return {'lat': None, 'lng': None}

    except Exception as e:
        print(f"Error geocoding '{address}': {e}")
        return {'lat': None, 'lng': None}


def process_apt_data(input_file: str, output_file: str = None, delay: float = 0.1):
    """
    아파트 실거래가 CSV 파일에 위도/경도 추가

    Args:
        input_file: 입력 CSV 파일 경로
        output_file: 출력 CSV 파일 경로 (기본값: 입력파일명_geocoded.csv)
        delay: API 요청 간 딜레이 (초)
    """
    if not KAKAO_API_KEY:
        print("Error: KAKAO_REST_API_KEY가 설정되지 않았습니다.")
        print(".env 파일에 다음을 추가하세요:")
        print("KAKAO_REST_API_KEY='your_api_key_here'")
        return None

    # CSV 읽기
    df = pd.read_csv(input_file, encoding='cp949')
    print(f"총 {len(df)}개 데이터 로드 완료")

    # 전체 주소 생성
    def make_full_address(row):
        sigungu = row['시군구']
        # '서울특별시 서초구 양재동' -> '서울특별시 서초구'
        parts = sigungu.rsplit(' ', 1)
        base_addr = parts[0] if len(parts) > 1 else sigungu
        return f"{base_addr} {row['도로명']}"

    df['full_address'] = df.apply(make_full_address, axis=1)

    # 중복 주소 제거하여 API 호출 최소화
    unique_addresses = df['full_address'].unique()
    print(f"고유 주소: {len(unique_addresses)}개")

    # Geocoding 수행
    address_coords = {}
    for i, addr in enumerate(unique_addresses):
        print(f"[{i+1}/{len(unique_addresses)}] {addr}", end=" ")
        coords = geocode_address(addr)
        address_coords[addr] = coords

        if coords['lat']:
            print(f"-> ({coords['lat']:.6f}, {coords['lng']:.6f})")
        else:
            print("-> 좌표 없음")

        time.sleep(delay)  # Rate limiting

    # 좌표 매핑
    df['lat'] = df['full_address'].map(lambda x: address_coords[x]['lat'])
    df['lng'] = df['full_address'].map(lambda x: address_coords[x]['lng'])

    # 결과 저장
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_geocoded.csv"

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n결과 저장: {output_file}")

    # 통계 출력
    success_count = df['lat'].notna().sum()
    print(f"Geocoding 성공: {success_count}/{len(df)} ({success_count/len(df)*100:.1f}%)")

    return df


if __name__ == "__main__":
    # 예시: 양재동 아파트 데이터 geocoding
    input_file = r"data\apt\서울특별시 서초구 양재동_2026.csv"

    result_df = process_apt_data(input_file)

    if result_df is not None:
        print("\n=== 결과 미리보기 ===")
        print(result_df[['단지명', 'full_address', 'lat', 'lng']].head(10))
