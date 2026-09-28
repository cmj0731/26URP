# 2nd session: Starlink downlink 방식 비교

OFDM, AFDM, OTFS, DD-a-OFDM 비교를 위한 공통 시나리오를 관리한다.
현재 `scenario.json`에는 위성과 지상국의 기하 조건 및 1차 평가 지표 BER을 정의했다.
무선 주파수·대역폭, 전파 채널, 수신기 가정은 아직 결정하지 않았다.

## 시나리오 초안

- 지상국: 성균관대학교 자연과학캠퍼스, 북위 37.2934°, 동경 126.9747°,
  고도 0 m(참고 저장소의 좌표 가정).
- 위성: STARLINK-5285, NORAD 55296. 2026-07-27 11:53:54.484800 UTC의
  고정 OMM 자료를 SGP4로 전파한다.
- 검색: 2026-07-29 UTC 하루, 고도각 10° 이상인 통과 중 최고 고도각이 가장
  높은 통과를 선택한다. 날짜·고도각 기준은 검토를 위한 초안이다.
- 선택된 통과: 07:42:36.760–07:51:02.674 UTC, 최고 고도각 85.164878°
  (07:46:50.803 UTC). 최근접 시각은 07:46:50.954 UTC다.

자료 출처는 [`cmj0731/starlink-downlink-simulation`의 `955f0818` 커밋](https://github.com/cmj0731/starlink-downlink-simulation/tree/955f0818f66104cb8d264bbcccf39c117319b26c)이다.
지상국 좌표와 궤도 자료만 고정 입력으로 가져왔으며, 원본의 파형·주파수·채널
설정은 이번 비교에 채택하지 않았다.

## 평가 지표

1차 지표는 BER(잘못 수신한 정보 비트 수 / 전송한 정보 비트 수)로 확정했다.
비교 축, 부호화 여부, 반복 횟수와 보조 지표는 이후 결정한다.

## 통과 구간 다시 계산하기

Python 3.12 이상 환경에서 이 폴더를 현재 작업 위치로 두고 실행한다.

```powershell
python -m pip install -r requirements.txt
python select_pass.py
```

`select_pass.py`가 출력한 결과를 `scenario.json`의 `selected_pass`와 비교한다.
좌표 변환에는 WGS-84와 1차 TEME→ECEF 회전을 사용하고, 원본과 같이 UTC를
UT1의 근사값으로 사용하며 극운동은 생략한다.
