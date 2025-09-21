# 실전! AI 추천 시스템 알고리즘 이해 및 구현


| 일정   | 시간 | 내용                                 |
|--------|------|--------------------------------------|
| 1일차  | 오전 | 환경 구성                             |
|        |      | MovieLens 데이터셋 소개               |
|        |      | 초간단 추천 로직 만들기               |
|        | 오후 | 추천 시스템 개론                      |
|        |      | 콘텐츠 기반 추천                      |
|        |      | 머신 러닝 I                           |
| 2일차  | 오전 | 머신 러닝 II                          |
|        |      | Matrix Factorization I - 1            |
|        |      | Matrix Factorization I - 2            |
|        | 오후 | Matrix Factorization II - 1           |
|        |      | Matrix Factorization II - 2           |
| 3일차  | 오전 | Matrix Factorization III - 1          |
|        |      | Matrix Factorization III - 2          |
|        | 오후 | Negative Sampling                     |

## 강사

멀티캠퍼스 강선구(sunku0316.kang@multicampus.com, sun9sun9@gmail.com)

## 기본 환경 구성

Windows 10 / Docker powered by WSL Ubuntu(24.04)

Window에서 가상 환경으로 Ubuntu를 구동 시켜서 진행합니다. 

순수한 Linux 환경이 사용이 어려우실 수 있어 WSL로 구성했습니다. 

동일 장비에서 테스트해본 결과 순수 Linux 환경과 WSL의 성능 차이는 약 5배 정도 차이납니다. 

여러 가지 실험을 이어 나가시고자 한다면, 순수 Linux 장비 구성을 강력히 권합니다.

순수 Linux는 Ubuntu를 사용한다면, 아래 방법과 크게 차이나지 않습니다.

## NVIDIA 드라이브 업데이트

- NVIDIA 장비 확인
시스템 > 정보 > 고급 시스템 설정

- NVIDIA 드라이버 업데이트

강의장 PC 기준

https://www.nvidia.com/ko-kr/drivers/details/241094/

설치 파일 다운로드 

실행 > NVIDIA 그래픽 드라이버 > 빠른 설치

## WSL 설치

커맨드창을 구동 시킵니다.

```cmd
wsl --install --distribution Ubuntu
```


## 실습환경 구축

### Docker 설치

[Docker](https://www.docker.com/get-started/)에서 설치 프로그램을 다운로드 받습니다.

### Docker 이미지 다운로드

[오라클 이미지](https://drive.google.com/file/d/1gjBAFlSTNfYWN4q5g-2pJ5LaTSkUNaT7/view?usp=drive_link)

[Qdrant 이미지](https://drive.google.com/file/d/1nnvgbnBvZtrAubOTuAajQllvcSWccb5h/view?usp=drive_link)

[실습환경 이미지](https://drive.google.com/file/d/1wZzDF3B2EYj5BpXIqBfE37UvZiILNw21/view?usp=drive_link)

### Docker 이미지 탑재

커멘드 창을 열고 docker 이미지가 있는 경로로 이동합니다.

1. Oracle Image 탑재

```
docker load -i oracle-rcmd.tar
```

2. Qdrant Image 탑재

```
docker load -i qdrant_rcmd.tar
```

3. 실습 환경 탑재

```
docker load -i multi_rcmd.tar
```

4. 구동 파일 수정

이 git에 run_rcmd.bat를 다운로드 받습니다.

run_rcmd.bat는 실습 환경을 구동시키는 docker 명령이 들어 있습니다. 명령은 아래 와 같습니다. 

```
docker run --gpus all --rm -p 8888:8888 -v D:\work/lecture/rcmd_2506:/work --network rcmd -it multi_rcmd:latest  jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser --notebook-dir=/work
```

이 경로를 실습 환경 내의 작업 경로를 로컬 PC에 해당 디렉터리로 바꾸어 줍니다.

위 내용에서 D:\work/lecture/rcmd_2506 가 작업 파일의 경로입니다. 이 경로를 로컬 PC의 경로로 변경하고 저장합니다. 


## 실습 환경 구동

커맨드 창에서 실습 폴더로 이동합니다. 

```cmd
run_rcmd
```
위 커맨드를 실행시키면, 실습 환경 이미지가 실행이 됩니다.

실습 환경은 Jupyter Lab 기반으로 되어 있습니다.

## 컨텐츠 기반 추천용 DB 구동

커맨드 창에서 실습 폴더로 이동합니다. 

DB 시작
```cmd
docker compose up -d
```

DB 종료

DB 종료시 DB에 기록된 내용들은 사라짐니다.
```cmd
docker compose down
```

