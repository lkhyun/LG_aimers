# LG_aimers
## LG_aimers 프로젝트 참가
- LG에서 제공해주신 자동차 디스플레이 공정 과정에서 정상과 비정상을 분류하는 AI 모델을 어떻게 잘 학습시킬 수 있는가에 대한 문제
- 비정상 데이터가 정상 데이터보다 월등히 적을 때 data imbalance를 최대한 해결하는 방향으로 진행

---
### ver1: 
    - 결측치 제거, feature one-hot encoding등 데이터 preprocessing 진행
    - 차원 감소를 위한 SVD 적용
    - 비정형 데이터 해결을 위한 SMOTE 기법 적용
    - randomforest classfier 사용

### ver2:
    - StandardScaler() -> MinMaxScaler() 변경
    - f1 score 19에서 21.679로 성능 증가
