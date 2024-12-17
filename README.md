# LG_aimers
## LG_aimers 프로젝트 참가
- LG에서 제공해주신 자동차 디스플레이 공정 과정에서 정상과 비정상을 분류하는 AI 모델을 어떻게 잘 학습시킬 수 있는가에 대한 문제
- 비정상 데이터가 정상 데이터보다 월등히 적을 때 data imbalance를 최대한 해결하는 방향으로 진행

---

<header>
    <h3> 💻 팀원 소개 </h3>
    
</header>
  
<table>
  <tbody>
    <tr>
      <td align="center">
  <img src="https://avatars.githubusercontent.com/u/102892446?v=4" width="100px;" alt="Lee Kang-hyun's profile photo"/><br />
  <sub><b>이강현</b></sub><br />
  <a href="https://github.com/lkhyun">lkhyun</a>
</td>
      <td align="center">
  <img src="https://avatars.githubusercontent.com/u/112750856?v=4" width="100px;" alt="Song Chae-young's profile photo"/><br />
  <sub><b>송채영</b></sub><br />
  <a href="https://github.com/cy0286">cy0286</a>
</td>
       <td align="center">
  <img src="https://avatars.githubusercontent.com/u/101550897?v=4" width="100px;" alt="Jo Hyun-ho's profile photo"/><br />
  <sub><b>조현호</b></sub><br />
  <a href="https://github.com/178kg78cm">178kg78cm</a>
</td>
    </tr>
  </tbody>
</table>

---

### ver1: 
    - 결측치 제거, feature one-hot encoding등 데이터 preprocessing 진행
    - 차원 감소를 위한 SVD 적용
    - 비정형 데이터 해결을 위한 SMOTE 기법 적용
    - randomforest classfier 사용

### ver2:
    - StandardScaler() -> MinMaxScaler() 변경
    - f1 score 19에서 21로 성능 증가

### 최종:
    - score 0.22093점으로 마무리
