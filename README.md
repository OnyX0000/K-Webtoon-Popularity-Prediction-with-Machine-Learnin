# K-Webtoon Popularity Prediction with Machine Learning

* Prediction of Webtoon popularity (rank, interest) with comment, genre, tag
* January, 2025 \~ February, 2025 (for two weeks)

---

## **Presentation** 📽️

* [Canva Presentation Link](https://www.canva.com/design/DAGeC0mo_Ys/U_E-Bp7JnTml4pxRkTmq4w/view?utm_content=DAGeC0mo_Ys&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=ha6cdc818c4)

---

## 1. **Project Goal and Background** ⚽

### 🚀 **Background**

* 웹툰 산업이 급성장하면서, 흥행 여부를 사전에 예측하여 **추천 알고리즘**이나 **마케팅 전략**에 활용하는 것이 중요해졌습니다.
* 특히, 네이버 웹툰의 경우 다수의 사용자 평가와 댓글 데이터가 존재하여, 이를 활용한 **흥행 예측 모델** 개발이 필요합니다.
* 웹툰의 **좋아요 수**, **댓글 수**, **평점** 등 다양한 지표를 기반으로 **흥행 가능성**을 사전에 예측하고자 하였습니다.

---

### 🎯 **Goal**

* 웹툰 데이터를 통해 **흥행 예측 모델**을 구축하여, 웹툰의 인기도를 사전에 예측
* 주요 지표와 **감정 분석 결과**를 결합하여, 웹툰 흥행 여부와 감정 간의 **상관관계**를 분석
* **실시간 예측 모델**을 배포하여 사용자에게 흥행 예측을 제공

---

## 2. **About Processing** ✂️

### ✅ **Requirements**

* Python 3.11
* Pandas, Numpy, Matplotlib, Scikit-learn
* LightGBM, XGBoost, CatBoost
* Streamlit, Folium

---

### ✅ **Used Python IDE**

* Google Colab
* JupyterLab & Notebook
* VSCode

---

### ✅ **Variables**

* **Likes:** 좋아요 개수
* **Comments:** 댓글 수
* **Rating:** 평점
* **Participation:** 별점 참여 수
* **Tags:** 장르 및 태그 정보
* **KOTE Sentiment:** KOTE 모델을 통한 감정 분석 결과

---

### ✅ **Others**

* 감정 분석 모델: KOTE
* 네이버 웹툰 데이터 크롤링
* 모델 저장: Joblib을 활용하여 `.pkl` 파일로 모델 저장

---
## 3. **About Conclusion** 💡

### ✅ **Sources**

* 네이버 웹툰 데이터 지표
* 베스트 댓글(연재 중, 완결 웹툰)

### ✅ **Raw Data**

* 총 35개의 CSV 파일
* 주요 데이터:

  * **train.csv** - 학습용 데이터
  * **validate.csv** - 검증용 데이터
  * **test.csv** - 테스트 데이터
  * **labels\_included.csv** - 라벨 포함 데이터
  * **final.csv** - 최종 전처리 데이터
* 주요 컬럼:

  * `webtoon_name`, `rating`, `interest`, `genre`, `comments_count`, `best_comments`
  * 감정 분석 컬럼: `즐거움/신남`, `짜증`, `행복`, `화남/분노`, `증오/혐오`

---

### ✅ **Model Performance**

<div align="center">
  <img src="https://github.com/OnyX0000/프로젝트명/assets/모델성과이미지경로.png" width="700"/>
</div>

- **모델 선정 사유**  
  LightGBM과 SoftMax 기반 모델은 학습결과의 분산이 컸던 반면, XGBoost는 상대적으로 안정적인 성능을 보여 선택하였습니다.  
  Cross-validation 기준 f1_macro는 **0.8788**로 측정되었습니다.

- **성능 지표**  
  다중 클래스 분류 기준, 클래스 간 불균형에도 불구하고 평균 F1 Score가 **0.61**, Accuracy **0.63**을 기록했습니다.

- **Feature Importance 분석 결과**  
  `별점 참여 수`가 가장 중요한 피처로 나타났으며, **작가의 전작 여부**보다는 **작품의 원작 여부(예: 웹소설 기반)**가 흥행에 더 큰 영향을 주는 것으로 분석되었습니다.

---

## 📌 **Project Structure**

```
app.py : Streamlit으로 웹 애플리케이션 구현
```

---

## 4. **Demo** 📹

* Streamlit 웹 애플리케이션 실행 화면
![Image](https://github.com/user-attachments/assets/c0b85d14-eb6e-4ee4-899f-1de11b0a236a)
---

## 5. **References** 📋

* [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
* [CatBoost Documentation](https://catboost.ai/en/docs/)
* [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/)

---

## 📌 **Improvement & Reflection**

* **데이터 수집의 한계**

  * 웹툰별로 데이터 수집 시 실제 조회수와 관련한 데이터가 없어 생각한 것보다 모델의 성능을 올리기 어려웠음
  * 추가 데이터 확보나 새로운 컬럼을 만들어야 함

* **모델 최적화 문제**

  * XGBoost와 CatBoost 모델 학습 시 GPU 메모리 사용 최적화 문제 발생
  * Batch 학습과 메모리 관리 기법을 통해 개선

* **향후 개선 방향**

  * LLM 기반 감정 분석을 도입하여 감정 분류 성능 향상
  * 다양한 웹툰 플랫폼 데이터를 통합하여 범용성 강화
  * 현재는 데이터 전처리를 거친 데이터만 받는 걸로 했지만 추후에는 웹툰 이름만 입력하면 예측해주는 파이프라인 구성


## 6. Participants & Supports 🧑‍🤝‍🧑
- 김형후
  -  [GitHub 🐈‍⬛](https://github.com/Shaerrr)
  - E-mail📧: kimjinsyll@gmail.com
  - blog 🏠: [tistroy](https://huhulog.tistory.com/ "티스토리 블로그") [Naver](https://blog.naver.com/dcfjk1234 "네이버 블로그") [Linked in](https://www.linkedin.com/in/%ED%98%95%ED%9B%84-%EA%B9%80-905659337/)
- 이진규 
  -  [GitHub 🐈‍⬛](https://github.com/OnyX0000)
  - E-mail📧: jinkyu2jinkyu@gmail.com
- 한지예
