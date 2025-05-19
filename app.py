import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb

# Streamlit 애플리케이션 제목
st.title("📊 웹툰 클러스터 예측")
st.write("업로드한 웹툰 데이터를 기반으로 XGBoost 분류모델이 클러스터를 예측합니다.")

# 클러스터 설명 추가
st.subheader("📌 클러스터 설명")
st.write("""
클러스터 값은 웹툰의 **흥행 정도**를 의미합니다.

- **1️⃣ 매우 낮음** : 인기 및 반응이 낮은 웹툰
- **2️⃣ 낮음** : 일정한 반응은 있지만 인기가 낮은 웹툰
- **3️⃣ 중간** : 평균적인 반응을 보이는 웹툰
- **4️⃣ 높음** : 대체로 인기가 높은 웹툰
- **5️⃣ 매우 높음** : 매우 흥행한 인기 웹툰
""")

# 클러스터 설명 매핑 (예측된 숫자 → 설명)
cluster_descriptions = {
    1: "매우 낮음 (인기 및 반응이 낮음)",
    2: "낮음 (일정한 반응은 있지만 인기가 낮음)",
    3: "중간 (평균적인 반응을 보이는 웹툰)",
    4: "높음 (대체로 인기가 높은 웹툰)",
    5: "매우 높음 (매우 흥행한 인기 웹툰)"
}

# XGBoost 모델 로드 (캐싱 적용)
@st.cache_resource
def load_model():
    try:
        with open("best_xgb_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ 모델 로딩 실패: {e}")
        return None

best_xgb_model = load_model()

# CSV 파일 업로드
uploaded_file = st.file_uploader("📂 예측할 CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    try:
        # 데이터 불러오기
        test_df = pd.read_csv(uploaded_file)

        # 'cluster' 열 제거 (타겟 값이므로 예측 시 포함하지 않음)
        if "cluster" in test_df.columns:
            test_df.drop(columns=["cluster"], inplace=True)

        # 예측 수행
        predictions = best_xgb_model.predict(test_df)

        # 예측된 클러스터 값에 1을 더해 1~5 범위로 변환
        test_df = pd.DataFrame({
            "예측된 클러스터": predictions + 1,
        })

        # 클러스터 설명 추가
        test_df["클러스터 설명"] = test_df["예측된 클러스터"].map(cluster_descriptions)

        # 예측 결과 출력
        st.subheader("📊 예측된 클러스터 결과")
        st.dataframe(test_df)

        # CSV 다운로드 버튼 추가
        st.download_button(
            label="📥 예측 결과 다운로드 (CSV)",
            data=test_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")