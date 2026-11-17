
import streamlit as st
import pandas as pd

from predictor import StudentPerformancePredictor

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------
# CUSTOM CSS
# --------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
}

.hero {
    background: linear-gradient(120deg, #173b68, #2879a8);
    padding: 30px;
    border-radius: 15px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
}

.hero p {
    color: white;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# HEADER
# --------------------------------

st.markdown("""
<div class="hero">

<h1>🎓 Student Performance Prediction</h1>

<p>
Predict student examination marks
using Machine Learning.
</p>

</div>
""", unsafe_allow_html=True)


# --------------------------------
# LOAD MODEL
# --------------------------------

@st.cache_resource
def load_model():

    return StudentPerformancePredictor(
        "student_performance_model.pkl"
    )


try:

    predictor = load_model()

except Exception as e:

    st.error("Model loading failed!")

    st.exception(e)

    st.stop()


# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.title("About Project")

st.sidebar.write("""
This application predicts student
exam scores using supervised
machine learning regression.

Enter student information and
click the prediction button.
""")

st.sidebar.info(
    "Algorithm: Trained Regression Model"
)


# --------------------------------
# STUDENT INPUT FORM
# --------------------------------

st.subheader("Enter Student Information")

st.write(
    "Please enter the following details:"
)

with st.form("student_form"):

    col1, col2 = st.columns(2)

    values = {}

    for i, feature in enumerate(predictor.feature_names):

        container = col1 if i % 2 == 0 else col2

        with container:

            label = feature.replace("_", " ")

            if feature in predictor.categorical_options:

                values[feature] = st.selectbox(
                    label,
                    predictor.categorical_options[feature]
                )

            elif feature in predictor.numeric_ranges:

                low, high, default, step = (
                    predictor.numeric_ranges[feature]
                )

                values[feature] = st.number_input(
                    label,
                    min_value=float(low),
                    max_value=float(high),
                    value=float(default),
                    step=float(step)
                )

            else:

                values[feature] = st.number_input(
                    label,
                    value=0.0
                )

    submit = st.form_submit_button(
        "🔍 Predict Student Performance",
        use_container_width=True
    )


# --------------------------------
# PREDICTION
# --------------------------------

if submit:

    try:

        prediction = predictor.predict(values)

        st.divider()

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Exam Score",
                f"{prediction:.2f} / 100"
            )

        with col2:

            if prediction >= 90:

                st.success("Excellent Performance 🌟")

            elif prediction >= 75:

                st.success("Good Performance 👍")

            elif prediction >= 60:

                st.info("Satisfactory Performance")

            elif prediction >= 40:

                st.warning("Needs Improvement")

            else:

                st.error("At Risk")

        st.progress(
            int(max(0, min(100, prediction)))
        )

        st.subheader("Student Input Details")

        input_df = pd.DataFrame([values])

        st.dataframe(
            input_df,
            use_container_width=True
        )

    except Exception as e:

        st.error("Prediction Failed!")

        st.exception(e)


# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.caption(
    "Student Performance Prediction | "
    "Machine Learning Project | Python + Streamlit"
)
