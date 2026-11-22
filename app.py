import streamlit as st
import csv
from pathlib import Path

# Project file paths
BASE_PATH = Path(__file__).resolve().parent
DATASET_FILE = BASE_PATH / "dataset" / "student_data.csv"
RESULT_FILE = BASE_PATH / "dataset" / "prediction_results.csv"

st.set_page_config(
    page_title="Academic Performance Prediction System",
    page_icon="🎓",
    layout="wide"
)

# Train the original Linear Regression model
@st.cache_data
def train_model():
    with open(DATASET_FILE, "r", newline="", encoding="utf-8") as file:
        students = list(csv.DictReader(file))

    if not students:
        raise ValueError("Training dataset is empty.")

    X = []
    Y = []

    for student in students:
        X.append([
            float(student["Study_Hours"]),
            float(student["Attendance"]),
            float(student["Previous_Marks"])
        ])
        Y.append(float(student["Final_Marks"]))

    minimum = [min(row[i] for row in X) for i in range(3)]
    maximum = [max(row[i] for row in X) for i in range(3)]

    def normalize(row):
        return [
            (row[i] - minimum[i]) / (maximum[i] - minimum[i])
            if maximum[i] != minimum[i] else 0
            for i in range(3)
        ]

    X_scaled = [normalize(row) for row in X]
    Y_scaled = [y / 100 for y in Y]

    weights = [0.0, 0.0, 0.0]
    bias = sum(Y_scaled) / len(Y_scaled)

    learning_rate = 0.1
    epochs = 5000
    n = len(X_scaled)

    for epoch in range(epochs):
        dw = [0.0, 0.0, 0.0]
        db = 0.0

        for x, y in zip(X_scaled, Y_scaled):
            prediction = sum(
                weights[i] * x[i] for i in range(3)
            ) + bias

            error = prediction - y

            for i in range(3):
                dw[i] += error * x[i]

            db += error

        for i in range(3):
            weights[i] -= learning_rate * dw[i] / n

        bias -= learning_rate * db / n

    return weights, bias, minimum, maximum


def predict_marks(study_hours, attendance, previous_marks, model):
    weights, bias, minimum, maximum = model

    values = [study_hours, attendance, previous_marks]

    scaled = [
        (values[i] - minimum[i]) / (maximum[i] - minimum[i])
        if maximum[i] != minimum[i] else 0
        for i in range(3)
    ]

    prediction = sum(
        weights[i] * scaled[i] for i in range(3)
    ) + bias

    return max(0, min(100, prediction * 100))


def classify_performance(marks):
    if marks >= 75:
        return "Excellent"
    elif marks >= 60:
        return "Good"
    elif marks >= 40:
        return "Average"
    return "Needs Improvement"


# Website heading
st.title("🎓 Academic Performance Prediction System")
st.write("Predict and analyse student academic performance.")

st.info("Machine Learning Model: Linear Regression")

# Train model
try:
    model = train_model()
    st.success("Machine Learning Model Trained Successfully!")
except Exception as error:
    st.error("Unable to train model: " + str(error))
    st.stop()

st.divider()

# Student details
st.header("📝 Enter Student Details")

name = st.text_input("Student Name")

col1, col2, col3 = st.columns(3)

with col1:
    study_hours = st.number_input(
        "Study Hours", min_value=0.0,
        max_value=24.0, value=5.0
    )

with col2:
    attendance = st.number_input(
        "Attendance (%)", min_value=0.0,
        max_value=100.0, value=80.0
    )

with col3:
    previous_marks = st.number_input(
        "Previous Marks", min_value=0.0,
        max_value=100.0, value=70.0
    )

if st.button("Predict Marks"):
    if not name.strip():
        st.warning("Please enter student name.")

    else:
        predicted_marks = predict_marks(
            study_hours, attendance, previous_marks, model
        )

        performance = classify_performance(predicted_marks)

        st.divider()
        st.header("📊 Prediction Result")

        st.metric(
            "Predicted Marks",
            f"{predicted_marks:.2f}/100"
        )

        st.success("Performance: " + performance)

        try:
            file_exists = RESULT_FILE.exists()

            with open(
                RESULT_FILE, "a", newline="", encoding="utf-8"
            ) as file:
                writer = csv.writer(file)

                if not file_exists or RESULT_FILE.stat().st_size == 0:
                    writer.writerow([
                        "Student Name",
                        "Study Hours",
                        "Attendance",
                        "Previous Marks",
                        "Predicted Marks",
                        "Performance"
                    ])

                writer.writerow([
                    name.strip(),
                    study_hours,
                    attendance,
                    previous_marks,
                    round(predicted_marks, 2),
                    performance
                ])

            st.success("Student record saved successfully!")

        except OSError as error:
            st.error("Could not save student record: " + str(error))


# Student records
st.divider()
st.header("📚 Student Records")

try:
    if RESULT_FILE.exists() and RESULT_FILE.stat().st_size > 0:
        with open(
            RESULT_FILE, "r", newline="", encoding="utf-8"
        ) as file:
            records = list(csv.DictReader(file))

        if records:
            st.write("Total Students:", len(records))

            marks = []
            categories = {
                "Excellent": 0,
                "Good": 0,
                "Average": 0,
                "Needs Improvement": 0
            }

            for record in records:
                marks.append(float(record["Predicted Marks"]))
                category = record["Performance"]

                if category in categories:
                    categories[category] += 1

            if marks:
                st.metric(
                    "Average Predicted Marks",
                    f"{sum(marks) / len(marks):.2f}"
                )

            st.subheader("Individual Student Records")

            for index, record in enumerate(records, start=1):
                with st.expander(
                    f"Student {index}: {record['Student Name']}"
                ):
                    for key, value in record.items():
                        st.write(key + ":", value)

            st.subheader("Performance Summary")

            for category, count in categories.items():
                st.write(category + ":", count, "students")

        else:
            st.info("No student records available.")

    else:
        st.info("No student records available yet.")

except (OSError, csv.Error, ValueError, KeyError) as error:
    st.error("Unable to display student records: " + str(error))