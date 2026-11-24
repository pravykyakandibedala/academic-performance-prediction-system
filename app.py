import streamlit as st
import csv
from pathlib import Path

# Project file paths
BASE_PATH = Path(__file__).resolve().parent
DATASET_FILE = BASE_PATH / "dataset" / "student_data.csv"
RESULT_FILE = BASE_PATH / "dataset" / "prediction_results_updated.csv"

st.set_page_config(
    page_title="Academic Performance Prediction System",
    page_icon="🎓",
    layout="wide"
)


# Train Linear Regression Model
@st.cache_data
def train_model(file_modified):

    with open(DATASET_FILE, "r", newline="", encoding="utf-8") as file:
        students = list(csv.DictReader(file))

    if not students:
        raise ValueError("Training dataset is empty.")

    X = []
    Y = []

    for student in students:

        row = [
            float(student["attendance"]),
            float(student["previous_marks"]),
            float(student["cie_marks"]),
            float(student["assignment_marks"])
        ]

        target = float(student["final_marks"])

        if not (
            0 <= row[0] <= 100
            and 0 <= row[1] <= 100
            and 0 <= row[2] <= 50
            and 0 <= row[3] <= 10
            and 0 <= target <= 100
        ):
            raise ValueError("Invalid marks in training dataset.")

        X.append(row)
        Y.append(target / 100)

    minimum = [
        min(row[i] for row in X) for i in range(4)
    ]

    maximum = [
        max(row[i] for row in X) for i in range(4)
    ]

    def normalize(row):

        return [
            (row[i] - minimum[i]) / (maximum[i] - minimum[i])
            if maximum[i] != minimum[i] else 0
            for i in range(4)
        ]

    X_scaled = [normalize(row) for row in X]

    weights = [0.0] * 4
    bias = sum(Y) / len(Y)

    learning_rate = 0.1
    epochs = 5000
    n = len(X_scaled)

    for epoch in range(epochs):

        dw = [0.0] * 4
        db = 0.0

        for x, y in zip(X_scaled, Y):

            prediction = sum(
                weights[i] * x[i] for i in range(4)
            ) + bias

            error = prediction - y

            for i in range(4):
                dw[i] += error * x[i]

            db += error

        for i in range(4):
            weights[i] -= learning_rate * dw[i] / n

        bias -= learning_rate * db / n

    return weights, bias, minimum, maximum


# Predict Final Marks
def predict_marks(attendance, previous_marks,
                  cie_marks, assignment_marks, model):

    weights, bias, minimum, maximum = model

    values = [
        attendance,
        previous_marks,
        cie_marks,
        assignment_marks
    ]

    scaled = [
        (values[i] - minimum[i]) / (maximum[i] - minimum[i])
        if maximum[i] != minimum[i] else 0
        for i in range(4)
    ]

    prediction = sum(
        weights[i] * scaled[i] for i in range(4)
    ) + bias

    return max(0, min(100, prediction * 100))


# Academic Risk Classification
def classify_risk(predicted, attendance, cie, assignment):

    reasons = []

    if attendance < 60:
        reasons.append("Low attendance")

    if cie < 20:
        reasons.append("Low CIE marks")

    if assignment < 4:
        reasons.append("Low assignment marks")

    if predicted < 40:
        reasons.append("Low predicted final marks")

    if reasons:
        return "High Risk", reasons

    if (
        attendance < 75
        or cie < 30
        or assignment < 6
        or predicted < 60
    ):

        if attendance < 75:
            reasons.append("Attendance needs improvement")

        if cie < 30:
            reasons.append("CIE performance needs improvement")

        if assignment < 6:
            reasons.append("Assignment performance needs improvement")

        if predicted < 60:
            reasons.append("Predicted marks need improvement")

        return "Medium Risk", reasons

    return "Low Risk", [
        "Academic performance is satisfactory"
    ]


# Website Heading
st.title("🎓 Academic Performance Prediction System")

st.write(
    "A Machine Learning based system to predict "
    "student academic performance and identify academic risks."
)

st.info("Machine Learning Algorithm: Linear Regression")


# Train Model
try:

    if not DATASET_FILE.exists():
        raise FileNotFoundError("Student dataset not found.")

    model = train_model(DATASET_FILE.stat().st_mtime)

    st.success("Machine Learning Model Trained Successfully!")

except Exception as error:

    st.error("Unable to train model: " + str(error))
    st.stop()


st.divider()


# Student Details
st.header("📝 Enter Student Details")

name = st.text_input("Student Name")

col1, col2 = st.columns(2)

with col1:

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0
    )

    previous_marks = st.number_input(
        "Previous Marks (out of 100)",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

with col2:

    cie_marks = st.number_input(
        "CIE Marks (out of 50)",
        min_value=0.0,
        max_value=50.0,
        value=35.0
    )

    assignment_marks = st.number_input(
        "Assignment Performance (out of 10)",
        min_value=0.0,
        max_value=10.0,
        value=7.0
    )


# Prediction
if st.button("Predict Marks"):

    if not name.strip():

        st.warning("Please enter student name.")

    else:

        predicted_marks = predict_marks(
            attendance,
            previous_marks,
            cie_marks,
            assignment_marks,
            model
        )

        risk, reasons = classify_risk(
            predicted_marks,
            attendance,
            cie_marks,
            assignment_marks
        )

        st.divider()
        st.header("📊 Prediction Result")

        st.metric(
            "Predicted Final Marks",
            f"{predicted_marks:.2f}/100"
        )

        st.subheader("Academic Risk Category")

        if risk == "High Risk":
            st.error("🔴 High Risk")

        elif risk == "Medium Risk":
            st.warning("🟡 Medium Risk")

        else:
            st.success("🟢 Low Risk")

        st.subheader("🚨 Early Warning Information")

        for reason in reasons:
            st.write("- " + reason)

        st.info(
            "Faculty Recommendation: Monitor student progress "
            "and provide academic support according to individual needs."
        )

        # Save Student Record
        try:

            RESULT_FILE.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            file_exists = RESULT_FILE.exists()

            with open(
                RESULT_FILE,
                "a",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.writer(file)

                if (
                    not file_exists
                    or RESULT_FILE.stat().st_size == 0
                ):

                    writer.writerow([
                        "Student Name",
                        "Attendance",
                        "Previous Marks",
                        "CIE Marks",
                        "Assignment Marks",
                        "Predicted Marks",
                        "Risk Category",
                        "Warning Information"
                    ])

                writer.writerow([
                    name.strip(),
                    attendance,
                    previous_marks,
                    cie_marks,
                    assignment_marks,
                    round(predicted_marks, 2),
                    risk,
                    "; ".join(reasons)
                ])

            st.success("Student record saved successfully!")

        except OSError as error:

            st.error("Unable to save record: " + str(error))


# Faculty Dashboard
st.divider()

st.header("📊 Faculty Academic Risk Dashboard")

try:

    if (
        RESULT_FILE.exists()
        and RESULT_FILE.stat().st_size > 0
    ):

        with open(
            RESULT_FILE,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            records = list(csv.DictReader(file))

        if records:

            total = len(records)

            high = sum(
                1 for r in records
                if r["Risk Category"] == "High Risk"
            )

            medium = sum(
                1 for r in records
                if r["Risk Category"] == "Medium Risk"
            )

            low = sum(
                1 for r in records
                if r["Risk Category"] == "Low Risk"
            )

            # Dashboard Summary
            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Total Students", total)
            c2.metric("High Risk", high)
            c3.metric("Medium Risk", medium)
            c4.metric("Low Risk", low)

            # Risk Distribution
            st.subheader("Academic Risk Distribution")

            st.write("🔴 High Risk:", high, "students")
            st.progress(high / total)

            st.write("🟡 Medium Risk:", medium, "students")
            st.progress(medium / total)

            st.write("🟢 Low Risk:", low, "students")
            st.progress(low / total)

            # Faculty Early Warning
            st.subheader("🚨 Faculty Early Warning List")

            warning_records = [
                r for r in records
                if r["Risk Category"] in [
                    "High Risk",
                    "Medium Risk"
                ]
            ]

            if warning_records:

                for index, record in enumerate(
                    warning_records, 1
                ):

                    with st.expander(
                        f"{index}. {record['Student Name']} - "
                        f"{record['Risk Category']}"
                    ):

                        st.write(
                            "Predicted Marks:",
                            record["Predicted Marks"]
                        )

                        st.write(
                            "Attendance:",
                            record["Attendance"], "%"
                        )

                        st.write(
                            "Previous Marks:",
                            record["Previous Marks"]
                        )

                        st.write(
                            "CIE Marks:",
                            record["CIE Marks"], "/50"
                        )

                        st.write(
                            "Assignment Marks:",
                            record["Assignment Marks"], "/10"
                        )

                        st.write(
                            "Warning:",
                            record["Warning Information"]
                        )

                        st.write(
                            "**Suggested Faculty Action:** "
                            "Provide academic guidance, monitor progress "
                            "and arrange additional support."
                        )

            else:

                st.success(
                    "No students currently require additional monitoring."
                )

            # All Student Records
            st.subheader("📚 All Student Records")

            for index, record in enumerate(records, 1):

                with st.expander(
                    f"Student {index}: {record['Student Name']}"
                ):

                    st.write("Attendance:", record["Attendance"], "%")

                    st.write(
                        "Previous Marks:",
                        record["Previous Marks"]
                    )

                    st.write(
                        "CIE Marks:",
                        record["CIE Marks"], "/50"
                    )

                    st.write(
                        "Assignment Marks:",
                        record["Assignment Marks"], "/10"
                    )

                    st.write(
                        "Predicted Marks:",
                        record["Predicted Marks"]
                    )

                    st.write(
                        "Risk Category:",
                        record["Risk Category"]
                    )

                    st.write(
                        "Warning:",
                        record["Warning Information"]
                    )

        else:

            st.info("No student records available yet.")

    else:

        st.info(
            "No student predictions available. "
            "Add student details first."
        )

except (OSError, csv.Error, ValueError, KeyError) as error:

    st.error("Unable to display records: " + str(error))
