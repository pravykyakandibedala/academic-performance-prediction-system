import streamlit as st
import csv
import io
from pathlib import Path
from supabase import create_client

# Project file paths
BASE_PATH = Path(__file__).resolve().parent
DATASET_FILE = BASE_PATH / "dataset" / "student_data.csv"
RESULT_FILE = BASE_PATH / "dataset" / "prediction_results_updated.csv"

st.set_page_config(
    page_title="Academic Performance Prediction System",
    page_icon="🎓",
    layout="wide"
)

# Connect to Supabase
@st.cache_resource
def get_supabase():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )

try:
    supabase = get_supabase()
except Exception as error:
    st.error(f"Database connection failed:{error}")
    st.stop()


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

    minimum = [min(row[i] for row in X) for i in range(4)]
    maximum = [max(row[i] for row in X) for i in range(4)]

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


# Read all records from Supabase
def get_records():

    response = (
        supabase.table("student_records")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )

    return response.data or []


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

        # Save record permanently in Supabase
        try:
            student_record = {
                "student_name": name.strip(),
                "attendance": attendance,
                "previous_marks": previous_marks,
                "cie_marks": cie_marks,
                "assignment_marks": assignment_marks,
                "predicted_marks": round(predicted_marks, 2),
                "risk_category": risk,
                "warning_information": "; ".join(reasons)
            }

            supabase.table("student_records").insert(
                student_record
            ).execute()

            st.success("Student record saved permanently in database!")

        except Exception as error:
            st.error("Unable to save student record: " + str(error))


# Faculty Dashboard
st.divider()
st.header("📊 Faculty Academic Risk Dashboard")

try:
    records = get_records()

    if records:

        total = len(records)

        high = sum(
            1 for r in records
            if r["risk_category"] == "High Risk"
        )

        medium = sum(
            1 for r in records
            if r["risk_category"] == "Medium Risk"
        )

        low = sum(
            1 for r in records
            if r["risk_category"] == "Low Risk"
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
            if r["risk_category"] in [
                "High Risk", "Medium Risk"
            ]
        ]

        if warning_records:
            for index, record in enumerate(warning_records, 1):

                with st.expander(
                    f"{index}. {record['student_name']} - "
                    f"{record['risk_category']}"
                ):

                    st.write("Predicted Marks:", record["predicted_marks"])
                    st.write("Attendance:", record["attendance"], "%")
                    st.write("Previous Marks:", record["previous_marks"])
                    st.write("CIE Marks:", record["cie_marks"], "/50")
                    st.write(
                        "Assignment Marks:",
                        record["assignment_marks"], "/10"
                    )
                    st.write(
                        "Warning:",
                        record["warning_information"]
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
                f"Student {index}: {record['student_name']}"
            ):

                st.write("Attendance:", record["attendance"], "%")
                st.write("Previous Marks:", record["previous_marks"])
                st.write("CIE Marks:", record["cie_marks"], "/50")
                st.write(
                    "Assignment Marks:",
                    record["assignment_marks"], "/10"
                )
                st.write("Predicted Marks:", record["predicted_marks"])
                st.write("Risk Category:", record["risk_category"])
                st.write(
                    "Warning:",
                    record["warning_information"]
                )

        # CSV Export
        st.subheader("📥 Download Student Records")

        csv_buffer = io.StringIO()

        fieldnames = [
            "Student Name",
            "Attendance",
            "Previous Marks",
            "CIE Marks",
            "Assignment Marks",
            "Predicted Marks",
            "Risk Category",
            "Warning Information"
        ]

        writer = csv.DictWriter(
            csv_buffer,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for record in records:
            writer.writerow({
                "Student Name": record["student_name"],
                "Attendance": record["attendance"],
                "Previous Marks": record["previous_marks"],
                "CIE Marks": record["cie_marks"],
                "Assignment Marks": record["assignment_marks"],
                "Predicted Marks": record["predicted_marks"],
                "Risk Category": record["risk_category"],
                "Warning Information": record["warning_information"]
            })

        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="Download CSV File",
            data=csv_data,
            file_name="prediction_results_updated.csv",
            mime="text/csv"
        )

        # Save a local CSV copy when running locally
        try:
            RESULT_FILE.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with open(
                RESULT_FILE,
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                file.write(csv_data)

        except OSError:
            pass

    else:
        st.info("No student predictions available. Add student details first.")

except Exception as error:
    st.error("Unable to load student records: " + str(error))
