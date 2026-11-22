import csv
from pathlib import Path

# File paths
base_path = Path(__file__).resolve().parent.parent
dataset_path = base_path / "dataset" / "student_data.csv"
result_path = base_path / "dataset" / "prediction_results.csv"

# Load student dataset
with open(dataset_path, "r", newline="") as file:
    students = list(csv.DictReader(file))

# Prepare training data
X = []
Y = []

for student in students:
    X.append([
        float(student["Study_Hours"]),
        float(student["Attendance"]),
        float(student["Previous_Marks"])
    ])
    Y.append(float(student["Final_Marks"]))

# Normalize input values
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

# Train Linear Regression model
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

print("Machine Learning Model Trained Successfully!")

# Student details
print("\n=== ACADEMIC PERFORMANCE PREDICTION ===")

name = input("Enter Student Name: ").strip()

try:
    study_hours = float(input("Enter Study Hours: "))
    attendance = float(input("Enter Attendance Percentage: "))
    previous_marks = float(input("Enter Previous Marks: "))

    if not name or study_hours < 0 or not 0 <= attendance <= 100 or not 0 <= previous_marks <= 100:
        print("Please enter valid student details.")

    else:
        new_student = normalize([
            study_hours,
            attendance,
            previous_marks
        ])

        predicted = sum(
            weights[i] * new_student[i] for i in range(3)
        ) + bias

        predicted_marks = max(0, min(100, predicted * 100))

        # Performance classification
        if predicted_marks >= 75:
            performance = "Excellent"
        elif predicted_marks >= 60:
            performance = "Good"
        elif predicted_marks >= 40:
            performance = "Average"
        else:
            performance = "Needs Improvement"

        # Display results
        print("\n===== PREDICTION RESULT =====")
        print("Student Name:", name)
        print("Predicted Marks:", round(predicted_marks, 2))
        print("Performance:", performance)
        print("============================")

        # Save results in CSV file
        file_exists = result_path.exists()

        with open(result_path, "a", newline="") as file:
            writer = csv.writer(file)

            if not file_exists or result_path.stat().st_size == 0:
                writer.writerow([
                    "Student Name",
                    "Study Hours",
                    "Attendance",
                    "Previous Marks",
                    "Predicted Marks",
                    "Performance"
                ])

            writer.writerow([
                name,
                study_hours,
                attendance,
                previous_marks,
                round(predicted_marks, 2),
                performance
            ])

        print("\nStudent result saved successfully!")
        print("Results File:", result_path)

except ValueError:
    print("Invalid input! Please enter numbers for marks, attendance and study hours.")