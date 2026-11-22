import csv
from pathlib import Path

# Locate prediction results file
base_path = Path(__file__).resolve().parent.parent
file_path = base_path / "dataset" / "prediction_results.csv"

# Check whether file exists
if not file_path.exists():
    print("No student records found!")
else:
    with open(file_path, "r", newline="") as file:
        students = list(csv.DictReader(file))

    if not students:
        print("No student records available!")

    else:
        total_students = len(students)

        # Calculate average marks
        marks = [
            float(student["Predicted Marks"])
            for student in students
        ]

        average_marks = sum(marks) / total_students

        # Count performance levels
        excellent = 0
        good = 0
        average = 0
        needs_improvement = 0

        for student in students:
            performance = student["Performance"]

            if performance == "Excellent":
                excellent += 1
            elif performance == "Good":
                good += 1
            elif performance == "Average":
                average += 1
            else:
                needs_improvement += 1

        # Display report
        print("\n================================")
        print(" STUDENT PERFORMANCE REPORT")
        print("================================")

        print("Total Students:", total_students)
        print("Average Predicted Marks:", round(average_marks, 2))

        print("\nPerformance Analysis")
        print("----------------------------")
        print("Excellent:", excellent)
        print("Good:", good)
        print("Average:", average)
        print("Needs Improvement:", needs_improvement)

        print("\nStudent Comparison")
        print("----------------------------")

        for student in students:
            print(
                student["Student Name"],
                "-",
                student["Predicted Marks"],
                "-",
                student["Performance"]
            )

        print("================================")
        # Student Performance Bar Chart
print("\n===== PERFORMANCE GRAPH =====")

for student in students:
    name = student["Student Name"]
    marks = float(student["Predicted Marks"])

    bar = "#" * int(marks / 2)

    print(f"{name}: {bar} {marks}%")

print("=============================")