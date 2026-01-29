import sys
import subprocess
import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib
import traceback  # Used to see the exact location of errors

# -----------------------------------------------------------
# 0. LIBRARY CHECK
# -----------------------------------------------------------
def install_and_import(package, import_name):
    try:
        __import__(import_name)
    except ImportError:
        pass  # Silently ignore for now

install_and_import("xgboost", "xgboost")
import xgboost

# -----------------------------------------------------------
# 1. SETTINGS AND MODEL LOADING
# -----------------------------------------------------------

models = {}
model_files = {
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "K-Means (Clustering)": "kmeans_cluster_model.pkl"
}

# Load models
print("--- MODEL LOADING REPORT ---")
for name, filename in model_files.items():
    if os.path.exists(filename):
        try:
            models[name] = joblib.load(filename)
            print(f"✅ Loaded Successfully: {name}")
        except Exception as e:
            print(f"❌ CORRUPTED: {filename} -> {e}")
    else:
        print(f"⚠️ NOT FOUND: {filename}")

# Categorical mappings
maps = {
    'Low': 1, 'Medium': 2, 'High': 3,
    'High School': 1, 'College': 2, 'Postgraduate': 3,
    'Negative': 1, 'Neutral': 2, 'Positive': 3,
    'Near': 1, 'Moderate': 2, 'Far': 3,
    'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0,
    'Public': 1, 'Private': 0
}

# Normalization boundaries
min_max_vals = {
    'Hours_Studied': (1, 44), 'Attendance': (60, 100), 'Sleep_Hours': (4, 10),
    'Physical_Activity': (0, 6), 'Tutoring_Sessions': (0, 8), 'Previous_Scores': (50, 100),
    'Motivation_Level': (1, 3), 'Parental_Involvement': (1, 3), 'Family_Income': (1, 3),
    'Teacher_Quality': (1, 3), 'Access_to_Resources': (1, 3), 'Parental_Education_Level': (1, 3),
    'Peer_Influence': (1, 3), 'Distance_from_Home': (1, 3),
    'Academic_Discipline': (69, 3783),
    'Learning_Efficiency': (1.333333, 48.500000)
}

# -----------------------------------------------------------
# 2. PREDICTION FUNCTION (DEBUG MODE)
# -----------------------------------------------------------
def tahmin_et():
    try:
        # 1. Model selection
        secilen_model = combo_model.get()
        if not secilen_model:
            messagebox.showerror("Error", "No model selected!")
            return

        if secilen_model not in models:
            messagebox.showerror("Error", f"'{secilen_model}' could not be loaded.")
            return

        model = models[secilen_model]

        # 2. Read numerical inputs
        try:
            raw_inputs = {
                'Hours_Studied': float(entries['Hours_Studied'].get()),
                'Attendance': float(entries['Attendance'].get()),
                'Sleep_Hours': float(entries['Sleep_Hours'].get()),
                'Physical_Activity': float(entries['Physical_Activity'].get()),
                'Tutoring_Sessions': float(entries['Tutoring_Sessions'].get()),
                'Previous_Scores': float(entries['Previous_Scores'].get())
            }
        except ValueError:
            messagebox.showerror("Error", "Invalid numeric input (do not leave empty, use dot).")
            return

        # 3. Categorical mapping
        inputs = raw_inputs.copy()
        inputs.update({
            'Motivation_Level': maps[combos['Motivation_Level'].get()],
            'Learning_Disabilities': maps[combos['Learning_Disabilities'].get()],
            'Parental_Involvement': maps[combos['Parental_Involvement'].get()],
            'Family_Income': maps[combos['Family_Income'].get()],
            'Teacher_Quality': maps[combos['Teacher_Quality'].get()],
            'Access_to_Resources': maps[combos['Access_to_Resources'].get()],
            'Parental_Education_Level': maps[combos['Parental_Education_Level'].get()],
            'Peer_Influence': maps[combos['Peer_Influence'].get()],
            'Distance_from_Home': maps[combos['Distance_from_Home'].get()],
            'Internet_Access': maps[combos['Internet_Access'].get()],
            'Extracurricular_Activities': maps[combos['Extracurricular_Activities'].get()],
            'School_Type': maps[combos['School_Type'].get()],
            'Gender': maps[combos['Gender'].get()]
        })

        # 4. Derived features
        inputs['Academic_Discipline'] = inputs['Attendance'] * inputs['Hours_Studied']
        inputs['Learning_Efficiency'] = inputs['Previous_Scores'] / (inputs['Hours_Studied'] + 1)

        # 5. Normalization
        for k, (min_v, max_v) in min_max_vals.items():
            if k in inputs:
                inputs[k] = (inputs[k] - min_v) / (max_v - min_v)

        # 6. Create DataFrame
        cols = [
            'Hours_Studied','Attendance','Sleep_Hours','Physical_Activity','Tutoring_Sessions',
            'Previous_Scores','Motivation_Level','Learning_Disabilities','Parental_Involvement',
            'Family_Income','Teacher_Quality','Access_to_Resources','Parental_Education_Level',
            'Peer_Influence','Distance_from_Home','Internet_Access','Extracurricular_Activities',
            'School_Type','Gender','Academic_Discipline','Learning_Efficiency'
        ]

        df_pred = pd.DataFrame([inputs], columns=cols)

        # ===========================================================
        # 7. CLUSTER CALCULATION — ONLY FOR RF AND XGBOOST
        # ===========================================================
        if secilen_model in ["Random Forest", "XGBoost"]:
            if os.path.exists("kmeans_cluster_model.pkl"):
                kmeans = joblib.load("kmeans_cluster_model.pkl")
                cluster_value = kmeans.predict(df_pred)[0]
                df_pred["Cluster"] = cluster_value
            else:
                messagebox.showerror("Error", "kmeans_cluster_model.pkl not found!")
                return

        # ===========================================================
        # 8. MODEL PREDICTION
        # ===========================================================
        sonuc = model.predict(df_pred)[0]

        # ===========================================================
        # 9. K-MEANS VISUAL RESULT
        # ===========================================================
        if "K-Means" in secilen_model:

            if os.path.exists("cluster_labels.pkl"):
                label_map = joblib.load("cluster_labels.pkl")
                kume_etiket = label_map[int(sonuc)]

                renk = {
                    "Yüksek Başarı": "green",
                    "Orta Başarı": "orange",
                    "Düşük Başarı": "red"
                }[kume_etiket]

                lbl_sonuc.config(
                    text=f"Student Profile: {kume_etiket}",
                    fg=renk,
                    font=("Arial", 16, "bold")
                )
                return
            else:
                messagebox.showerror("Error", "cluster_labels.pkl not found!")
                return

        # ===========================================================
        # 10. NORMAL GRADE PREDICTION
        # ===========================================================
        color = "green" if sonuc >= 50 else "red"
        lbl_sonuc.config(text=f"Predicted Grade: {sonuc:.2f}", fg=color)

    except Exception as e:
        traceback.print_exc()
        messagebox.showerror("Critical Error", str(e))

# -----------------------------------------------------------
# 3. USER INTERFACE
# -----------------------------------------------------------
root = tk.Tk()
root.title("Student Grade Prediction System (DEBUG MODE) 🔧")
root.geometry("800x650")

# Scrollbar setup
main_frame = tk.Frame(root); main_frame.pack(fill=tk.BOTH, expand=1)
canvas = tk.Canvas(main_frame); canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
scrollbar = ttk.Scrollbar(main_frame, command=canvas.yview); scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
frame_inputs = tk.Frame(canvas, padx=20, pady=20)
canvas.create_window((0, 0), window=frame_inputs, anchor="nw")

# Model selection
tk.Label(frame_inputs, text="AI Model:", font=("Arial", 12)).grid(row=0, column=0, columnspan=2)
combo_model = ttk.Combobox(frame_inputs, values=list(models.keys()), state="readonly", width=35)
if models: combo_model.current(0)
combo_model.grid(row=1, column=0, columnspan=2, pady=(0, 15))

# Title
tk.Label(frame_inputs, text="Student Information", font=("Arial", 14, "bold")).grid(row=2, column=0, columnspan=2, pady=10)

entries = {}
combos = {}
row = 3

def add_numeric(text, key):
    global row
    tk.Label(frame_inputs, text=text).grid(row=row, column=0, sticky="e", padx=5, pady=5)
    e = tk.Entry(frame_inputs); e.grid(row=row, column=1, sticky="w")
    entries[key] = e
    row += 1

def add_combo(text, key, vals):
    global row
    tk.Label(frame_inputs, text=text).grid(row=row, column=0, sticky="e", padx=5, pady=5)
    c = ttk.Combobox(frame_inputs, values=vals, state="readonly")
    c.grid(row=row, column=1, sticky="w")
    c.current(0)
    combos[key] = c
    row += 1

add_numeric("Weekly Study Hours:", "Hours_Studied")
add_numeric("Attendance Rate (0-100):", "Attendance")
add_numeric("Sleep Hours:", "Sleep_Hours")
add_numeric("Physical Activity:", "Physical_Activity")
add_numeric("Tutoring Sessions:", "Tutoring_Sessions")
add_numeric("Previous Grade:", "Previous_Scores")

tk.Label(frame_inputs, text="--- Other Factors ---", font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2, pady=10); row+=1

add_combo("Motivation Level:", "Motivation_Level", ["Low", "Medium", "High"])
add_combo("Learning Disability:", "Learning_Disabilities", ["No", "Yes"])
add_combo("Parental Involvement:", "Parental_Involvement", ["Low", "Medium", "High"])
add_combo("Family Income:", "Family_Income", ["Low", "Medium", "High"])
add_combo("Teacher Quality:", "Teacher_Quality", ["Low", "Medium", "High"])
add_combo("Access to Resources:", "Access_to_Resources", ["Low", "Medium", "High"])
add_combo("Parental Education:", "Parental_Education_Level", ["High School", "College", "Postgraduate"])
add_combo("Peer Influence:", "Peer_Influence", ["Negative", "Neutral", "Positive"])
add_combo("Distance from Home:", "Distance_from_Home", ["Near", "Moderate", "Far"])
add_combo("Internet Access:", "Internet_Access", ["Yes", "No"])
add_combo("Extracurricular Activities:", "Extracurricular_Activities", ["Yes", "No"])
add_combo("School Type:", "School_Type", ["Public", "Private"])
add_combo("Gender:", "Gender", ["Male", "Female"])

btn = tk.Button(frame_inputs, text="CALCULATE", font=("Arial", 12, "bold"), bg="#d9534f", fg="white", command=tahmin_et)
btn.grid(row=row, column=0, columnspan=2, pady=20); row+=1

lbl_sonuc = tk.Label(frame_inputs, text="Waiting for result...", font=("Arial", 16, "bold"))
lbl_sonuc.grid(row=row, column=0, columnspan=2, pady=10)

root.mainloop()
