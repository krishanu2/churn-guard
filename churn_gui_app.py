import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load model and data
model = joblib.load("churn_model.joblib")
df = pd.read_csv("telco_customer_churn.csv")

root = tk.Tk()
root.title("Customer Churn Predictor")
root.geometry("1050x650")

# Create main layout frames
form_frame = tk.Frame(root, padx=20, pady=20)
form_frame.pack(side="left", fill="both", expand=True)

graph_frame = tk.Frame(root, padx=20, pady=20)
graph_frame.pack(side="right", fill="both", expand=True)

# Store canvas globally
canvas = None

# Input Fields
fields = [
    ("Gender", "combo", ["Male", "Female"]),
    ("Senior Citizen", "combo", ["0", "1"]),
    ("Partner", "combo", ["Yes", "No"]),
    ("Dependents", "combo", ["Yes", "No"]),
    ("Tenure", "spin", (0, 72)),
    ("Phone Service", "combo", ["Yes", "No"]),
    ("Multiple Lines", "combo", ["Yes", "No", "No phone service"]),
    ("Internet Service", "combo", ["DSL", "Fiber optic", "No"]),
    ("Online Security", "combo", ["Yes", "No", "No internet service"]),
    ("Online Backup", "combo", ["Yes", "No", "No internet service"]),
    ("Device Protection", "combo", ["Yes", "No", "No internet service"]),
    ("Tech Support", "combo", ["Yes", "No", "No internet service"]),
    ("Streaming TV", "combo", ["Yes", "No", "No internet service"]),
    ("Streaming Movies", "combo", ["Yes", "No", "No internet service"]),
    ("Contract", "combo", ["Month-to-month", "One year", "Two year"]),
    ("Paperless Billing", "combo", ["Yes", "No"]),
    ("Payment Method", "combo", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]),
    ("Monthly Charges", "spin", (10, 120)),
    ("Total Charges", "spin", (0, 10000))
]

widgets = {}

# Place all widgets in form_frame using grid
for idx, (label_text, widget_type, options) in enumerate(fields):
    tk.Label(form_frame, text=label_text).grid(row=idx, column=0, sticky="w", pady=3)
    if widget_type == "combo":
        cb = ttk.Combobox(form_frame, values=options, state="readonly", width=30)
        cb.current(0)
        cb.grid(row=idx, column=1, pady=3, padx=10)
        widgets[label_text] = cb
    elif widget_type == "spin":
        sb = tk.Spinbox(form_frame, from_=options[0], to=options[1], width=33)
        sb.grid(row=idx, column=1, pady=3, padx=10)
        widgets[label_text] = sb

# Prediction Function
def predict():
    try:
        input_data = {
            "gender": widgets["Gender"].get(),
            "SeniorCitizen": int(widgets["Senior Citizen"].get()),
            "Partner": widgets["Partner"].get(),
            "Dependents": widgets["Dependents"].get(),
            "tenure": int(widgets["Tenure"].get()),
            "PhoneService": widgets["Phone Service"].get(),
            "MultipleLines": widgets["Multiple Lines"].get(),
            "InternetService": widgets["Internet Service"].get(),
            "OnlineSecurity": widgets["Online Security"].get(),
            "OnlineBackup": widgets["Online Backup"].get(),
            "DeviceProtection": widgets["Device Protection"].get(),
            "TechSupport": widgets["Tech Support"].get(),
            "StreamingTV": widgets["Streaming TV"].get(),
            "StreamingMovies": widgets["Streaming Movies"].get(),
            "Contract": widgets["Contract"].get(),
            "PaperlessBilling": widgets["Paperless Billing"].get(),
            "PaymentMethod": widgets["Payment Method"].get(),
            "MonthlyCharges": float(widgets["Monthly Charges"].get()),
            "TotalCharges": float(widgets["Total Charges"].get())
        }

        input_df = pd.DataFrame([input_data])
        input_encoded = pd.get_dummies(input_df)

        for col in model.feature_names_in_:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model.feature_names_in_]

        prediction = model.predict(input_encoded)[0]
        prob = model.predict_proba(input_encoded)[0][1]

        messagebox.showinfo("Prediction Result", f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}\nProbability: {prob:.2f}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Plotting Function
def show_churn_plot():
    global canvas
    fig, ax = plt.subplots(figsize=(4.5, 3))
    churn_counts = df["Churn"].value_counts()
    ax.bar(churn_counts.index, churn_counts.values, color=["green", "red"])
    ax.set_title("Churn Distribution")
    ax.set_ylabel("Count")

    if canvas:
        canvas.get_tk_widget().destroy()

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Buttons
tk.Button(form_frame, text="Predict Churn", command=predict, bg="#4CAF50", fg="white", width=25)\
    .grid(row=len(fields)+1, column=0, columnspan=2, pady=15)

tk.Button(form_frame, text="Show Churn Distribution", command=show_churn_plot, bg="#2196F3", fg="white", width=25)\
    .grid(row=len(fields)+2, column=0, columnspan=2)

# Show Graph Immediately (optional)
# show_churn_plot()

root.mainloop()
