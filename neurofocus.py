# Importing all libraries
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Dataset loading
data = pd.read_csv('EEG_Eye_State_Classification.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Sound setup
pygame.mixer.init()
pygame.mixer.music.load('alert.mp3')

# Globals
session_data = []
beep_allowed = True
cooldown_seconds = 3
running = False
session_completed = False
session_seconds = 0
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Tkinter Setup
window = tk.Tk()
window.title("NeuroFocus Pro: Multi-Session Real-Time Focus Detector")
window.geometry("1400x800")
window.configure(bg="#E8F0F2")

video_frame = tk.Frame(window, bg="#ffffff", bd=4, relief="groove")
video_frame.place(x=20, y=20, width=640, height=480)

video_label = tk.Label(video_frame)
video_label.pack()

status_label = tk.Label(window, text="Status: Waiting to Start", font=("Arial", 28, "bold"), fg="#2E86AB", bg="#E8F0F2")
status_label.place(x=700, y=50)

counter_label = tk.Label(window, text="Focus: 0 | Distract: 0", font=("Arial", 22), bg="#E8F0F2")
counter_label.place(x=700, y=120)

timer_label = tk.Label(window, text="Session Time: 00:00", font=("Arial", 22), bg="#E8F0F2")
timer_label.place(x=700, y=170)

start_button = tk.Button(window, text="▶ Start New Session", font=("Arial", 18), bg="#28A745", fg="white", width=25, command=lambda: threading.Thread(target=start_detection).start())
start_button.place(x=700, y=230)

quit_button = tk.Button(window, text="⏹ Quit and View Summary", font=("Arial", 18), bg="#DC3545", fg="white", width=25, command=lambda: show_full_summary())
quit_button.place(x=700, y=290)

fig, ax = plt.subplots(figsize=(4, 4))
canvas = FigureCanvasTkAgg(fig, master=window)
canvas.get_tk_widget().place(x=700, y=370)

def play_alert():
    pygame.mixer.music.play()

def cooldown():
    global beep_allowed
    time.sleep(cooldown_seconds)
    beep_allowed = True

def update_timer():
    global session_seconds, running
    while running:
        mins, secs = divmod(session_seconds, 60)
        timer_label.config(text=f"Session Time: {mins:02}:{secs:02}")
        time.sleep(1)
        session_seconds += 1

def update_graph(focus, distract):
    ax.clear()
    labels = ['Focus', 'Distraction']
    values = [focus, distract]
    colors = ['#28A745', '#DC3545']
    ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Focus vs Distraction', fontsize=16)
    canvas.draw()

def update_frame():
    global idx, running, beep_allowed, focus_count, distract_count, session_completed

    idx = 0
    focus_count = 0
    distract_count = 0
    session_completed = False

    while running and idx < len(X_test):
        ret, frame = cap.read()
        if not ret:
            running = False
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        eye_detected = False

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) > 0:
                eye_detected = True
                break

        input_data = X_test.iloc[idx].values.reshape(1, -1)
        eeg_prediction = model.predict(input_data)[0]

        if eeg_prediction == 1 and not eye_detected:
            status_label.config(text="Distracted ❌", fg="#DC3545")
            distract_count += 1
            if beep_allowed:
                threading.Thread(target=play_alert).start()
                beep_allowed = False
                threading.Thread(target=cooldown).start()
        else:
            status_label.config(text="Focused ✅", fg="#28A745")
            focus_count += 1
            beep_allowed = True

        counter_label.config(text=f"Focus: {focus_count} | Distract: {distract_count}")
        update_graph(focus_count, distract_count)

        frame = cv2.resize(frame, (640, 480))
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        idx += 1
        time.sleep(0.1)

    session_completed = True
    cap.release()
    status_label.config(text="Session Complete!")
    save_session(focus_count, distract_count, session_seconds)

def start_detection():
    global running, session_seconds
    cap.open(0)
    running = True
    session_seconds = 0

    threading.Thread(target=update_frame).start()
    threading.Thread(target=update_timer).start()

def save_session(focus, distract, duration):
    session_data.append({
        'Focus': focus,
        'Distraction': distract,
        'Duration': duration
    })
    show_session_summary(focus, distract, duration)

def show_session_summary(focus, distract, duration):
    summary_window = tk.Toplevel(window)
    summary_window.title("Session Summary")
    summary_window.geometry("400x300")

    focus_rate = (focus / (focus + distract)) * 100 if (focus + distract) > 0 else 0

    tk.Label(summary_window, text="Session Completed!", font=("Arial", 18)).pack(pady=10)
    tk.Label(summary_window, text=f"Duration: {duration // 60:02}:{duration % 60:02}", font=("Arial", 14)).pack()
    tk.Label(summary_window, text=f"Focus Count: {focus}", font=("Arial", 14)).pack()
    tk.Label(summary_window, text=f"Distraction Count: {distract}", font=("Arial", 14)).pack()
    tk.Label(summary_window, text=f"Focus Rate: {focus_rate:.2f}%", font=("Arial", 14)).pack()

def show_full_summary():
    global running

    if running and not session_completed:
        confirm = messagebox.askyesno("Session Running", "Session is still running. Do you want to quit?")
        if confirm:
            running = False
            return
        else:
            return

    if len(session_data) == 0:
        messagebox.showinfo("No Sessions", "You haven't completed any sessions yet.")
        return

    summary_window = tk.Toplevel(window)
    summary_window.title("Full Multi-Session Summary")
    summary_window.geometry("600x600")

    total_sessions = len(session_data)
    best_focus_rate = 0
    best_session = None

    for session in session_data:
        focus = session['Focus']
        distract = session['Distraction']
        rate = (focus / (focus + distract)) * 100 if (focus + distract) > 0 else 0
        if rate > best_focus_rate:
            best_focus_rate = rate
            best_session = session

    tk.Label(summary_window, text=f"Total Sessions: {total_sessions}", font=("Arial", 16)).pack(pady=10)

    for i, session in enumerate(session_data):
        focus = session['Focus']
        distract = session['Distraction']
        duration = session['Duration']
        focus_rate = (focus / (focus + distract)) * 100 if (focus + distract) > 0 else 0

        tk.Label(summary_window, text=f"Session {i+1}: Duration {duration // 60:02}:{duration % 60:02}, Focus: {focus}, Distract: {distract}, Focus Rate: {focus_rate:.2f}%", font=("Arial", 12)).pack()

    tk.Label(summary_window, text=f"\nBest Focus Rate: {best_focus_rate:.2f}%", font=("Arial", 14, "bold"), fg="#28A745").pack(pady=10)

window.mainloop()
