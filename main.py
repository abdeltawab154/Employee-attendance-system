import cv2
import face_recognition
import numpy as np
import datetime
import mysql.connector
from PIL import Image, ImageTk
import tkinter as tk
import os
import threading
from ttkbootstrap import Style

# MySQL database connection configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'attendance_db'
}

# Directory path for employee images
IMAGE_DIR = r"C:\Users\DOMAIN\Desktop\path"

# Global variables
video_display = None
attendance_log = {}
display_timer = None
shutdown_event = threading.Event()
known_encodings = []
known_employee_names = []

# Initialize MySQL connection
try:
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(buffered=True)
    print("Connected to MySQL database.")
except mysql.connector.Error as e:
    print(f"Error connecting to MySQL database: {e}")
    exit(1)

def load_known_encodings():
    """Load known encodings and employee names from the image directory."""
    global known_encodings, known_employee_names

    for filename in os.listdir(IMAGE_DIR):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            employee_name = os.path.splitext(filename)[0]
            image_path = os.path.join(IMAGE_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_employee_names.append(employee_name)
            else:
                print(f"No face found in image: {filename}")

def capture_video():
    """Initialize the video capture from the default webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return None
    
    # Adjust capture properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Adjust width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Adjust height
    cap.set(cv2.CAP_PROP_FPS, 30)  # Adjust FPS
    
    return cap

def detect_landmarks(frame):
    """Detect facial landmarks using face_recognition library."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)
    
    if not face_landmarks_list:
        return None, None

    landmarks = face_landmarks_list[0]
    return face_locations[0], landmarks

def draw_spiderweb(frame, landmarks):
    """Draw a spiderweb-like structure around the face using facial landmarks."""
    spiderweb_color = (230, 204, 128)  # RGB color for spiderweb
    mask = np.zeros_like(frame)

    def draw_lines(points):
        """Helper function to draw lines between facial landmarks."""
        if len(points) > 1:
            for i in range(0, len(points) - 1, 2):
                cv2.line(mask, points[i], points[i + 1], spiderweb_color, 1)
            cv2.line(mask, points[-1], points[0], spiderweb_color, 1)

    draw_lines(landmarks['chin'])
    draw_lines(landmarks['left_eyebrow'])
    draw_lines(landmarks['right_eyebrow'])
    draw_lines(landmarks['nose_bridge'])
    draw_lines(landmarks['nose_tip'])
    draw_lines(landmarks['left_eye'])
    draw_lines(landmarks['right_eye'])
    draw_lines(landmarks['top_lip'])
    draw_lines(landmarks['bottom_lip'])

    alpha = 0.5  # Transparency factor
    cv2.addWeighted(mask, alpha, frame, 1 - alpha, 0, frame)

def fetch_employee_info(employee_name):
    """Fetch employee info from the database."""
    try:
        cursor.execute("SELECT id, name, job_title, address FROM employees WHERE name = %s", (employee_name,))
        employee_info = cursor.fetchone()
        return employee_info
    except mysql.connector.Error as e:
        print(f"Error fetching employee info: {e}")
        return None

def register_new_employee(employee_name):
    """Register a new employee in the database."""
    try:
        cursor.execute("INSERT INTO employees (name, job_title, address) VALUES (%s, %s, %s)", (employee_name, "Unknown", "Unknown"))
        conn.commit()
        print(f"New employee registered: {employee_name}")
    except mysql.connector.Error as e:
        print(f"Error registering new employee: {e}")

def log_attendance(employee_id):
    """Log attendance for the given employee."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO attendance (employee_id, timestamp) VALUES (%s, %s)", (employee_id, timestamp))
        conn.commit()
        print(f"Attendance logged for employee ID: {employee_id}")
    except mysql.connector.Error as e:
        print(f"Error logging attendance: {e}")

def fetch_last_attendance_time(employee_id):
    """Fetch the last attendance time for the given employee."""
    try:
        cursor.execute("SELECT timestamp FROM attendance WHERE employee_id = %s ORDER BY timestamp DESC LIMIT 1", (employee_id,))
        attendance_time = cursor.fetchone()
        return attendance_time[0] if attendance_time else None
    except mysql.connector.Error as e:
        print(f"Error fetching last attendance time: {e}")
        return None

def display_employee_info(employee_name):
    """Display employee info and image in Tkinter window for 10 seconds."""
    global display_timer
    employee_info = fetch_employee_info(employee_name)
    
    if employee_info:
        employee_id, employee_name, job_title, address = employee_info
        employee_name_label.config(text=f"Employee Name: {employee_name}", font=("Helvetica", 14, 'bold'))
        employee_id_label.config(text=f"Employee ID: {employee_id}", font=("Helvetica", 12))
        job_title_label.config(text=f"Job Title: {job_title}", font=("Helvetica", 12))
        address_label.config(text=f"Address: {address}", font=("Helvetica", 12))

        last_attendance_time = fetch_last_attendance_time(employee_id)
        if last_attendance_time:
            last_attendance_time_label.config(text=f"Last Attendance: {last_attendance_time}", font=("Helvetica", 12))
        else:
            last_attendance_time_label.config(text="Last Attendance: N/A", font=("Helvetica", 12))

        image_path = os.path.join(IMAGE_DIR, f"{employee_name}.jpg")
        
        try:
            if os.path.exists(image_path):
                img = Image.open(image_path)
            else:
                raise FileNotFoundError(f"No image found for employee: {employee_name}")
        except FileNotFoundError:
            img = Image.open(os.path.join(IMAGE_DIR, "free.jpg"))
        
        img = img.resize((150, 150), Image.BICUBIC)  # Changed from Image.ANTIALIAS to Image.BICUBIC
        img = ImageTk.PhotoImage(img)
        employee_image_label.configure(image=img, text="")
        employee_image_label.image = img

        if employee_name not in attendance_log:
            log_attendance(employee_id)
            attendance_log[employee_name] = True
        
        # Schedule reset of displayed info after 10 seconds
        if display_timer:
            root.after_cancel(display_timer)
        display_timer = root.after(10000, reset_displayed_info)  # 10000 ms = 10 seconds
    else:
        print(f"No information found for employee: {employee_name}")

def reset_displayed_info():
    """Reset displayed employee info and image."""
    employee_name_label.config(text="Employee Name:", font=("Helvetica", 14, 'bold'))
    employee_id_label.config(text="Employee ID:", font=("Helvetica", 12))
    job_title_label.config(text="Job Title:", font=("Helvetica", 12))
    address_label.config(text="Address:", font=("Helvetica", 12))
    last_attendance_time_label.config(text="Last Attendance:", font=("Helvetica", 12))
    employee_image_label.configure(image=None, text="")

def capture_and_process_frame():
    """Capture video from webcam and process each frame."""
    global video_display
    
    cap = capture_video()

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    print("Press 'q' to quit.")

    while cap.isOpened() and not shutdown_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        face_location, landmarks = detect_landmarks(frame)
        
        if face_location:
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            if landmarks:
                draw_spiderweb(frame, landmarks)
                
                # Recognize employee or register new employee
                employee_name = recognize_or_register_employee(frame)
                
                if employee_name:
                    display_employee_info(employee_name)
        else:
            # Display default image when no face is detected
            img = Image.open(os.path.join(IMAGE_DIR, "free.jpg"))
            img = img.resize((150, 150), Image.BICUBIC)
            img = ImageTk.PhotoImage(img)
            employee_image_label.configure(image=img, text="")
            employee_image_label.image = img

        # Convert frame to RGB format for displaying in Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(image=frame)
        
        if video_display is None:
            video_display = tk.Label(root, image=frame)
            video_display.image = frame
            video_display.pack(fill=tk.BOTH, expand=True)
        else:
            video_display.configure(image=frame)
            video_display.image = frame

        root.update_idletasks()
        root.update()

    cap.release()

def recognize_or_register_employee(frame):
    """Recognize or register employee based on face recognition."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding)
        face_distances = face_recognition.face_distance(known_encodings, encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            return known_employee_names[best_match_index]
    
    # If no match found, register as a new employee
    if face_encodings:
        register_new_employee("New Employee")
        load_known_encodings()  # Reload known encodings after registering new employee
        return "New Employee"

    return None

def on_closing():
    """Function to handle window closing."""
    global shutdown_event
    shutdown_event.set()
    root.destroy()

# Load known face encodings and employee names
load_known_encodings()

# Create Tkinter window
root = tk.Tk()
root.title("Employee Attendance System")
root.geometry("1000x800")
root.configure(bg='#263238')

# Define styles
style = Style(theme='flatly')

# Employee information display
employee_info_frame = tk.Frame(root, padx=20, pady=20, bg='#263238')
employee_info_frame.pack(pady=20)

employee_name_label = tk.Label(employee_info_frame, text="Employee Name:", font=("Helvetica", 14, 'bold'), bg='#263238', fg='#ffffff')
employee_name_label.grid(row=0, column=0, sticky='w')

employee_id_label = tk.Label(employee_info_frame, text="Employee ID:", font=("Helvetica", 12), bg='#263238', fg='#ffffff')
employee_id_label.grid(row=1, column=0, sticky='w')

job_title_label = tk.Label(employee_info_frame, text="Job Title:", font=("Helvetica", 12), bg='#263238', fg='#ffffff')
job_title_label.grid(row=2, column=0, sticky='w')

address_label = tk.Label(employee_info_frame, text="Address:", font=("Helvetica", 12), bg='#263238', fg='#ffffff')
address_label.grid(row=3, column=0, sticky='w')

last_attendance_time_label = tk.Label(employee_info_frame, text="Last Attendance:", font=("Helvetica", 12), bg='#263238', fg='#ffffff')
last_attendance_time_label.grid(row=4, column=0, sticky='w')

employee_image_label = tk.Label(employee_info_frame, bg='#263238')
employee_image_label.grid(row=0, column=1, rowspan=5, padx=20)

# Start video capture and processing in a separate thread
video_thread = threading.Thread(target=capture_and_process_frame)
video_thread.start()

# Bind closing event
root.protocol("WM_DELETE_WINDOW", on_closing)

# Run the main Tkinter loop
root.mainloop()

# Cleanup
conn.close()
