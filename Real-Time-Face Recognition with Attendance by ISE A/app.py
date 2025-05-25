from flask import Flask,render_template,make_response,jsonify,request,session,send_file, current_app, flash, redirect, url_for
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from datetime import date
import sqlite3
import json
import pandas as pd
import matplotlib.pyplot as plt
import io
import squarify
import sqlite3
import csv
from flask import Flask, render_template, redirect, url_for, request, flash
import urllib.request
import time
from collections import defaultdict
import time

def test_camera_connection(source):
    try:
        if isinstance(source, str) and (source.startswith('rtsp://') or source.startswith('http://')):
            stream = urllib.request.urlopen(source)
            return True
        else:
            cap = cv2.VideoCapture(source)
            if cap is None or not cap.isOpened():
                return False
            cap.release()
            return True
    except:
        return False

def get_available_camera():
    """
    Get available camera with fallback options and proper error handling
    """
    def try_camera(index):
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_ANY)  # Try any available backend
            if not cap.isOpened():
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Try DirectShow
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cap.release()
                    return index
            cap.release()
        except Exception as e:
            print(f"Error testing camera {index}: {str(e)}")
        return None

    try:
        with open('camera_config.json', 'r') as f:
            config = json.load(f)
            ip_camera_url = config.get('ip_camera_url')
            if ip_camera_url:
                cap = cv2.VideoCapture(ip_camera_url)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        cap.release()
                        return ip_camera_url
                cap.release()
    except Exception as e:
        print(f"Error accessing IP camera: {str(e)}")

    for i in range(-1, 2):  
        result = try_camera(i)
        if result is not None:
            return result

    return None

def initialize_camera(source):
    """
    Initialize camera with proper error handling and settings
    """
    try:
        cap = cv2.VideoCapture(source, cv2.CAP_ANY)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            raise Exception("Failed to open camera")

        if isinstance(source, str):  # IP camera
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
        else:  
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ret, frame = cap.read()
        if not ret:
            raise Exception("Failed to grab frame")

        return cap
    except Exception as e:
        print(f"Error initializing camera: {str(e)}")
        return None

name=" "
app = Flask(__name__)
app.secret_key = "secret_key"

def get_camera_config():
    try:
        with open('camera_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def update_camera_config(ip_camera_url, resolution, fps):
    config = {
        'ip_camera_url': ip_camera_url,
        'resolution': resolution,
        'fps': fps
    }
    with open('camera_config.json', 'w') as f:
        json.dump(config, f)

@app.route('/camera-settings', methods=['GET', 'POST'])
def camera_settings():
    if request.method == 'POST':
        camera_type = request.form.get('camera_type')
        ip_address = request.form.get('ip_address')
        port = request.form.get('port')
        username = request.form.get('username')
        password = request.form.get('password')
        protocol = request.form.get('protocol')
        
        if protocol == 'rtsp':
            if username and password:
                camera_url = f'rtsp://{username}:{password}@{ip_address}:{port}/stream'
            else:
                camera_url = f'rtsp://{ip_address}:{port}/stream'
        else:  
            if username and password:
                camera_url = f'http://{username}:{password}@{ip_address}:{port}/video'
            else:
                camera_url = f'http://{ip_address}:{port}/video'
        
        try:
            update_camera_config(
                ip_camera_url=camera_url,
                resolution={"width": 1280, "height": 720},
                fps=30
            )
            flash('Camera settings updated successfully!', 'success')
        except Exception as e:
            flash(f'Error updating camera settings: {str(e)}', 'error')
        
        return redirect(url_for('camera_settings'))
    
    current_config = get_camera_config()
    return render_template('camera_settings.html', config=current_config)

@app.route('/new', methods=['GET', 'POST'])
def new():
    if request.method=="POST":
        return render_template('index.html')
    else:
        return "Everything is okay!"

@app.route('/name', methods=['GET', 'POST'])
def name():
    if request.method=="POST":
        name1=request.form['name1']
        name2=request.form['name2']

        camera_source = get_available_camera()
        if camera_source is None:
            return 'No camera found'

        cam = cv2.VideoCapture(camera_source)

        while True:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("Press Space to capture image", frame)

            k = cv2.waitKey(1)
            if k%256 == 27:
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                path = os.path.join(current_app.root_path, 'Training images')
                os.makedirs(path, exist_ok=True)
                img_name = f"{name1}.png"
                img_path = os.path.join(path, img_name)
                cv2.imwrite(img_path, frame)
                print(f"{img_name} written to {img_path}")
                break

        cam.release()
        cv2.destroyAllWindows()
        return render_template('image.html')
    else:
        return 'All is not well'

def update_face_tracking(tracked_faces, current_faces, threshold=0.6):
    """
    Update face tracking with temporal smoothing
    tracked_faces: dict of tracked face data
    current_faces: list of current face detections
    """
    current_time = time.time()
    
    for name, face_loc, is_marked in current_faces:
        if name in tracked_faces:
            tracked_faces[name]['locations'].append(face_loc)
            tracked_faces[name]['last_seen'] = current_time
            tracked_faces[name]['is_marked'] = is_marked
            if len(tracked_faces[name]['locations']) > 5:
                tracked_faces[name]['locations'].pop(0)
        else:
            tracked_faces[name] = {
                'locations': [face_loc],
                'last_seen': current_time,
                'is_marked': is_marked
            }
    
    tracked_faces = {
        name: data for name, data in tracked_faces.items()
        if current_time - data['last_seen'] < 1.0
    }
    
    return tracked_faces

def get_smooth_location(locations):
    """Calculate smooth face location from history"""
    if not locations:
        return None
    
    avg_loc = [0, 0, 0, 0]
    weight_sum = 0
    for i, loc in enumerate(locations):
        weight = (i + 1)  
        for j in range(4):
            avg_loc[j] += loc[j] * weight
        weight_sum += weight
    
    return [int(coord / weight_sum) for coord in avg_loc]

@app.route("/", methods=["GET", "POST"])
def recognize():
    if request.method == "POST":
        path = os.path.join(current_app.root_path, 'Training images')
        images = []
        classNames = []
        myList = os.listdir(path)
        print(myList)
        for cl in myList:
            curImg = cv2.imread(os.path.join(path, cl))
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])
        print(classNames)
        
        def findEncodings(images):
            encodeList = []
            for img in images:
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                try:
                    encode = face_recognition.face_encodings(img)[0]
                    encodeList.append(encode)
                except IndexError:
                    print("No face found in image")
                    continue
            return encodeList

        def markData(name, marked_names):
            if name in marked_names:
                return False
            
            print("The Attended Person is ", name)
            now = datetime.now()
            dtString = now.strftime('%H:%M')
            today = date.today()
            
            conn = sqlite3.connect('information.db')
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM Attendance 
                WHERE NAME = ? AND Date = ? AND Time = ?
            """, (name, today, dtString))
            
            if cursor.fetchone()[0] == 0:
                conn.execute('''CREATE TABLE IF NOT EXISTS Attendance
                            (NAME TEXT NOT NULL,
                             Time TEXT NOT NULL,
                             Date TEXT NOT NULL)''')
                conn.execute("INSERT INTO Attendance (NAME,Time,Date) values (?,?,?)", 
                           (name, dtString, today))
                conn.commit()
                print(f"Attendance marked for {name} at {dtString}")
                marked_names.add(name)
                return True
            
            conn.close()
            return False

        def markAttendance(name, marked_names):
            if name in marked_names:
                return
                
            with open('attendance.csv', 'r+', errors='ignore') as f:
                myDataList = f.readlines()
                nameList = []
                for line in myDataList:
                    entry = line.split(',')
                    nameList.append(entry[0])
                if name not in nameList:
                    now = datetime.now()
                    dtString = now.strftime('%H:%M')
                    f.writelines(f'\n{name},{dtString}')

        encodeListKnown = findEncodings(images)
        print('Encoding Complete')
        
        camera_source = get_available_camera()
        if camera_source is None:
            flash("No camera found or camera access denied", "error")
            return redirect(url_for('home'))

        cap = initialize_camera(camera_source)
        if cap is None:
            flash("Failed to initialize camera", "error")
            return redirect(url_for('home'))

        marked_names = set()
        frame_count = 0
        last_frame_time = time.time()
        fps_limit = 30
        
        tracked_faces = {}
        
        face_buffer = defaultdict(int)
        
        try:
            while True:
                current_time = time.time()
                if (current_time - last_frame_time) < 1.0/fps_limit:
                    continue
                last_frame_time = current_time

                ret, img = cap.read()
                if not ret:
                    print("Failed to grab frame, retrying...")
                    cap.release()
                    cap = initialize_camera(camera_source)
                    if cap is None:
                        break
                    continue

                frame_count += 1
                
                display_img = img.copy()
                
                current_faces = []
                if frame_count % 3 == 0:
                    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            
                    facesCurFrame = face_recognition.face_locations(imgS)
                    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
            
                    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.5)
                        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                    
                        if len(faceDis) > 0:
                            matchIndex = np.argmin(faceDis)
                            if matches[matchIndex]:
                                name = classNames[matchIndex].upper()
                                face_buffer[name] += 1
                                
                                if face_buffer[name] >= 5 and name not in marked_names:
                                    if markData(name, marked_names):
                                        markAttendance(name, marked_names)
                                
                                y1, x2, y2, x1 = faceLoc
                                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                                current_faces.append((name, (y1, x2, y2, x1), name in marked_names))
                            else:
                                name = 'Unknown'
                                y1, x2, y2, x1 = faceLoc
                                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                                current_faces.append((name, (y1, x2, y2, x1), False))
                
                tracked_faces = update_face_tracking(tracked_faces, current_faces)
                
                for name, data in tracked_faces.items():
                    smooth_loc = get_smooth_location(data['locations'])
                    if smooth_loc:
                        y1, x2, y2, x1 = smooth_loc
                        
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), 
                                    (0, 255, 0) if data['is_marked'] else (255, 200, 0), 2)
                        
                        cv2.rectangle(display_img, (x1, y2 - 35), (x2, y2), 
                                    (0, 255, 0) if data['is_marked'] else (255, 200, 0), cv2.FILLED)
                        
                        status_text = f"{name} (Marked)" if data['is_marked'] else name
                        font_scale = 0.6
                        font_thickness = 2
                        
                        (text_width, text_height), _ = cv2.getTextSize(
                            status_text, cv2.FONT_HERSHEY_COMPLEX, font_scale, font_thickness)
                        text_x = x1 + 6
                        text_y = y2 - 6
                        
                        cv2.putText(display_img, status_text, (text_x, text_y),
                                  cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 255, 255), font_thickness)
                        
                        if data['is_marked']:
                            cv2.putText(display_img, "✓", (x2 - 25, y1 + 25),
                                      cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                attendance_text = f"Attendance Marked: {len(marked_names)}"
                cv2.putText(display_img, attendance_text, (10, 30),
                          cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('Punch your Attendance', display_img)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        except Exception as e:
            print(f"Error during recognition: {str(e)}")
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()

        return render_template('first.html')
    else:
        return render_template('main.html')


@app.route('/login', methods=['POST'])
def login():
    json_data = json.loads(request.data.decode())
    username = json_data['username']
    password = json_data['password']
    
    df = pd.read_csv('cred.csv')
    if len(df.loc[(df['username'] == username) & (df['password'] == password)]) > 0:
        session['username'] = username
        return jsonify({'message': 'success'})
    else:
        return jsonify({'error': 'Invalid username or password'})

@app.route('/checklogin')
def checklogin():
    if 'username' in session:
        return session['username']
    return 'False'

@app.route('/change_password', methods=['POST'])
def change_password():
    json_data = json.loads(request.data.decode())
    username = json_data['username']
    old_password = json_data['old_password']
    new_password = json_data['new_password']

    df = pd.read_csv('cred.csv')

    if len(df.loc[(df['username'] == username) & (df['password'] == old_password)]) > 0:
        # Update the password in the DataFrame
        df.loc[df['username'] == username, 'password'] = new_password
        # Save the updated DataFrame back to the CSV file
        df.to_csv('cred.csv', index=False)
        return jsonify({'message': 'Password updated successfully'})
    else:
        return jsonify({'error': 'Invalid username or password'})


def verify_admin_password(entered_password):
    with open('cred.csv', mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 2:  
                continue
            
            username, correct_password = row
            
            if entered_password == correct_password:
                return True
    return False

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/reset', methods=['POST'])
def reset_data():
    entered_password = request.form['password']
    
    if verify_admin_password(entered_password):
        try:
            conn = sqlite3.connect('information.db')
            cursor = conn.cursor()

            # Reset all data in the Attendance table
            cursor.execute("DELETE FROM Attendance")
            
            # Commit changes and close the connection
            conn.commit()
            conn.close()

            return jsonify({"status": "success", "message": "Data has been reset successfully."})
        except Exception as e:
            return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"})
    else:
        return jsonify({"status": "error", "message": "Incorrect password."})
@app.route('/how',methods=["GET","POST"])
def how():
    return render_template('form.html')
@app.route('/data',methods=["GET","POST"])
def data():
    '''user=request.form['username']
    pass1=request.form['pass']
    if user=="tech" and pass1=="tech@321" :
    '''
    if request.method=="POST":
        today=date.today()
        print(today)
        conn = sqlite3.connect('information.db')
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        print ("Opened database successfully");
        cursor = cur.execute("SELECT DISTINCT NAME,Time, Date from Attendance where Date=?",(today,))
        rows=cur.fetchall()
        print(rows)
        for line in cursor:

            data1=list(line)
        print ("Operation done successfully");
        conn.close()

        return render_template('form2.html',rows=rows)
    else:
        return render_template('form1.html')


            
@app.route('/whole', methods=["GET", "POST"])
def whole():
    """Render the table with all attendance data"""
    conn = sqlite3.connect('information.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = "SELECT DISTINCT NAME, Time, Date FROM Attendance"
    rows = cur.execute(query).fetchall()
    conn.close()

    return render_template('form3.html', rows=rows)


@app.route('/download_excel', methods=["GET"])
def download_excel():
    try:
        conn = sqlite3.connect('information.db')

        query = "SELECT DISTINCT NAME, Time, Date FROM Attendance"
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return "No attendance records found for today.", 404

        file_path = os.path.abspath("attendance.xlsx")
        df.to_excel(file_path, index=False)
        print(f"Excel file generated at: {file_path}")

        return send_file(file_path, as_attachment=True)

    except Exception as e:
        print(f"Error occurred: {e}")
        return f"An error occurred: {e}", 500

@app.route('/download_today_excel', methods=["GET"])
def download_today_excel():
    try:
        today = date.today().strftime('%Y-%m-%d')

        conn = sqlite3.connect('information.db')

        query = "SELECT DISTINCT NAME, Time, Date FROM Attendance WHERE Date = ?"
        df = pd.read_sql_query(query, conn, params=(today,))
        conn.close()

        if df.empty:
            return "No attendance records found for today.", 404

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name="Today's Attendance")
        output.seek(0)

        response = make_response(output.read())
        response.headers["Content-Disposition"] = "attachment; filename=today_attendance.xlsx"
        response.headers["Content-Type"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        return response

    except Exception as e:
        print(f"Error: {e}")
        return f"An error occurred: {e}", 500


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    """Render the dashboard with graphs."""
    conn = sqlite3.connect('information.db')
    
    query_all = "SELECT DISTINCT NAME, Time, Date FROM Attendance"
    df_all = pd.read_sql_query(query_all, conn)
    
    query_today = "SELECT DISTINCT NAME, Time, Date FROM Attendance WHERE Date = DATE('now')"
    df_today = pd.read_sql_query(query_today, conn)
    
    conn.close()  # Close connection after all queries are executed

    graph1_path = None
    graph2_path = None
    treemap_path = None
    pie_chart_path = None

    if not df_all.empty:
        attendance_counts = df_all['NAME'].value_counts()
        plt.figure(figsize=(10, 6))
        attendance_counts.plot(kind='bar', color='skyblue')
        plt.title('Attendance Count Per Person')
        plt.xlabel('Name')
        plt.ylabel('Attendance Count')
        plt.xticks(rotation=45)
        graph1_path = 'static/attendance_count.png'
        plt.savefig(graph1_path, bbox_inches='tight', transparent=True)  # Ensure transparent background
        plt.close()

        df_all['Date'] = pd.to_datetime(df_all['Date'])
        attendance_trend = df_all.groupby('Date').size()
        plt.figure(figsize=(10, 6))
        attendance_trend.plot(kind='line', marker='o', color='green')
        plt.title('Attendance Trend Over Time')
        plt.xlabel('Date')
        plt.ylabel('Attendance Count')
        graph2_path = 'static/attendance_trend.png'
        plt.savefig(graph2_path, bbox_inches='tight', transparent=True)  # Ensure transparent background
        plt.close()

        names = attendance_counts.index
        counts = attendance_counts.values
        colors = plt.cm.Paired.colors[:len(names)]  # Assign distinct colors for each name

        plt.figure(figsize=(10, 6))
        squarify.plot(sizes=counts, label=names, color=colors, alpha=0.7)
        plt.title('Attendance Distribution (Treemap)')
        treemap_path = 'static/attendance_treemap.png'
        plt.savefig(treemap_path, bbox_inches='tight', transparent=True)  # Ensure transparent background
        plt.close()

    if not df_today.empty:
        present_today_count = len(df_today['NAME'].unique())
        absent_today_count = len(df_all['NAME'].unique()) - present_today_count

        pie_data = [present_today_count, absent_today_count]
        pie_labels = ['Present Today', 'Absent Today']
        
        plt.figure(figsize=(7, 7))
        plt.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=['#42A5F5', '#FF5733'])
        plt.title('Today\'s Attendance (Present vs Absent)')
        pie_chart_path = 'static/todays_attendance_pie.png'
        plt.savefig(pie_chart_path, bbox_inches='tight', transparent=True)  # Ensure transparent background
        plt.close()

    return render_template('dashboard.html', graph1=graph1_path, graph2=graph2_path, pie_chart=pie_chart_path, treemap=treemap_path)


def create_students_table():
    conn = sqlite3.connect('information.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS students
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     name TEXT NOT NULL,
                     reg_id TEXT NOT NULL UNIQUE)''')
    conn.close()

if __name__ == '__main__':
    create_students_table()
    app.run(debug=True)