from flask import Flask, jsonify
import sqlite3

app = Flask(__name__)

# Route to clear the database
@app.route('/clear_database', methods=['GET'])
def clear_database():
    try:
        # Step 3: Connect to the SQLite Database
        conn = sqlite3.connect('information.db')
        cursor = conn.cursor()

        # Step 4: Execute SQL Command to Clear Data
        cursor.execute('DELETE FROM Attendance')  # Adjust table name as needed
        
        # Step 5: Commit the Transaction
        conn.commit()

        return 'Database cleared successfully'
    except sqlite3.Error as e:
        return f'Error clearing database: {e}'
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    app.run(debug=True)
