#app.py
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Login password
CORRECT_PASSWORD = "5555"

# Route for the login page
@app.route('/')
def login():
    return render_template('login.html', error=None)

# Route to handle login form submission
@app.route('/login', methods=['POST'])
def login_submit():
    password_entered = request.form.get('password')

    if password_entered == CORRECT_PASSWORD:
        return redirect(url_for('home'))  # Redirect to home page on successful login
    else:
        error = "Incorrect password. Please try again."
        return render_template('login.html', error=error)

# Route for the home automation control page
@app.route('/home')
def home():
    return render_template('home_automation.html')


# Route to control lights
@app.route('/lights')
def lights():
    return render_template('lights.html')

# Route to control fans
@app.route('/fans')
def fans():
    return render_template('fans.html')

# Route to control AC
@app.route('/ac')
def ac():
    return render_template('ac.html')

# Control logic (simulated)
@app.route('/control/<device>/<action>')
def control(device, action):
    if device == 'lights':
        if action == 'on':
            # Code to turn lights on
            return 'Lights turned ON'
        elif action == 'off':
            # Code to turn lights off
            return 'Lights turned OFF'
    elif device == 'fans':
        if action == 'on':
            # Code to turn fans on
            return 'Fans turned ON'
        elif action == 'off':
            # Code to turn fans off
            return 'Fans turned OFF'
    elif device == 'ac':
        if action == 'on':
            # Code to turn AC on
            return 'AC turned ON'
        elif action == 'off':
            # Code to turn AC off
            return 'AC turned OFF'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)