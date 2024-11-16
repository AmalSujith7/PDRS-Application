from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from phishing_detector import detect_phishing
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from transformers import logging as transformers_logging
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
transformers_logging.set_verbosity_error()
app = Flask(__name__)
CORS(app)
SENDER_EMAIL = "aerobottechlabs@gmail.com"
SENDER_PASSWORD = "ucqssewavrhcilaz"
RECEIVER_EMAIL = "amalsmenon7@gmail.com"  
def send_notification_email(email_content):
    """Function to send a notification email to the fixed recipient."""
    try:
        subject = "Phishing Alert Detected in Your Email"
        body = f"""\
        Dear User,

        Our system detected a potential phishing attempt in the following email content:

        {email_content}

        Please do not respond to this email and avoid clicking any links in the original message.

        Best Regards,
        Phishing Detection System
        """
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        
        print("Notification email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form.get('email_content')

    if not email_text:
        return jsonify({"error": "Email content is required"}), 400
    prediction = detect_phishing(email_text)
    result = "Phishing Detected" if prediction == 1 else "Not Phishing"
    if prediction == 1:
        send_notification_email(email_text)

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
