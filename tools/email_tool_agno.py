import os
import smtplib
import json
from email.message import EmailMessage
from agno.tools import Toolkit
from agno.utils.log import logger


class SendEmailTools(Toolkit):
    def __init__(self):
        super().__init__(name="send_email_tool")
        self.register(self.send_email)

    def send_email(self, to: str, subject: str, body: str) -> str:
        """Send an email using SMTP and return the status as a JSON response.

        Args:
            to (str): Recipient email address.
            subject (str): Email subject.
            body (str): Email body content.

        Returns:
            str: JSON response indicating success or failure.
        """
        SMTP_SERVER = "smtp.gmail.com"  # Modify based on your SMTP provider
        SMTP_PORT = 465
        SMTP_USER = os.getenv('SMTP_USER')  # Sender's email
        SMTP_PASS = os.getenv('SMTP_PASS')  # App password

        logger.info(f"Attempting to send email to {to}")

        try:
            msg = EmailMessage()
            msg["From"] = SMTP_USER
            msg["To"] = to
            msg["Subject"] = subject
            msg.set_content(body)

            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
                server.login(SMTP_USER, SMTP_PASS)
                server.send_message(msg)

            logger.info(f"Email successfully sent to {to}")
            return json.dumps({"status": "success", "message": f"Email sent to {to}"})
        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            return json.dumps({"status": "error", "message": str(e)})


# Example usage
if __name__ == "__main__":
    email_tool = SendEmailTools()
    response = email_tool.send_email("test@example.com", "Test Subject", "Hello, this is a test email.")
    print(response)
