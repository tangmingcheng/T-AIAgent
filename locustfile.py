from locust import HttpUser, task, between
import uuid

class AgentUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def send_task(self):
        task_text = "你是谁？"
        session_task = {"task": f"{task_text} - {uuid.uuid4()}"}
        self.client.post("/process_task", json=session_task)
