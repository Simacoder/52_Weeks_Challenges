from django.test import TestCase
from django.urls import reverse

class PredictorTests(TestCase):
    def test_home_renders(self):
        resp = self.client.get(reverse("home"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Iris Predictor")

    def test_api_predict(self):
        url = reverse("predict_api")
        payload = {
            "sepal_length": 5.0,
            "sepal_width": 3.6,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }
        resp = self.client.post(url, payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("class_name", data)
        self.assertIn("probabilities", data)