# 🌸 Machine Learning on Django with Simanga Mchunu

A complete **end-to-end machine learning web application** built using **Django** and **Scikit-learn**.  
This project trains a **Random Forest classifier** on the famous **Iris dataset**, serves predictions via a web form & API,  
and includes an interactive **data analysis dashboard** with plots and feature statistics.

---

## 📌 Features

✅ **Train & Save Model** – Random Forest classifier trained on the Iris dataset  
✅ **Django Web Form** – Users can input flower measurements and get predictions  
✅ **REST API Endpoint** – `/api/predict/` accepts JSON and returns predictions  
✅ **Interactive Analysis Page** – Displays dataset statistics, class distribution, and plots  
✅ **Modern UI** – Clean, centered layout with navigation bar for seamless switching  
✅ **Automated Tests** – Ensures predictions and web pages work as expected  

---

## 🛠️ Tech Stack

- **Backend:** Django, Python
- **Machine Learning:** scikit-learn, joblib
- **Data Analysis:** pandas, numpy, matplotlib, seaborn
- **Frontend:** HTML5, CSS (custom styles)
- **Testing:** Django's built-in test framework

---

## 🚀 Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/simacoder/52_weeks_challenges.git
cd 52_weeks_challenges
cd week_37
```
Create a virtual environment & activate it

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
Install dependencies

```bash
pip install -r requirements.txt
```
Train the model & generate statistics

```bash
python train.py
```
Apply database migrations

```bash
python manage.py migrate
```
Run the development server

```bash
python manage.py runserver
```
**Open in browser**

Web Form (Prediction): http://127.0.0.1:8000/

Dataset Analysis: http://127.0.0.1:8000/analysis/

# 🔗 API Usage

You can send a POST request to the API endpoint:

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```
Sample Response:

```bash
{
  "class_index": 0,
  "class_name": "setosa",
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  }
}
```
# 🧪 Running Tests
This project includes basic unit tests for the homepage and API.

```bash
python manage.py test
```
# 📊 Analysis Dashboard

**The /analysis/ page shows:**

- Feature-wise statistics (mean, std, min, max)

- Class distribution counts

**Visualizations:**

- Pairplot of features

- Correlation heatmap

# 📁 Project Structure

```bash
django-ml-app/
├── mlapp/                 # Django project settings
├── predictor/             # App: ML services, forms, views
│   ├── model/             # Saved model + stats
│   ├── templates/
│   │   └── predictor/
│   │       ├── predict_form.html
│   │       └── analysis.html
│   ├── static/
│   │   └── predictor/plots/  # Generated plots
│   ├── services.py        # Model loading + prediction
│   ├── views.py           # Web & API views
│   ├── forms.py           # Iris input form
│   └── tests.py           # Unit tests
├── train.py               # Model training + analysis script
├── requirements.txt
└── manage.py
```
# 🎯 Future Improvements
- Add user authentication & save prediction history

- Deploy to Docker + Gunicorn + Nginx

- Extend to other datasets or custom ML models

- AJAX API calls for instant predictions without reloading page

# 👨‍💻 Author

**Simanga Mchunu**
Passionate about machine learning, Django development, and creating educational ML projects.

# 📜 License

This project is licensed under the MIT License – feel free to use and modify it.

🌟 Acknowledgements
[scikit-learn](https://scikit-learn.org/stable/)– for the Iris dataset and ML utilities

[Django](https://www.djangoproject.com/) – for the web framework

[KDnuggets](https://www.kdnuggets.com/building-machine-learning-application-with-django) Tutorial – inspiration for this project