# student-dropout-predictor

## ปัญหาและที่มา
ระบบการศึกษาออนไลน์มีอัตราการ dropout สูง การทำนายล่วงหน้าว่า
นักเรียนคนไหนมีความเสี่ยงช่วยให้สถาบันเข้าไปช่วยได้ทันท่วงที

## Dataset
- 5,000 นักเรียน, 10 features, 3 กลุ่มเป้าหมาย (active/at-risk/dropped)
- ไม่มี missing values, class ค่อนข้าง balanced

## วิธีรัน
pip install -r requirements.txt |
streamlit run app/app.py

## App URL
https://student-dropout-predictor-rwpbfeqy8fvwfuk36syol4.streamlit.app/

## ผลลัพธ์โมเดล
Macro F1: 0.998 % | Accuracy: 0.998 %
