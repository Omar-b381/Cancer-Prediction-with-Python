
# 🧪 مشروع التنبؤ بسرطان الثدي باستخدام خوارزميات تعلم الآلة

هذا المشروع يهدف إلى بناء نموذج تعلم آلة يمكنه التنبؤ بإصابة المريض بسرطان الثدي بناءً على بيانات طبية مأخوذة من صور الأشعة.

## 📂 هيكل المشروع

```
ML-project/
├── data/                    # ملفات البيانات (raw, processed, interim)
├── notebooks/              # دفاتر Jupyter للتجربة والتحليل
├── reports/                # التقارير والرسومات الناتجة
├── cancer_prediction_with_python/
│   ├── dataset.py          # تحميل وتنظيف البيانات
│   ├── plots.py            # التصوير البياني
│   └── modeling/
│       └── train.py        # تدريب النماذج وضبط المعاملات
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 📊 وصف البيانات

- **المصدر**: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- يحتوي على 569 عينة، و30 ميزة مشتقة من صور الأشعة.
- المتغير المستهدف `diagnosis`:
  - `M` = ورم خبيث
  - `B` = ورم حميد

## 🎯 هدف المشروع

تطوير نموذج دقيق يمكنه تصنيف الأورام إلى حميدة أو خبيثة بناءً على الخصائص الفيزيائية للخلايا.

## ⚙️ خطوات العمل

1. تحميل البيانات من Kaggle وتنظيفها.
2. استكشاف البيانات بصريًا لفهم التوزيعات والأنماط.
3. تجهيز الميزات وتقسيم البيانات إلى تدريب واختبار.
4. تدريب نموذج Decision Tree مع ضبط المعاملات باستخدام Grid Search.
5. تقييم النموذج على بيانات الاختبار.

## 📈 الأداء

- تم استخدام خوارزمية Decision Tree.
- تم استخدام `GridSearchCV` لضبط المعاملات.
- دقة التصنيف على مجموعة الاختبار تم تقييمها باستخدام accuracy score.

## 🛠️ المتطلبات

```
pandas
matplotlib
seaborn
scikit-learn
```

لتثبيت الحزم:

```bash
pip install -r requirements.txt
```

## 👨‍💻 كيف تبدأ

```bash
git clone https://github.com/username/breast-cancer-prediction
cd breast-cancer-prediction
jupyter notebook
```

## 🧠 نماذج مستقبلية

- تجربة خوارزميات أخرى مثل:
  - Random Forest
  - Logistic Regression
  - SVM
- استخدام تقنيات تحسين الميزات Feature Selection.

## 📝 الرخصة

هذا المشروع مفتوح المصدر تحت رخصة MIT.
