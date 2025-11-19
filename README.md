# HW2 - Machine Learning Algorithms Implementation

這份儲存庫 (Repository) 包含了機器學習作業二 (HW2) 的程式碼實作。主要使用多種機器學習演算法針對 **Adult** 與 **Boston** 資料集進行分析、模型訓練與評估，並包含模型可解釋性分析。

## 📂 檔案結構說明

本專案使用 Jupyter Notebook 進行實作，各檔案功能如下：

| 檔案名稱 | 說明 |
| :--- | :--- |
| **01_knn_adult.ipynb** | 使用 **K-Nearest Neighbors (KNN)** 演算法對 Adult 資料集進行分析與績效指標評估。 |
| **02_svr_adult.ipynb** | 使用 **Support Vector Regression (SVR)** 演算法對 Adult 資料集進行分析與績效指標評估。 |
| **03_rf_adult.ipynb** | 使用 **Random Forest (隨機森林)** 演算法對 Adult 資料集進行分析與績效指標評估。 |
| **04_xgboost_adult.ipynb** | 使用 **XGBoost** 演算法對 Adult 資料集進行分析與績效指標評估。 |
| **05_xgboost_boston_cv.ipynb** | 針對 Boston 房價資料集使用 **XGBoost**，並結合 **Cross-Validation (交叉驗證)** 進行模型評估。 |
| **06_shap_analysis.ipynb** | 使用 **SHAP (SHapley Additive exPlanations)** 套件進行模型的可解釋性分析，探討特徵重要性。 |
| **data/** | 存放專案所需的資料集檔案。 |

## 🛠️ 使用工具與套件

本專案主要依賴以下 Python 套件：

* **Python 3.x**
* **Scikit-learn** (KNN, SVR, Random Forest)
* **XGBoost** (Gradient Boosting)
* **SHAP** (Model Interpretability)
* **Pandas / NumPy** (Data Processing)
* **Matplotlib / Seaborn** (Visualization)

## 🚀 如何執行

1.  確保已安裝上述必要套件：
    ```bash
    pip install scikit-learn xgboost shap pandas numpy matplotlib seaborn
    ```
2.  開啟 Jupyter Notebook：
    ```bash
    jupyter notebook
    ```
3.  依序執行各個 `.ipynb` 檔案即可重現實驗結果。