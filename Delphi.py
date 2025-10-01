# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import arabic_reshaper
from bidi.algorithm import get_display
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# تابع اصلاح متن فارسی 
def fix_text(text):
    try:
        if pd.isna(text) or text == "":
            return text
            
        text_str = str(text)
        # بررسی آیا متن حاوی کاراکتر فارسی است
        if any('\u0600' <= ch <= '\u06FF' for ch in text_str):
            reshaped = arabic_reshaper.reshape(text_str)
            return get_display(reshaped)
        else:
            return text_str  # انگلیسی بدون تغییر
    except:
        return str(text)

def setup_fonts_by_language(language):
    """تنظیم فونت بر اساس زبان"""
    # پاکسازی تنظیمات فعلی
    plt.rcParams.update(plt.rcParamsDefault)
    
    if language == "persian":
        # فونت‌های فارسی
        rcParams['font.sans-serif'] = ['B Nazanin', 'Tahoma', 'DejaVu Sans']
        rcParams['axes.unicode_minus'] = False
    else:
        # فونت‌های انگلیسی/بین‌المللی - استفاده از فونت پیشفرض
        rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
        rcParams['axes.unicode_minus'] = True

def detect_language(df):
    """تشخیص زبان داده‌ها"""
    try:
        sample_text = str(df["Criteria"].iloc[0]) if len(df) > 0 else ""
        if any('\u0600' <= ch <= '\u06FF' for ch in sample_text):
            return "persian"
        else:
            return "english"
    except:
        return "english"

def compute_importance_and_weight(df, total_respondents):
    column_to_xi = {
        "Much lower importance": 1,
        "Low importance": 3,
        "Medium importance": 5,
        "Higher importance": 7,
        "Much higher importance": 9
    }

    column_to_yi = {
        "Much lower importance": 0.04,
        "Low importance": 0.12,
        "Medium importance": 0.20,
        "Higher importance": 0.28,
        "Much higher importance": 0.36
    }

    importance_percentages = []
    weight_scores = []
    raw_weights = []

    for _, row in df.iterrows():
        Zi_sum = 0
        Xi_sum = 0
        for col in column_to_yi.keys():
            count = row[col]
            Zi_sum += count * column_to_yi[col]
            Xi_sum += count * column_to_xi[col]

        importance = (Zi_sum / total_respondents) * 100
        weight_score = Xi_sum / total_respondents
        raw_weight = (importance / 100) * weight_score

        importance_percentages.append(importance)
        weight_scores.append(weight_score)
        raw_weights.append(raw_weight)

    df["Importance_percent"] = importance_percentages
    df["Weight_score"] = weight_scores
    df["Raw_weight"] = raw_weights

    total_raw = sum(raw_weights)
    df["Normalized_weight"] = df["Raw_weight"] / total_raw

    result = df[["Criteria", "Weight_score", "Importance_percent", "Normalized_weight"]]
    return result

def main():
    file_path = input("Enter Excel file path: ").strip()
    df = pd.read_excel(file_path)

    total_respondents = int(input("Enter number of respondents: "))

    # تشخیص زبان و تنظیم فونت
    language = detect_language(df)
    print(f"Detected language: {language}")
    setup_fonts_by_language(language)

    result = compute_importance_and_weight(df, total_respondents)

    # ذخیره در اکسل
    out_path = file_path.replace(".xlsx", "_results.xlsx")
    result.to_excel(out_path, index=False)
    print("✅ Output saved to:", out_path)

    # نمودار 1:
    result_sorted = result.sort_values(by="Normalized_weight", ascending=False)

    # ایجاد نمودار جدید با تنظیمات فونت
    plt.figure(figsize=(12, 7))
    
    if language == "persian":
        labels = [fix_text(c) for c in result_sorted["Criteria"]]
        ylabel = fix_text("ضریب اهمیت نرمالایز شده")
        title = fix_text("مقایسه معیارها بر اساس ضریب اهمیت نرمالایز شده")
    else:
        labels = [str(c) for c in result_sorted["Criteria"]]
        ylabel = "Normalized Importance Coefficient"
        title = "Criteria Comparison based on Normalized Importance Coefficient"
    
    # استفاده از شاخص برای موقعیت‌های بار
    x_pos = range(len(labels))
    bars = plt.bar(x_pos, result_sorted["Normalized_weight"], color="skyblue", alpha=0.7)
    
    plt.xticks(x_pos, labels, rotation=45, ha="right")
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()

    chart_path = file_path.replace(".xlsx", "_chart.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Bar chart saved to:", chart_path)

    # نمودار 2
    fig, ax = plt.subplots(figsize=(14, 8))
    
    if language == "persian":
        xlabel = fix_text('درصد اهمیت معیار')
        ylabel = fix_text('درجه اهمیت')
        chart_title = fix_text('نمودار دلفی - اهمیت معیارها')
        legend_title = fix_text('معیارها')
    else:
        xlabel = 'Importance Percentage'
        ylabel = 'Importance Degree'
        chart_title = 'Delphi Chart - Criteria Importance'
        legend_title = 'Criteria'

    colors = plt.cm.tab20(np.linspace(0, 1, len(result)))
    
    for i, row in result.iterrows():
        label_text = f"{i+1}: {fix_text(row['Criteria']) if language == 'persian' else row['Criteria']}"
        ax.scatter(row['Importance_percent'], row['Weight_score'],
                   marker='o', s=120, color=colors[i % len(colors)],
                   label=label_text)
        ax.annotate(i+1, (row['Importance_percent'], row['Weight_score']),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=10, color='black')

    # خطوط راهنما
    half_x = result["Importance_percent"].max() / 2
    median_y = 5
    ax.axvline(x=half_x, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=median_y, color='black', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(chart_title, fontsize=16, pad=20)
    
    # قراردادن لیجن در خارج از نمودار
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
               title=legend_title, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    scatter_path = file_path.replace(".xlsx", "_scatter.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Scatter plot saved to:", scatter_path)

if __name__ == "__main__":
    main()