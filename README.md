 **Project Overview: AI-Driven Optimisation in Iron Ore Flotation**

📊 **Scale of the problem**
<br>
* The mining industry produces roughly **12.7 billion metric tons of tailings per year across all minerals**. 
* Specifically for **iron ore, around 1.4 billion tons** of tailings are generated annually 
* With average iron content in tailings at ≈11%, that means about **154 million tons of iron go unrecovered each year**
* **Source: en.wikipedia.org**

<br>

**💰 Lost economic value**

* Iron ore prices vary, but a reasonable mid-2025 average is around $100 per ton.
* 154 million tons × $100 ≈ **$15.4 billion lost annually just in iron content**.
* This **doesn't include other valuable metals** (copper, nickel, cobalt, etc.) **or downstream losses** from lower concentrate quality.
* In **North America alone**, the metals **lost** in tailings are valued at **around $20 billion per year**.

**✏️ Implications for the project**
<br>

**If using AI to automate amine dosing can recover even a fraction of that lost iron—say 5 %** of the 154 million tons (~7.7 Mt)—it could **translate to $770 million in recovered value annually** (at $100/t). 

<br>

🔍 **This project leverages AI to reduce those losses.**

I build machine learning model—Random Forest—to predict optimal reagent flow (Amine collector) based on real-time operational data. The goal: enable intelligent control of flotation processes to maximize iron recovery and minimize chemical waste.


<br>

📚 **Dataset & Feature Engineering**

**Raw data**: 580,000+ rows × 29 features, covering ore mineralogy, reagent flow, column sensor data, and concentrate quality.

 **Feature selection**: After cleaning, exploratory analysis, multicollinearity checks, and **OLS-based p-value filtering**, I identified **6 highly predictive and minimally correlated features** for Amina Flow(Collector):

* Flotation Column 03 Level
* Flotation Column 06 Level
* Starch Flow
* Ore Pulp Flow
* Ore Pulp Density
* % Silica Concentrate

📉 These features align with metallurgical intuition and provide a robust basis for interpretable AI-driven predictions.

<br>

🔧 **Technical Highlights**

* 🧪 Feature selection via **statistical inference (p-values)**
* 🧹 Time-synchronized multi-source data merging
* 📊 Hourly resampling and visualization to identify plant behavior, shutdowns, and sensor drift
* 🔁 Final model-ready dataset crafted to ensure **low multicollinearity and strong process relevance**

🧠 AI models trained to support **real-time or batch optimisation** of flotation reagent dosing

🚀**Quick Start**
* Fork the repository to your own GitHub account
* Clone your forked repo:
* Run the pipeline using: "python main.py"
* Click on the gradio live link and screen like below will open.
  
![Screenshot 2025-06-13 055735](https://github.com/user-attachments/assets/bf48cdbd-87ce-4bc3-91c2-627a7210b5fc) ![Screenshot 2025-06-13 055751](https://github.com/user-attachments/assets/6d55d044-144f-4b93-a259-e0bd4550db28)


  
