# Machine Learning Dashboard
Grund idé
* Problem: Jag vill kunna förstå vad som påverkar kunders beslut att genomföra ett köp eller inte, i B2B-buisness där försäljningscykeln kan vara lång och ibland lite komplex för att förstå anledningarna. Bland annat vilka är nyckelfaktorerna som driver kunder till ett köp etc.? Detta kommer att hjälpa företag att få bättre förståelse och ge rätt insikter kring hur man ska agera i t.ex försäljningsstratergier, prisättningar. Problemet kan vara ett regressionsproblem men även classificeringsproblem, i projektet har jag använt random forest för att kunna bygga en modell som kan visa vilka nyckelfaktorer som spelar in för sannolikheterna för ett köp av en kund. 


## Installation
### Förutsättningar
* Python 3.x installerat på din dator.

### Steg
1. **Klonar repo:**  
    git clone: https://github.com/eljoh-97/ML-Dashboard
2. **Installera eventuella beroenden:**  
    pip install pandas
    pip install numpy
    pip install -U scikit-learn
    pip install category_encoders
    pip install dash
    pip install plotly
    pip install dash_bootstrap_components
    pip install dash_bootstrap_templates   

## Användning
1. #### Starta den interactiva appen:
   * Navigera till projektmappen i din terminal och kör:
   * python project_ai.py
   * Öppna dash applicationen via terminalen eller i webbläsaren med följande URL: http://127.0.0.1:8050/

2. #### UI-guidence:
    * Component 1 "Training Accuarcy" värdet från modellen
    * Component 2 "Test Accuarcy" värdet från modellen
    * Component 3: "Classification Report" matisen från den tränade modellen.
    * Component 4: "Feature dropdown", en dropdown ruta där du kan välja vilken feature som skall reflekteras på "SHAP dependence Value". Dropdownen är filterad till att endast visa top 10.
    * Component 5: "Classification Dropdown" här kan du välja mellan de 3 classerna: 0, 1, 2. Detta reflekteras i "SHAP Dependence Value" ihop med "Feature dropdown".
    * Component 6: "SHAP Feature Importance - Top 10" en simpel bar-chart som visar top vilka features som påverkade modellen mest. x-axeln visare vilka "Features", y-axeln visar Importances av modellen.
    * Component 6: "SHAP Dependence Value" en scatter plot som visualisera varje enskild features påverkan i modellen. Här går det även att se hur varje features påverkan i varje class.

## Licens

## Kontakt
* Namn: Elias Johansson
* E-post: ezza97@gmail.com
* GitHub: https://github.com/eljoh-97

## Teknologier som används
* Python 3.x: Huvudspråk för utveckling.
* Data manipulation: Pandas
* Model Ensemle: RandomForestClassifier
* Model Selection: train_test_split
* Model preprocessing: StandardScaler, LabelEncoder, OneHotEncoder TargetEncoder, LeaveOneOutEncoder
* Model metrics: accuracy_score, classification_report
* Interactive App: Plotly dash
* App themes: dash_bootstrap_components, dash_bootstrap_templates

## Förbättringspotiential
* Uppdatera kod enligt objektorienterad-approach
* Kunna testa olika ML-modeller i gränsnittet och se skillnaderna. 
* Kunna ladda upp egen data i Appen för egen utvärdering
* Kunna ladda ned en sammanställning av modellen från gränsnittet.  
