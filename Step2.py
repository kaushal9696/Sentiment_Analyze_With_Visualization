import json
import matplotlib.pyplot as plt
import pandas as pd

from Test1 import Selected_Rows

file2 = '2.json'

product = []
with open(file2, encoding="utf8", errors="ignore") as data_file:
    data = data_file.read()
    for i in data.split('\n'):
        product.append(i)

# Firstly cleaning the data by converting files into proper json format files by some replacements and
# then Making a list of Tuples containg all the data of json files.
productDataframe = []
for x in product:
    try:
        y = x.replace("'", '"')
        jdata = json.loads(y)
        productDataframe.append((jdata['asin'], jdata['reviewerName']))
    except:
        pass

# Creating a dataframe using the list of Tuples got in the previous step.
Product_dataset = pd.DataFrame(productDataframe, columns=['Asin', 'Name'])
Sentimemt_Score_Product = Selected_Rows[['Asin', 'Sentiment_Score']]
Sentimemt_Score_Product = Sentimemt_Score_Product.groupby(['Asin', 'Sentiment_Score']).size().reset_index()

# Creating a new column with value of 'Sentimemt_Score_Product[0]' so that it is accessible with the index name.
Sentimemt_Score_Product['Count'] = Sentimemt_Score_Product[0]

# Taking the Required columns only.
Sentimemt_Score_Product = Sentimemt_Score_Product[['Asin', 'Sentiment_Score', 'Count']]

x1 = Sentimemt_Score_Product.sort_values(['Asin', 'Count'], ascending=True).groupby(['Asin']).head()
x2 = Product_dataset
result = pd.merge(x2, x1, on='Asin', how='inner')

result.to_csv('Sentiment_Distribution_Across_Product.csv')
Positive = result[result.Sentiment_Score == 'pos']

# Selecting the rows whose sentiment is negative
Negative = result[result.Sentiment_Score == 'neg']

# Selecting the rows whose sentiment is neutral
Neutral = result[result.Sentiment_Score == 'neu']

result_Positive = Positive.sort_values('Count', ascending=False).reset_index()
result_Negative = Negative.sort_values('Count', ascending=False).reset_index()
result_Neutral = Neutral.sort_values('Count', ascending=False).reset_index()

result_Positive = result_Positive.drop('index', 1)
result_Negative = result_Negative.drop('index', 1)
result_Neutral = result_Neutral.drop('index', 1)

result_Positive.to_csv('Positive_Sentiment_Max.csv')
result_Negative.to_csv('Negative_Sentiment_Max.csv')
result_Neutral.to_csv('Neutral_Sentiment_Max.csv')

Percentage = result.groupby('Sentiment_Score')['Count'].sum().reset_index()

Percentage['Percentage'] = (Percentage.Count / Percentage.Count.sum()) * 100

Percentage.to_csv('Sentiment_Percentage.csv')
Selected_Rows['Review_Time'] = pd.to_datetime(Selected_Rows['Review_Time'])

Selected_Rows['Month'] = Selected_Rows['Review_Time'].dt.month
Selected_Rows['Year'] = Selected_Rows['Review_Time'].dt.year

Sentiment_Year = Selected_Rows.groupby(['Year', 'Sentiment_Score'])['Asin'].count().reset_index()
Sentiment_Year = Sentiment_Year.rename(columns={'Asin': 'Count'})

Positive_Year = Sentiment_Year[Sentiment_Year.Sentiment_Score == 'pos']
Negative_Year = Sentiment_Year[Sentiment_Year.Sentiment_Score == 'neg']
Neutral_Year = Sentiment_Year[Sentiment_Year.Sentiment_Score == 'neu']

Sentiment_Total_Year = Sentiment_Year.groupby('Year')['Count'].sum().reset_index()
Sentiment_Total_Year = Sentiment_Total_Year.rename(columns={'Count': 'Total_Count'})

result_Positive_Year = pd.merge(Positive_Year, Sentiment_Total_Year, on='Year', how='inner')
result_Negative_Year = pd.merge(Negative_Year, Sentiment_Total_Year, on='Year', how='inner')
result_Neutral_Year = pd.merge(Neutral_Year, Sentiment_Total_Year, on='Year', how='inner')

result_Positive_Year['Percentage'] = (result_Positive_Year['Count'] / result_Positive_Year['Total_Count']) * 100
result_Negative_Year['Percentage'] = (result_Negative_Year['Count'] / result_Negative_Year['Total_Count']) * 100
result_Neutral_Year['Percentage'] = (result_Neutral_Year['Count'] / result_Neutral_Year['Total_Count']) * 100

result_Positive_Year.to_csv('Pos_Sentiment_Percentage_vs_Year.csv')
result_Negative_Year.to_csv('Neg_Sentiment_Percentage_vs_Year.csv')
result_Neutral_Year.to_csv('Neu_Sentiment_Percentage_vs_Year.csv')

result_Positive_Year.plot(x="Year", y="Percentage", kind="bar",
                          title="Positive Reviews over the Years based on Sentiments")
plt.show()
result_Negative_Year.plot(x="Year", y="Percentage", kind="bar",
                          title="Negative Reviews over the Years based on Sentiments")
plt.show()
c = result_Neutral_Year.plot(x="Year", y="Percentage", kind="bar",
                         title="Neutral Reviews over the Years based on Sentiments")
plt.show()
