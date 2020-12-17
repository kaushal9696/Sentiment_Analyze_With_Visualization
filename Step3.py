import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import calendar

file = '2.json'

review = []
with open(file, encoding="utf8", errors="ignore") as data_file:
    data = data_file.read()
    for i in data.split('\n'):
        review.append(i)

# Making a list of Tuples containg all the data of json files.
reviewDataframe = []
for x in review:
    try:
        jdata = json.loads(x)
        reviewDataframe.append((jdata['overall'], jdata['verified'], jdata['reviewTime'], jdata['reviewerID'],
                                jdata['asin'], jdata['reviewerName'], jdata['reviewText'], jdata['summary'],
                                jdata['unixReviewTime']))
    except:
        pass

    # Creating a dataframe using the list of Tuples got in the previous step.
dataset = pd.DataFrame(reviewDataframe,
                       columns=['Rating', 'Verified', 'Review_Time', 'Reviewer_ID', 'Asin', 'Reviewer_Name',
                                'Review_Text', 'Summary', 'Unix_Review_Time'])

dataset['Review_Time'] = pd.to_datetime(dataset['Review_Time'])
dataset['Month'] = dataset['Review_Time'].dt.month
dataset['Year'] = dataset['Review_Time'].dt.year

Yearly = dataset.groupby(['Year'])['Reviewer_ID'].count().reset_index()
Yearly = Yearly.rename(columns={'Reviewer_ID': 'Number_Of_Reviews'})
Yearly.head()
Yearly.to_csv('Year_VS_Reviews.csv')
Yearly.plot(x="Year", y="Number_Of_Reviews", kind="line", title="NUMBER OF REVIEWS OVER THE YEARS")
plt.show()

Monthly = dataset.groupby(['Month'])['Reviewer_ID'].count().reset_index()
Monthly['Month'] = Monthly['Month'].apply(lambda x: calendar.month_name[x])
Monthly = Monthly.rename(columns={'Reviewer_ID': 'Number_of_Reviews'})
Monthly.head()
Monthly.to_csv('Month_VS_Reviews.csv')
Monthly.plot(x="Month", y="Number_of_Reviews", kind="bar", title="NUMBER OF REVIEWS OVER THE MONTH")
plt.show()

Overall_Rating = dataset.groupby(['Rating'])['Reviewer_ID'].count().reset_index()
Overall_Rating = Overall_Rating.rename(columns={'Reviewer_ID': 'Number_of_Reviews'})
Overall_Rating.to_csv('Rating_VS_Reviews.csv')
Overall_Rating.plot(x="Rating", y="Number_of_Reviews", kind="bar", title="NUMBER OF REVIEWS OVER THE MONTH")
plt.show()

Yearly_Avg_Rating = dataset.groupby(['Year'])['Rating'].mean().reset_index()
Yearly_Avg_Rating['Moving_Average'] = Yearly_Avg_Rating['Rating'].rolling(window=3).mean()
Yearly_Avg_Rating.head()
Yearly_Avg_Rating.to_csv('Yearly_Avg_Rating.csv')
Yearly_Avg_Rating.plot(x="Year", y=["Rating", "Moving_Average"], kind="bar",
                       title="AVERAGE OVERALL RATINGS OVER THE YEARS")
plt.show()

Review_Length = dataset[['Reviewer_ID', 'Reviewer_Name', 'Review_Text']]
Review_Length['Word_Length'] = Review_Length['Review_Text'].apply(lambda x: len(x.split()))
Review_Length['Character_Length'] = Review_Length['Review_Text'].apply(lambda x: len(x))
Review_Length.head()

Char_Review_Length = Review_Length.groupby(pd.cut(Review_Length.Character_Length, np.arange(0, 1501, 100))).count()
Char_Review_Length = Char_Review_Length.rename(columns={'Character_Length': 'Count'})
result_Char_Review_Length = Char_Review_Length.reset_index()

Word_Review_Length = Review_Length.groupby(pd.cut(Review_Length.Word_Length, np.arange(0, 801, 100))).count()
Word_Review_Length = Word_Review_Length.rename(columns={'Word_Length': 'Count'})
result_Word_Review_Length = Word_Review_Length.reset_index()

result_Char_Review_Length[["Character_Length", "Count"]].to_csv('Character_Length_Distribution.csv')
result_Word_Review_Length[["Word_Length", "Count"]].to_csv('Word_Length_Distribution.csv')

result_Char_Review_Length[["Character_Length", "Count"]].head()
result_Char_Review_Length.plot(x="Character_Length", y="Count", kind="bar", title="Distribution of Character Length")
plt.show()

result_Word_Review_Length[["Word_Length", "Count"]].head()
result_Word_Review_Length.plot(x="Word_Length", y="Count", kind="bar", title="Distribution of Word Length")
plt.show()
