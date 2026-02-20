
import streamlit as st
st.title("Student Social Media & Relationships")
st.text("This app will examine the digital lives of high school, undergraduate, and graduate students, and whether there are relationships between their social media usage and the health of their relationships. ")


import pandas as pd
st.subheader("Exploring Our Dataset")

df = pd.read_csv("data/social_media.csv")

st.write("Here's our data!")
st.dataframe(df)

country = st.selectbox("Select a country", df["Country"].unique())
filtered_df = df[df["Country"] == country]

st.write(f"People in {country}")
st.dataframe(filtered_df)


import seaborn as sns
import matplotlib.pyplot as plt

st.write("Does social media usage affect mental health?")
fig1, ax1 = plt.subplots()
scatter_plot1 = sns.scatterplot(x=df["Avg_Daily_Usage_Hours"], y=df["Mental_Health_Score"])
st.pyplot(fig1)
st.write("Here we can see that there is definitely a negative correlation between the amount of hours students spend on social media and their mental health.")
st.write("")
#I chose to write "" so that there would be a space in between the text line analyzing the previous graph and the text line introducing the next

st.write("Does social media's effect on mental health differ based on the platform students use the most?")
fig2, ax2 = plt.subplots()
scatter_plot2 = sns.scatterplot(x=df["Avg_Daily_Usage_Hours"], y=df["Mental_Health_Score"], hue=df["Most_Used_Platform"])
st.pyplot(fig2)
st.write("Here it seems like students are more likely to have good mental health when their most used app is Facebook. The most used apps that dominate the high screen time and low mental health area of the graph are Instagram, Whatsapp and Tiktok.")
st.write("")


st.write("Which platforms have the highest levels of addiction among students? Which relationship statuses are the most likely to be addicted?")
st.bar_chart(df, x="Most_Used_Platform", y="Addicted_Score", color = "Relationship_Status")
st.write("The three social media apps that are the highest in addictiveness combined with popularity are Instagram, Tiktok and Facebook. Facebook users appear to be more likely to be in relationships than Instagram users.")
st.write("")

st.write("Does increased social media usage lead to less sleep per night?") 
fig3, ax3 = plt.subplots()
scatter_plot3 = sns.scatterplot(x=df["Avg_Daily_Usage_Hours"], y=df["Sleep_Hours_Per_Night"])
st.pyplot(fig3)
st.write("There is a pretty strong negative correlation between hours students spend on social media and hours they spend sleeping.")
st.write("")

st.write("Are people who use social media very frequently more likely to experience relationship conflicts?") 
fig4, ax4 = plt.subplots()
scatter_plot4 = sns.scatterplot(x=df["Avg_Daily_Usage_Hours"], y=df["Conflicts_Over_Social_Media"])
st.pyplot(fig4)
st.write("There seems to be a positive correlation between the time people spend on social media and how many conflicts they have.")