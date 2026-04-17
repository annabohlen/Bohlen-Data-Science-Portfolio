#Import necessary functions
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score
)

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns

# Import Sample Dataset (Cereals)
@st.cache_data
def load_cereal_data():
    return pd.read_csv("cereal.csv")

# Page Setup with a sidebar
st.set_page_config(page_title="ML Explorer", layout="wide")

st.title("🤖 Machine Learning Explorer")
st.markdown("Welcome to my app! Here you can upload a dataset or use a sample, train models, and then instantly see how performance changes as you adjust parameters!")

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["📂 Data", "🧠 Model Setup", "📊 Results"])

# Page 1: Data
if page == "📂 Data":

    st.subheader("📂 Choose Your Dataset")

    data_option = st.radio("Select Data Source", ["Upload CSV", "Use Sample Dataset"])
#Allow the user to upload their own dataset
    if data_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.info("Upload a dataset to begin.")
            st.stop()
#Give the user the option to use the sample dataset provided
    else:
        st.info("Using sample cereal dataset")
        df = load_cereal_data()

    # Show preview of the dataset
    st.subheader("Preview")
    st.dataframe(df.head())

    st.write("Shape:", df.shape)

    target = st.selectbox("Select Target Column", df.columns)

    st.session_state["df"] = df
    st.session_state["target"] = target


# Page 2: Model Setup
elif page == "🧠 Model Setup":
#Make sure the data exists (the user has either selected the sample dataset or has uploaded their own dataset that works)
    if "df" not in st.session_state:
        st.warning("Upload or select data first.")
        st.stop()
#Retrieve the dataset
    df = st.session_state["df"].copy()
    target = st.session_state["target"]
#Remove missing target values
    df = df.dropna(subset=[target])
#Split the data into features (X) and target (y)
    X = df.drop(columns=[target])
    y = df[target]

    # Detect problem type (classification or regression)
    is_numeric = pd.api.types.is_numeric_dtype(y)
    unique_ratio = y.nunique() / len(y)

    if is_numeric and unique_ratio > 0.05:
        problem_type = "Regression"
    else:
        problem_type = "Classification"

    # Determine variable type wording
    if is_numeric and unique_ratio > 0.05:
        variable_type = "numerical"
    else:
        variable_type = "categorical"

    # Explain to the user the problem type of the target variable they chose in a visually appealing format
    st.markdown(f"""
    <div style="
     background-color: #f0f2f6;
     padding: 20px;
     border-radius: 12px;
     border-left: 6px solid #4CAF50;
     margin-bottom: 15px;
">
        <h4 style="margin-bottom: 5px;">
         Detected Problem Type: {problem_type}
     </h4>
        <p style="font-size: 14px; color: #555;">
            Because you chose <b>{target}</b> as your target column, which is a <b>{variable_type}</b> variable,
            we are dealing with a <b>{problem_type.lower()}</b> problem.
     </p>
    </div>
    """, unsafe_allow_html=True)

#Preprocessing
    #Handle missing values
    X = X.fillna(X.mean(numeric_only=True))
   #Convert categorical features and labels into numbers
    X = pd.get_dummies(X)
    if problem_type == "Classification" and y.dtype == "object":
        y = pd.factorize(y)[0]
    #Allow the user to choose their test size
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
    #Split the data into train and test sets depending on the user's test size selection
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y if problem_type == "Classification" else None
    )
    #Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

 # Model selection
    model_name = st.selectbox("Choose Model", ["Decision Tree", "KNN"])

    if problem_type == "Classification":
        #If the problem type is classification and the user chooses Decision Tree as their model, allow the user to choose the max depth and provide a brief assessment of their choice
        if model_name == "Decision Tree":
            max_depth = st.slider("Max Depth", 1, 20, 5)
            model = DecisionTreeClassifier(max_depth=max_depth)
            #Explain to the user the implications of the max depth they chose
            if max_depth <= 3:
                depth_behavior = "simple and may underfit (high bias)"
            elif max_depth <= 10:
                depth_behavior = "balanced complexity"
            else:
                depth_behavior = "very complex and may overfit (high variance)"

            st.markdown(f"""
            <div style="
                background-color: #f3fff3;
                padding: 15px;
                border-radius: 10px;
                border-left: 5px solid #4CAF50;
                margin-top: 10px;
            ">
            <b>What does Max Depth = {max_depth} mean?</b><br>
            This controls how deep the decision tree can grow.<br><br>
            This model is currently: <b>{depth_behavior}</b>.
            </div>
            """, unsafe_allow_html=True)
        #If the problem type is classification and the user chooses K Nearest Neighbors as their model, allow the user to choose the K and provide a brief assessment of their choice
        else:
            k = st.slider("Number of Neighbors (K)", 1, 15, 5)
            model = KNeighborsClassifier(n_neighbors=k)

            if k <= 3:
                bias_variance = "low bias but high variance (can overfit)"
            elif k <= 7:
                bias_variance = "balanced bias and variance"
            else:
                bias_variance = "high bias but low variance (can underfit)"

            st.markdown(f"""
            <div style="
                background-color: #eef6ff;
                padding: 15px;
                border-radius: 10px;
                border-left: 5px solid #2196F3;
                margin-top: 10px;
            ">
            <b>What does K = {k} mean?</b><br>
            K controls how many neighbors the model looks at when making predictions.<br><br>
            • Smaller K → more sensitive to noise<br>
            • Larger K → smoother, more stable predictions<br><br>
            This choice results in <b>{bias_variance}</b>.
            </div>
            """, unsafe_allow_html=True)
     #If the problem type is regression and the user chooses Decision Tree as their model, allow the user to choose the max depth and provide a brief assessment of their choice
    else:
        if model_name == "Decision Tree":
            max_depth = st.slider("Max Depth", 1, 20, 5)
            model = DecisionTreeRegressor(max_depth=max_depth)

            if max_depth <= 3:
                depth_behavior = "simple and may underfit (high bias)"
            elif max_depth <= 10:
                depth_behavior = "balanced complexity"
            else:
                depth_behavior = "very complex and may overfit (high variance)"

            st.markdown(f"""
            <div style="
                background-color: #f3fff3;
                padding: 15px;
                border-radius: 10px;
                border-left: 5px solid #4CAF50;
                margin-top: 10px;
            ">
            <b>What does Max Depth = {max_depth} mean?</b><br>
            This controls how deep the decision tree can grow.<br><br>
            This model is currently: <b>{depth_behavior}</b>.
            </div>
            """, unsafe_allow_html=True)
        #If the problem type is regression and the user chooses K Nearest Neighbors as their model, allow the user to choose the K and provide a brief assessment of their choice
        else:
            k = st.slider("Number of Neighbors (K)", 1, 15, 5)
            model = KNeighborsRegressor(n_neighbors=k)

            # Explanation bubble for K
            if k <= 3:
                bias_variance = "low bias but high variance (can overfit)"
            elif k <= 7:
                bias_variance = "balanced bias and variance"
            else:
                bias_variance = "high bias but low variance (can underfit)"

            st.markdown(f"""
            <div style="
                background-color: #eef6ff;
                padding: 15px;
                border-radius: 10px;
                border-left: 5px solid #2196F3;
                margin-top: 10px;
            ">
            <b>What does K = {k} mean?</b><br>
            K controls how many neighbors the model looks at when making predictions.<br><br>
            • Smaller K → more sensitive to noise<br>
            • Larger K → smoother, more stable predictions<br><br>
            This choice results in <b>{bias_variance}</b>.
            </div>
            """, unsafe_allow_html=True)
#Save what the user selected on previous pages so that the user can view a personalized results page
    st.session_state["model"] = model
    st.session_state["data_split"] = (X_train, X_test, y_train, y_test)
    st.session_state["problem_type"] = problem_type
#Give the user an encouraging message once their model has been configured
    st.success("Model configured!")



# Page 3: Results
elif page == "📊 Results":
#Check to make sure a model exists
    if "model" not in st.session_state:
        st.info("Configure a model first.")
        st.stop()
#Load everything that was saved on the model setup page
    model = st.session_state["model"]
    X_train, X_test, y_train, y_test = st.session_state["data_split"]
    problem_type = st.session_state["problem_type"]
#Train the model
    model.fit(X_train, y_train)
#Make predictions on the test set
    y_pred = model.predict(X_test)
#Create 2 column layout
    col1, col2 = st.columns(2)
#If the problem type is classification, calculate the accuracy and display a confusion matrix and a heatmap
    if problem_type == "Classification":

        acc = accuracy_score(y_test, y_pred)

        with col1:
            st.metric("🎯 Accuracy", f"{acc:.2f}")

        with col2:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            st.pyplot(fig)
    #Explain accuracy to the user
        st.markdown(f"""
        <div style="
            background-color: #f8f9fa;
            padding: 12px;
            border-radius: 10px;
            border-left: 5px solid #333;
            margin-top: 10px;
        ">
        <b>What is Accuracy?</b><br>
        Accuracy measures how often the model predicts correctly.<br><br>
        It is calculated as:<br>
        <b>(Correct Predictions ÷ Total Predictions)</b><br><br>
        An accuracy of <b>{acc:.2f}</b> means the model is correct about this proportion of the time.
        </div>
        """, unsafe_allow_html=True)
    #Display performance summary 
        st.subheader("Classification Report")
        st.code(classification_report(y_test, y_pred))
    #Compute and display ROC curve and AUC score, if applicable
        try:
            y_proba = model.predict_proba(X_test) #
            #Check if binary classification
            if len(np.unique(y_train)) == 2 and len(np.unique(y_test)) == 2:
                y_score = y_proba[:, 1] #take the probability of the first class

                fpr, tpr, _ = roc_curve(y_test, y_score) #compute ROC curve values
                roc_auc = auc(fpr, tpr) #compute AUC score
                #Plot the ROC curve
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                ax.plot([0, 1], [0, 1], linestyle="--")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend()

                st.pyplot(fig)
            #If the user's chosen problem is not binary classification, display message that a ROC curve cannot be shown
            else:
                st.info("ROC curve is only shown for binary classification problems.")
        #Catch if anything fails
        except:
            st.warning("ROC curve could not be computed for this model.")

    else:
        #If the problem type is regression, calculate the MSE and R^2
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        with col1:
            st.metric("MSE", f"{mse:.2f}")

        with col2:
            st.metric("R² Score", f"{r2:.2f}")
        #Explain to the user the significance of the MSE and R^2
        st.markdown(f"""<div style="
                background-color: #fff8e6;
                padding: 15px;
                border-radius: 10px;
                border-left: 5px solid #f0ad4e;
                margin-top: 10px;
            ">
            <b>Understanding Regression Metrics</b><br><br>

            Mean Squared Error (MSE):
            MSE measures the average squared difference between predicted and actual values.
            • Lower values are better
            • Penalizes large errors more heavily

            Your MSE of {mse:.2f} reflects how far predictions are from actual values on average.

            R² Score (Coefficient of Determination):
            R² measures how well the model explains the variability in the data.
            • 1.0 = perfect fit
            • 0.0 = no better than predicting the mean
            • Negative = worse than a simple average model

            Your R² score of {r2:.2f} indicates how well your model fits the data.
            """, unsafe_allow_html=True)
    #Create a plot that plots predictions vs. actual values and includes a reference line
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)

        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val])

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")

        st.pyplot(fig)