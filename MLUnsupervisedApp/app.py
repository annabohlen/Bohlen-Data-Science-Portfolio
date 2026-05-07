# Import necessary functions
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage

import matplotlib.pyplot as plt
import seaborn as sns

# Import Sample dataset (Iris)
@st.cache_data
def load_sample_data():
    return sns.load_dataset("iris")

# Page setup with a sidebar
st.set_page_config(page_title="Unsupervised ML Explorer", layout="wide")

st.title("🤖 Unsupervised Machine Learning Explorer")
st.markdown("Upload a dataset, explore clustering or dimensionality reduction techniques, and see how changing parameters affects results!")

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["📂 Data", "🧠 Model Setup", "📊 Results"])

# ---------------------------
# PAGE 1: DATA
# ---------------------------
if page == "📂 Data":

    st.subheader("📂 Choose Your Dataset")

    data_option = st.radio("Select Data Source", ["Upload CSV", "Use Sample Dataset"])
#Allow user to upload their own dataset
    if data_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.info("Upload a dataset to begin.")
            st.stop()
   #Give the user the option to use the sample dataset provided
    else:
        st.info("Using sample Iris dataset")
        df = load_sample_data()

    #Show preview of the dataset
    st.subheader("Preview")
    st.dataframe(df.head())

    st.write("Shape:", df.shape)

    st.session_state["df"] = df


# ---------------------------
# PAGE 2: MODEL SETUP
# ---------------------------
elif page == "🧠 Model Setup":
#Make sure the data exists (the user has either selected the sample dataset or has uploaded their own dataset that works)
    if "df" not in st.session_state:
        st.warning("Upload or select data first.")
        st.stop()
    #Retrieve the dataset
    df = st.session_state["df"].copy()

    # Keep only numeric columns
    df = df.select_dtypes(include=np.number)

    if df.shape[1] < 2:
        st.error("Dataset must have at least 2 numeric columns.")
        st.stop()

    # Handle missing values
    df = df.fillna(df.mean())

    # Scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    # Model selection
    model_name = st.selectbox(
        "Choose Method",
        ["K-Means Clustering", "Hierarchical Clustering", "PCA"]
    )

    # ---------------- K-MEANS ----------------
    if model_name == "K-Means Clustering":

        # General clustering explanation
        st.markdown("""
        <div style="
            background-color:#eef6ff;
            padding:15px;
            border-radius:10px;
            border-left:5px solid #2196F3;
            margin-bottom:15px;
        ">
        <b>What is Clustering?</b><br><br>

        Clustering is an unsupervised machine learning technique that groups similar data points together based on their features.<br><br>

        Unlike supervised learning, clustering does not use labeled target values. Instead, it attempts to discover natural patterns or groupings within the data.
        </div>
        """, unsafe_allow_html=True)

        # K-Means explanation
        st.markdown("""
        <div style="
            background-color:#f8f9fa;
            padding:15px;
            border-radius:10px;
            border-left:5px solid #333;
            margin-bottom:15px;
        ">
        <b>What is K-Means Clustering?</b><br><br>

        K-Means clustering divides the data into a user-selected number of clusters (k).<br><br>

        The algorithm works by assigning each data point to the nearest cluster center and repeatedly adjusting the centers until the clusters stabilize.
        </div>
        """, unsafe_allow_html=True)
        #Allow user to choose a hyperparameter (k)
        k = st.slider("Number of Clusters (k)", 2, 10, 3)

        model = KMeans(n_clusters=k, random_state=42, n_init=10)

        st.markdown(f"""
        <div style="
            background-color:#eef6ff;
            padding:15px;
            border-radius:10px;
            border-left:5px solid #2196F3;
        ">
        <b>What does k = {k} mean?</b><br><br>

        k determines how many clusters the algorithm will form.<br><br>

        • Smaller k → broader groups<br>
        • Larger k → more detailed clusters
        </div>
        """, unsafe_allow_html=True)


    # ---------------- HIERARCHICAL ----------------
    elif model_name == "Hierarchical Clustering":

        # General clustering explanation
        st.markdown("""
        <div style="
            background-color:#eef6ff;
            padding:15px;
            border-radius:10px;
            border-left:5px solid #2196F3;
            margin-bottom:15px;
        ">
        <b>What is Clustering?</b><br><br>

        Clustering is an unsupervised machine learning technique that groups similar data points together based on their features.<br><br>

        Unlike supervised learning, clustering does not use labeled target values. Instead, it attempts to discover natural patterns or groupings within the data.
        </div>
        """, unsafe_allow_html=True)

        # Hierarchical explanation
        st.markdown("""
        <div style="
            background-color:#f8f9fa;
            padding:15px;
            border-radius:10px;
            border-left:5px solid #333;
            margin-bottom:15px;
        ">
        <b>What is Hierarchical Clustering?</b><br><br>

        Hierarchical clustering builds clusters step-by-step by merging the most similar groups together.<br><br>

        The relationships between clusters are visualized using a dendrogram, which shows how clusters combine at different distances.
        </div>
        """, unsafe_allow_html=True)
        #Allow the user to choose a hyperparameter (linkage)
        linkage_type = st.selectbox(
            "Linkage Method",
            ["ward", "complete", "average", "single"]
        )

        n_clusters = st.slider("Number of Clusters", 2, 10, 3)

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_type
        )

        # Different explanations for each linkage type
        if linkage_type == "ward":
            linkage_explanation = (
                "Ward linkage merges clusters in a way that minimizes the increase "
                "in variance within clusters.<br><br>"
                "This usually creates compact, balanced clusters and is often "
                "the most commonly used linkage method."
            )

        elif linkage_type == "complete":
            linkage_explanation = (
                "Complete linkage measures the distance between the farthest points "
                "in two clusters.<br><br>"
                "This tends to create tighter, more separated clusters and is "
                "less sensitive to chaining."
            )

        elif linkage_type == "average":
            linkage_explanation = (
                "Average linkage measures the average distance between all points "
                "in two clusters.<br><br>"
                "This creates clusters that are generally balanced between "
                "compactness and flexibility."
            )

        else:
            linkage_explanation = (
                "Single linkage measures the distance between the closest points "
                "in two clusters.<br><br>"
                "This can create long “chains” of points and is more sensitive "
                "to noise and outliers."
            )

        st.markdown(f"""
        <div style="
            background-color:#f3fff3;
            padding:15px;
            border-radius:10px;
            border-left:5px solid #4CAF50;
            margin-top:10px;
        ">
            <b>Linkage Method: {linkage_type}</b><br><br>
            {linkage_explanation}
        </div>
        """, unsafe_allow_html=True)


    # ---------------- PCA ----------------
    else:

        # General dimensionality reduction explanation
        st.markdown("""
        <div style="
            background-color:#fff8e6;
            padding:15px;
            border-radius:10px;
            border-left:5px solid #f0ad4e;
            margin-bottom:15px;
        ">
        <b>What is Dimensionality Reduction?</b><br><br>

        Dimensionality reduction is an unsupervised learning technique that reduces the number of variables in a dataset while trying to preserve as much important information as possible.<br><br>

        This makes complex datasets easier to visualize and analyze.
        </div>
        """, unsafe_allow_html=True)

        # PCA explanation
        st.markdown("""
        <div style="
            background-color:#f8f9fa;
            padding:15px;
            border-radius:10px;
            border-left:5px solid #333;
            margin-bottom:15px;
        ">
        <b>What is PCA?</b><br><br>

        Principal Component Analysis (PCA) transforms the original variables into a smaller set of new variables called principal components.<br><br>

        These components capture the maximum amount of variation in the data while reducing dimensionality.
        </div>
        """, unsafe_allow_html=True)
        #Allow user to choice a hyperparameter (n_components)
        n_components = st.slider(
            "Number of Components",
            1,
            df.shape[1],
            2
        )

        model = PCA(n_components=n_components)

        st.markdown(f"""
        <div style="
            background-color:#fff8e6;
            padding:15px;
            border-radius:10px;
            border-left:5px solid #f0ad4e;
        ">
        <b>n_components = {n_components}</b><br><br>

        Controls how many dimensions are kept.<br><br>

        Lower values simplify the data but may lose information.
        </div>
        """, unsafe_allow_html=True)

    # Save
    st.session_state["model"] = model
    st.session_state["X"] = X
    st.session_state["model_name"] = model_name

    st.success("Model configured!")


# ---------------------------
# PAGE 3: RESULTS
# ---------------------------
elif page == "📊 Results":
#Prevent code from running if model has not yet been configured
    if "model" not in st.session_state:
        st.info("Configure a model first.")
        st.stop()
#Retrieve stored values
    model = st.session_state["model"]
    X = st.session_state["X"]
    model_name = st.session_state["model_name"]

    col1, col2 = st.columns(2)

    # ---------------- K-MEANS RESULTS ----------------
    if model_name == "K-Means Clustering":
        #Train model and assign cluster groups
        model.fit(X)
        labels = model.labels_

        # Silhouette Score
        score = silhouette_score(X, labels)

        with col1:
            st.metric("Silhouette Score", f"{score:.2f}")

        # Silhouette explanation bubble
            st.markdown(f"""
            <div style="
                background-color: #eef6ff;
                padding: 15px;
                border-radius: 10px;
                border-left: 5px solid #2196F3;
                margin-top: 10px;
            ">
            <b>Understanding the Silhouette Score</b><br><br>

            If the silhouette score is near 1, it means that the data points are near the points in their own cluster and far away from points in other clusters, which is an indication of good clustering.<br><br>

            If the silhouette score is around 0, there is likely some overlap between the clusters.<br><br>

            If the silhouette score is near -1, the data points are far away from the points in their own cluster and close to points in other clusters. This means that the clustering is bad.
            </div>
            """, unsafe_allow_html=True)

        # Scatter plot
        pca_vis = PCA(n_components=2)
        X_vis = pca_vis.fit_transform(X)

        fig, ax = plt.subplots()
        ax.scatter(X_vis[:, 0], X_vis[:, 1], c=labels)
        ax.set_title("Cluster Visualization")
        st.pyplot(fig)

        # Elbow plot
        inertia = []
        k_range = range(2, 11)

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertia.append(km.inertia_)

        fig2, ax2 = plt.subplots()
        ax2.plot(k_range, inertia, marker="o")
        ax2.set_xlabel("k")
        ax2.set_ylabel("Inertia")
        ax2.set_title("Elbow Plot")
        st.pyplot(fig2)

        
                # Elbow plot explanation bubble
        st.markdown(f"""
        <div style="
            background-color: #fff8e6;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #f0ad4e;
            margin-top: 10px;
        ">
        <b>Understanding Inertia & Elbow Plots</b><br><br>

        Inertia represents how tightly grouped the data points are within each cluster, and is measured by adding together the squares of the distance each observation is from its cluster center.<br><br>

        A lower inertia represents more tightly grouped clusters, which usually means the clusters are better.<br><br>

        A higher inertia represents looser clusters, which usually means the clusters are worse.<br><br>

        To pick the best k value, look for the “elbow” (inflection point on the curve) on the elbow plot, which provides the tightest, most accurate clusters while avoiding overfitting.
        </div>
        """, unsafe_allow_html=True)


    # ---------------- HIERARCHICAL RESULTS ----------------
    elif model_name == "Hierarchical Clustering":
        #Train model and assign cluster groups
        model.fit(X)
        labels = model.labels_

        # Dendrogram
        Z = linkage(X, method="ward")

        fig, ax = plt.subplots()
        dendrogram(Z, ax=ax)
        ax.set_title("Dendrogram")
        st.pyplot(fig)

        # Dendrogram explanation bubble
        st.markdown(f"""
        <div style="
            background-color: #f3fff3;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #4CAF50;
            margin-top: 10px;
        ">
        <b>Understanding the Dendrogram</b><br><br>

        A dendrogram is a tree-like diagram that shows how clusters are merged together during hierarchical clustering.<br><br>

        Each branch represents a cluster, and branches that merge lower on the graph are more similar to each other than branches that merge higher on the graph.<br><br>

        The height of each merge represents the distance between the clusters being combined.<br><br>

        To estimate a good number of clusters, look for large vertical gaps in the dendrogram and imagine drawing a horizontal line through the graph. The number of branches the line crosses can help determine the appropriate number of clusters.
        </div>
        """, unsafe_allow_html=True)

        # Scatter
        pca_vis = PCA(n_components=2)
        X_vis = pca_vis.fit_transform(X)

        fig2, ax2 = plt.subplots()
        ax2.scatter(X_vis[:, 0], X_vis[:, 1], c=labels)
        ax2.set_title("Cluster Visualization")
        st.pyplot(fig2)

    # ---------------- PCA RESULTS ----------------
    else:
        #Train model and assign cluster groups
        X_pca = model.fit_transform(X)

        explained = model.explained_variance_ratio_.sum()

        with col1:
            st.metric("Variance Explained", f"{explained:.2f}")

        # ---------------- PCA SCATTERPLOT ----------------
        if X_pca.shape[1] >= 2:

            fig, ax = plt.subplots()
            ax.scatter(X_pca[:, 0], X_pca[:, 1])
            ax.set_title("PCA Scatterplot")
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            st.pyplot(fig)

            st.markdown("""
            <div style="
                background-color:#fff8e6;
                padding:15px;
                border-radius:10px;
                border-left:5px solid #f0ad4e;
                margin-top:10px;
            ">
            <b>Understanding the PCA Scatterplot</b><br><br>

            The PCA scatterplot shows each data point after reducing the dataset into two principal components.<br><br>

            • Points that are close together are more similar in the original dataset.<br>
            • Points that are far apart are more different.<br><br>

            This helps reveal patterns or groupings in high-dimensional data in a 2D space.
            </div>
            """, unsafe_allow_html=True)

        # ---------------- EXPLAINED VARIANCE PLOT ----------------
        fig2, ax2 = plt.subplots()
        ax2.plot(np.cumsum(model.explained_variance_ratio_), marker="o")
        ax2.set_title("Explained Variance")
        ax2.set_xlabel("Number of Components")
        ax2.set_ylabel("Cumulative Variance Explained")
        st.pyplot(fig2)

        st.markdown(f"""
        <div style="
            background-color:#eef6ff;
            padding:15px;
            border-radius:10px;
            border-left:5px solid #2196F3;
            margin-top:10px;
        ">
        <b>Understanding Variance Explained</b><br><br>

        Variance explained tells you how much information in the original dataset is captured by the principal components.<br><br>

        Your value of <b>{explained:.2f}</b> means that the selected components preserve that proportion of the original variation in the data.<br><br>

        Higher values mean less information is lost during dimensionality reduction.
        </div>
        """, unsafe_allow_html=True)