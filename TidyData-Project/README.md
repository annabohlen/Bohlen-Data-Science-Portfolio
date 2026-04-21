# **Welcome to my Tidy Data Project!**
The goal of this project was to practice cleaning up a dataset that is untidy. In order for a computer to be able to effectively
perform calculations on a dataset, it must be in a "Tidy" format. For a data set to be tidy, the following must be true:
1. Each variable forms a column.
2. Each observation forms a row.
3. Each type of observational unit forms a table.


To run this app locally:
1. Clone the repository:<br>
   git clone https://github.com/annabohlen/Bohlen-Data-Science-Portfolio/TidyData-Project.git<br>
   cd TidyData-Project
2. Download the necessary libraries and versions:<br>
   pandas>=1.5,<br>
   seaborn>=0.12,<br>
   matplotlib>=3.7
3. Run the program (MoneyballEDA.ipynb)

Here are some of the resources I used to build this app:<br>
Mutant Moneyball Dataset: https://github.com/EliCash82/mutantmoneyball<br>
Tidy Data Book Chapter: https://vita.had.co.nz/papers/tidy-data.pdf<br>
Tidy Data Pandas Cheat Sheet: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf<br>

The dataset I chose is called "Mutant Moneyball", and it is a dataset from the user "EliCash82" which contains information about the value of X-men comics sold.
The data is sorted by X-men members that appear in the comics, and the variables include the decades in which the comics were sold and the 
sources people bought them from.
Through some basic EDA, I was able to see that the value of the X-men comics decreased across the decades.

