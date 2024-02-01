![Israel_monthly_Average_Temperature](https://github.com/libbyyosef/Machine-Learning---Linear-Regression-Model-Fitting/assets/36642026/e5d1d0b2-c7da-4622-addf-6b3aa259c1f0)# Linear-Regression-Model-Fitting


**Linear Regression Model Fitting**

This repository contains Python implementations for fitting a linear regression model to real-world datasets. The implementation includes various components such as loss functions, linear regression models, data preprocessing, feature selection, and polynomial fitting.

**Overview**

This linear regression model fitting exercise is a continuation of the Univariate-and-Multivariate-Gaussian-Estimation project. The repository contains Python implementations for fitting a linear regression model to real-world datasets. The implementation includes various components such as loss functions, linear regression models, data preprocessing, feature selection, and polynomial fitting.

**Implemented Steps:**

- Loading and Preprocessing Data: The load_data function loads and preprocesses a city daily temperature dataset. Preprocessing steps include handling missing values, removing duplicates, and deriving additional features such as 'DayOfYear'.
- Exploring Data for Specific Country (Israel): Data exploration for the country 'Israel' involves plotting a scatter plot of temperature against DayOfYear, color-coded by year, and plotting a bar plot showing the standard deviation of temperatures for each month.
- Exploring Differences Between Countries: Data for multiple countries is grouped by month, and the average monthly temperatures are plotted with error bars representing the standard deviation. This helps in comparing temperature patterns between different countries.
- Fitting Model for Different Values of k (Polynomial Fitting): A polynomial regression model is fitted for different degrees (k) of polynomials, and the test error for each value of k is recorded. The results are visualized using a bar plot.
- Evaluating Fitted Model on Different Countries: The polynomial regression model fitted for Israel is evaluated on data from other countries (Jordan, South Africa, The Netherlands), and the test error for each country is recorded. Results are visualized using a bar plot.

**File Structure**

The repository includes the following files relevant to the implemented linear regression model fitting:

- **city_temperature_prediction.py:** Contains the implemented code for loading temperature data, preprocessing, data exploration, polynomial fitting, and model evaluation specific to temperature data.
- **datasets/City_Temperature.csv:** Dataset file containing city daily temperature data.
- **Israel_Daily_Temperature.png:** Image file containing the scatter plot of temperature against DayOfYear for Israel.
- **Israel_monthly_Average_Temperature.png:** Image file containing the bar plot of standard deviation of temperatures for each month in Israel.
- **Mean_Temp_Different_Countries.png:** Image file containing the line plot of average monthly temperatures for different countries with error bars.
- **Israel_Different_k.png:** Image file containing the bar plot of test error for different values of k in polynomial fitting for Israel.
- **Other_Countries_Loss.png:** Image file containing the bar plot of test error for other countries when using the model fitted over Israel.


![Avg_Loss_As_Function_Of_Training_Size](https://github.com/libbyyosef/Machine-Learning---Linear-Regression-Model-Fitting/assets/36642026/64fc7c90-037b-452e-a418-13801d84d549)


![Israel_Daily_Temperature](https://github.com/libbyyosef/Machine-Learning---Linear-Regression-Model-Fitting/assets/36642026/753c98ce-5bb5-41a2-ab48-9a80abb0188c)


![Israel_Different_k](https://github.com/libbyyosef/Machine-Learning---Linear-Regression-Model-Fitting/assets/36642026/5d7bb3ec-eccb-409c-b4c2-50627b3b4783)


![Pearson_Correlation_sqft_living](https://github.com/libbyyosef/Machine-Learning---Linear-Regression-Model-Fitting/assets/36642026/568fe3f2-50cb-44ff-8d09-7770b7465676)


![Israel_monthly_Average_Temperature](https://github.com/libbyyosef/Machine-Learning---Linear-Regression-Model-Fitting/assets/36642026/d3da15da-d578-4c49-95f6-d8b17f92f1bc)


![Pearson_Correlation_grade](https://github.com/libbyyosef/Machine-Learning---Linear-Regression-Model-Fitting/assets/36642026/de16fa57-0fca-42a9-82da-197aeb6106b1)


![Mean_Temp_Different_Countries](https://github.com/libbyyosef/Machine-Learning---Linear-Regression-Model-Fitting/assets/36642026/6bad9647-f762-4e86-bf68-2eb761909714)


![Other_Countries_Loss](https://github.com/libbyyosef/Machine-Learning---Linear-Regression-Model-Fitting/assets/36642026/23f0bcf4-9b3e-4a0d-8aa9-403b4d4cc0e5)


**Instructions**

To reproduce the analysis:
1. Clone this repository to your local machine.
2. Ensure you have Python installed along with necessary dependencies such as NumPy, Pandas, Plotly, etc.
3. Run the Python script city_temperature_prediction.py to execute the implemented steps and generate results.
4. Review the generated image files (*.png) for visualizations of temperature data and model evaluations.



