# Chapter 5: Data Collection and Storage

## 🎯 **Learning Objectives**

By the end of this chapter, you will be able to:

- Collect real data from public APIs and online sources
- Download and work with datasets from GitHub and other repositories
- Implement web scraping techniques on real websites
- Store and manage data using relational databases and file systems
- Build complete data pipelines with real-world data

## 📚 **Chapter Overview**

This chapter covers essential data collection and storage concepts using **REAL DATA** from public sources. You'll learn to work with live APIs, download datasets, scrape websites, and build robust data storage systems - all using actual data instead of simulated examples.

## 🔍 **Key Topics**

### **5.1 Real Data Collection Methods**

- **Public APIs**: COVID-19 data, stock market data, weather data
- **Built-in Datasets**: sklearn datasets (iris, diabetes, breast cancer, wine)
- **GitHub Repositories**: Download real datasets from public sources
- **Web Scraping**: Extract data from actual websites

### **5.2 Data Storage Systems**

- **Relational Databases**: SQLite with real data schemas
- **File-based Storage**: JSON, CSV, and database files
- **Data Warehousing**: Dimensional modeling with real datasets

### **5.3 Data Pipelines and ETL**

- **Extract**: Pull data from multiple real sources
- **Transform**: Clean and process actual data
- **Load**: Store transformed data in destination systems

## 🚀 **Real Data Examples**

### **COVID-19 Data Collection**

```python
# Real API call to disease.sh
covid_url = "https://disease.sh/v3/covid-19/countries"
covid_response = requests.get(covid_url, timeout=10)
covid_data = covid_response.json()

# Results: 231 countries with real statistics
# USA: 111,820,082 cases, 1,219,487 deaths
# India: 45,035,393 cases, 533,570 deaths
# France: 40,138,560 cases, 167,642 deaths
```

### **Iris Dataset from sklearn**

```python
from sklearn.datasets import load_iris
iris = load_iris()
# 150 samples, 4 features, 3 species
# Real biological measurements of iris flowers
```

### **Web Scraping Real Quotes**

```python
# Scrape from quotes.toscrape.com
quotes_url = "http://quotes.toscrape.com"
# Results: 10 real inspirational quotes from famous authors
# "The world as we have created it is a process of our thinking..." - Albert Einstein
```

## 📊 **Generated Outputs**

### **Data Files Created**

- `covid_data.json`: Real COVID-19 statistics for 10 countries
- `iris_data.csv`: 150 iris flower measurements
- `quotes_data.json`: 10 scraped inspirational quotes
- `weather_data.json`: London weather conditions

### **Database Tables**

- `covid_data`: Country statistics and metrics
- `iris_data`: Flower measurements and species
- `quotes_data`: Scraped quotes and authors

### **Data Pipeline Results**

- ✅ Collected data from 4+ real sources
- ✅ Stored in 1 database system
- ✅ Created 3+ transformed datasets
- ✅ Established data warehouse with 2 dimension tables

## 🛠 **Technical Implementation**

### **Required Libraries**

```python
import requests          # API calls
import pandas as pd     # Data manipulation
import sqlite3          # Database operations
from sklearn.datasets   # Built-in datasets
from bs4 import BeautifulSoup  # Web scraping
```

### **API Endpoints Used**

- **COVID-19**: `https://disease.sh/v3/covid-19/countries`
- **Weather**: `https://api.openweathermap.org/data/2.5/weather`
- **Stocks**: `https://www.alphavantage.co/query`

### **Data Sources**

1. **Live APIs**: Real-time data collection
2. **sklearn Datasets**: Built-in machine learning datasets
3. **GitHub Repositories**: Public dataset downloads
4. **Web Scraping**: Real website content extraction

## 📈 **Learning Outcomes**

### **Practical Skills**

- Working with real-time data from APIs
- Handling actual data quality issues
- Building production-ready data pipelines
- Managing multiple data sources

### **Real-World Applications**

- **COVID-19 Tracking**: Global pandemic statistics
- **Financial Analysis**: Stock market data collection
- **Weather Monitoring**: Real-time climate data
- **Content Analysis**: Web scraping and text processing

## 🔧 **Hands-on Activities**

### **1. Run the Chapter**

```bash
cd book/ch05
python ch05_data_collection_storage.py
```

### **2. Explore Generated Data**

- Check the `collected_data/` directory for raw data
- Examine the `transformed_data/` directory for processed data
- Query the SQLite database for insights

### **3. Modify and Experiment**

- Change API endpoints to collect different data
- Add new data sources to the pipeline
- Customize data transformation logic
- Extend the database schema

## 📚 **Next Steps**

After completing this chapter, you'll be ready for:

- **Chapter 6**: Data Cleaning and Preprocessing
- **Chapter 7**: Exploratory Data Analysis with real datasets
- **Chapter 9**: Machine Learning on actual data

## 🌟 **Key Takeaways**

✅ **Real Data Collection**: APIs, web scraping, and dataset downloads
✅ **Data Storage**: Relational databases and file systems
✅ **Data Pipelines**: Complete ETL processes with real data
✅ **Practical Experience**: Working with actual data quality issues
✅ **Production Ready**: Industry-standard data collection techniques

---

**Ready to collect real data?** 🚀

This chapter transforms theoretical concepts into practical, hands-on experience with real-world data sources!
