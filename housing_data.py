from pathlib import Path 
import pandas as pd
import tarfile 
import urllib.request

def load_housing_data():
    tarbell_path = Path("datasets/housing/housing.tgz")
    
    """Checks if datasets/hosuing is a file
    otherwise creates directory and downloads it locally"""
    if not tarbell_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarbell_path)

    with tarfile.open(tarbell_path) as housing_tarbell:
        """filter = 'data' limits extraction algo and improves security"""
        housing_tarbell.extractall(path="datasets", filter="data")
        
    """loads csv file into pandas DataFrame"""
    return pd.read_csv(Path('datasets/housing/housing.csv'))

housing_full = load_housing_data()
"""
Data sets attributes
longitude, latitude, housing_median_age, 
total_rooms, total_bedrooms, population, 
households, median_income, median_house_value, and ocean_proximity
"""
"""
print(type(housing_full))
print(housing_full.head())
print(housing_full.tail())
print(housing_full.info())
print(housing_full["ocean_proximity"].value_counts())
print(housing_full.describe())
print(housing_full["median_income"].describe())
"""
