import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def map_to_region(country):
    """Map countries to regions"""
    asia = ['China', 'India', 'Japan', 'Thailand', 'Vietnam', 'South Korea', 'Indonesia', 
            'Malaysia', 'Singapore', 'Philippines', 'Taiwan', 'Hong Kong']
    europe = ['Germany', 'France', 'UK', 'Italy', 'Switzerland', 'Spain', 'Poland', 
              'Netherlands', 'Belgium', 'Sweden', 'Norway', 'Denmark', 'Finland', 'Austria']
    americas = ['US', 'Canada', 'Mexico', 'Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru']
    middle_east = ['Turkey', 'UAE', 'Saudi Arabia', 'Qatar', 'Kuwait', 'Oman', 'Bahrain', 'Egypt']
    
    if country in asia:
        return 'Asia'
    elif country in europe:
        return 'Europe'
    elif country in americas:
        return 'Americas'
    elif country in middle_east:
        return 'Middle East'
    else:
        return 'Other'

def load_and_preprocess_data(data_path, k_features=8):
    """Load and preprocess data"""
    # Load data
    data = pd.read_csv(data_path)
    
    # Create region labels
    data['Region'] = data['Countries'].apply(map_to_region)
    
    # Prepare features and target
    X = data.iloc[:, 1:-1]  # All quarterly data, excluding Countries and Region
    y = data['Region']      # Target is region
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection
    selector = SelectKBest(f_classif, k=k_features)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Get selected feature names
    feature_names = X_train.columns[selector.get_support()].tolist()
    
    # Convert to DataFrame and preserve column names
    X_train_selected = pd.DataFrame(X_train_selected, columns=feature_names)
    X_test_selected = pd.DataFrame(X_test_selected, columns=feature_names)
    
    return X_train_selected, X_test_selected, y_train, y_test

def main():
    """Main function"""
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        'data/gold_demand.csv',
        k_features=8
    )
    
    # Save processed data
    X_train.to_csv('data/processed_train_data.csv', index=False)
    X_test.to_csv('data/processed_test_data.csv', index=False)
    pd.Series(y_train).to_csv('data/train_labels.csv', index=False)
    pd.Series(y_test).to_csv('data/test_labels.csv', index=False)

if __name__ == '__main__':
    main() 