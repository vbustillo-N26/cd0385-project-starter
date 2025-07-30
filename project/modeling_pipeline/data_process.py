import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_log_transformed_data(train_df, target_col='count'):
    """Transform the target variable to log space"""
    train_log = train_df.copy()
    # Add 1 to avoid log(0), then take log
    train_log[target_col] = np.log1p(train_log[target_col])
    return train_log

def inverse_log_transform(predictions):
    """Transform predictions back from log space"""
    # Use expm1 which is the inverse of log1p
    return np.expm1(predictions)

def numerical_to_quartiles(df_train, df_test, num_cols):
    """ Build Quartiles based on df_train, apply them to test.
    """
    # Compute quantiles on the training set
    quantiles = df_train[num_cols].quantile([0.25, 0.5, 0.75]).to_dict()

    # Apply quantiles to both train and test sets
    for col in num_cols:
        df_train[col + '_quart'] = pd.cut(df_train[col], bins=[-np.inf] + list(quantiles[col].values()) + [np.inf], labels=False)
        df_test[col + '_quart'] = pd.cut(df_test[col], bins=[-np.inf] + list(quantiles[col].values()) + [np.inf], labels=False)

    # Build concatenated string of weather quartiles
    df_train['meteo_bin'] = df_train[['atemp_quart', 'humidity_quart', 'windspeed_quart']].astype(str).agg('-'.join, axis=1)
    df_test['meteo_bin'] = df_test[['atemp_quart', 'humidity_quart', 'windspeed_quart']].astype(str).agg('-'.join, axis=1)

    return df_train, df_test

def enrich_featureset(df,categorical_features):
    """Create additional features from datetime columns"""
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['dayofweek'] = df['datetime'].dt.dayofweek

    df['is_morning'] = df['hour'].apply(lambda x: 1 if 6 <= x < 16 else 0)  # Daytime indicator
    df['is_evening'] = df['hour'].apply(lambda x: 1 if 16 <= x < 23 else 0)  # Evening indicator
    df['is_night'] = df['hour'].apply(lambda x: 1 if (x < 6) | (x >= 23) else 0)  # Nighttime indicator

    for feature in categorical_features:  
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame")
        df[feature] = df[feature].astype('category')

    return df


def plot_temporal_data_distribution(train_df, test_df, datetime_col='datetime', figsize=(20, 8)):
    """
    Create a timeline visualization showing the temporal distribution of train and test data.
    Each hour is represented as a thin vertical line.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training dataframe with datetime column
    test_df : pandas.DataFrame
        Test dataframe with datetime column
    datetime_col : str, default='datetime'
        Name of the datetime column
    figsize : tuple, default=(20, 8)
        Figure size for the plot
    
    Returns:
    --------
    None (displays plot)
    """
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, height_ratios=[2, 1, 1])
    
    # Get datetime ranges
    train_times = pd.to_datetime(train_df[datetime_col])
    test_times = pd.to_datetime(test_df[datetime_col])
    
    # Combined timeline view (top plot)
    ax1.scatter(train_times, [1]*len(train_times), c='blue', alpha=0.6, s=1, label=f'Train (n={len(train_times)})')
    ax1.scatter(test_times, [0]*len(test_times), c='red', alpha=0.6, s=1, label=f'Test (n={len(test_times)})')
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Test', 'Train'])
    ax1.set_title('Temporal Distribution of Train vs Test Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Train data histogram (middle plot)
    ax2.hist(train_times, bins=100, color='blue', alpha=0.7, edgecolor='none')
    ax2.set_title('Train Data Distribution Over Time')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)
    
    # Test data histogram (bottom plot)
    ax3.hist(test_times, bins=100, color='red', alpha=0.7, edgecolor='none')
    ax3.set_title('Test Data Distribution Over Time')
    ax3.set_ylabel('Count')
    ax3.set_xlabel('Time')
    ax3.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("Temporal Data Summary:")
    print("=" * 50)
    print(f"Train data period: {train_times.min()} to {train_times.max()}")
    print(f"Test data period: {test_times.min()} to {test_times.max()}")
    print(f"Train duration: {(train_times.max() - train_times.min()).days} days")
    print(f"Test duration: {(test_times.max() - test_times.min()).days} days")
    
    # Check for overlaps
    overlap_start = max(train_times.min(), test_times.min())
    overlap_end = min(train_times.max(), test_times.max())
    
    if overlap_start <= overlap_end:
        print(f"Overlap period: {overlap_start} to {overlap_end}")
        print(f"Overlap duration: {(overlap_end - overlap_start).days} days")
    else:
        print("No overlap between train and test periods")
    
    # Gap analysis
    if test_times.min() > train_times.max():
        gap = test_times.min() - train_times.max()
        print(f"Gap between train and test: {gap.days} days")
    elif train_times.min() > test_times.max():
        gap = train_times.min() - test_times.max()
        print(f"Gap between test and train: {gap.days} days")
    else:
        print("Train and test periods overlap or are adjacent")


def plot_hourly_patterns(train_df, test_df, datetime_col='datetime', figsize=(15, 6)):
    """
    Create visualization showing hourly patterns in train vs test data.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training dataframe with datetime column
    test_df : pandas.DataFrame
        Test dataframe with datetime column
    datetime_col : str, default='datetime'
        Name of the datetime column
    figsize : tuple, default=(15, 6)
        Figure size for the plot
    
    Returns:
    --------
    None (displays plot)
    """
    # Extract hours
    train_hours = pd.to_datetime(train_df[datetime_col]).dt.hour
    test_hours = pd.to_datetime(test_df[datetime_col]).dt.hour
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Hourly distribution comparison
    hour_counts_train = train_hours.value_counts().sort_index()
    hour_counts_test = test_hours.value_counts().sort_index()
    
    hours = range(24)
    ax1.bar([h - 0.2 for h in hours], [hour_counts_train.get(h, 0) for h in hours], 
            width=0.4, color='blue', alpha=0.7, label='Train')
    ax1.bar([h + 0.2 for h in hours], [hour_counts_test.get(h, 0) for h in hours], 
            width=0.4, color='red', alpha=0.7, label='Test')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Number of Records')
    ax1.set_title('Data Distribution by Hour of Day')
    ax1.set_xticks(hours)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Daily pattern over time
    train_daily = pd.to_datetime(train_df[datetime_col]).dt.date.value_counts().sort_index()
    test_daily = pd.to_datetime(test_df[datetime_col]).dt.date.value_counts().sort_index()
    
    ax2.plot(train_daily.index, train_daily.values, color='blue', alpha=0.7, linewidth=1, label='Train')
    ax2.plot(test_daily.index, test_daily.values, color='red', alpha=0.7, linewidth=1, label='Test')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Records per Day')
    ax2.set_title('Daily Data Frequency Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def plot_target_vs_categorical_features(df, categorical_features, target_col='count', figsize=(15, 10), bins=30):
    """
    Create histograms showing the distribution of target variable across categorical features.
    All histograms use the same bins for easy comparison.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    categorical_features : list
        List of categorical feature column names
    target_col : str, default='count'
        Name of the target column
    figsize : tuple, default=(15, 10)
        Figure size for the plot
    bins : int, default=30
        Number of bins for histograms
    
    Returns:
    --------
    None (displays plots)
    """
    # Filter only categorical features that exist in the dataframe
    available_features = [feat for feat in categorical_features if feat in df.columns]
    
    if not available_features:
        print("No categorical features found in the dataframe")
        return
    
    # Define consistent bins for all histograms based on overall target range
    target_min = df[target_col].min()
    target_max = df[target_col].max()
    bin_edges = np.linspace(target_min, target_max, bins + 1)
    
    # Calculate number of rows and columns for subplots
    n_features = len(available_features)
    n_cols = 3  # 3 columns
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    for i, feature in enumerate(available_features):
        ax = axes_flat[i]
        
        # Get unique values for this categorical feature
        unique_values = sorted(df[feature].unique())
        
        # Create histograms for each category with consistent bins
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_values)))
        
        for j, category in enumerate(unique_values):
            category_data = df[df[feature] == category][target_col]
            ax.hist(category_data, bins=bin_edges, alpha=0.7, 
                   label=f'{category} (n={len(category_data)})', 
                   color=colors[j], density=True)
        
        ax.set_title(f'{target_col} distribution by {feature}')
        ax.set_xlabel(target_col)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set consistent x-axis limits
        ax.set_xlim(target_min, target_max)
    
    # Hide any unused subplots
    for i in range(n_features, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_target_vs_numerical_features_quartiles(df, numerical_features, target_col='count', figsize=(15, 10), bins=30):
    """
    Create histograms showing the distribution of target variable across quartiles of numerical features.
    All histograms use the same bins for easy comparison.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    numerical_features : list
        List of numerical feature column names
    target_col : str, default='count'
        Name of the target column
    figsize : tuple, default=(15, 10)
        Figure size for the plot
    bins : int, default=30
        Number of bins for histograms
    
    Returns:
    --------
    None (displays plots)
    """
    # Filter only numerical features that exist in the dataframe
    available_features = [feat for feat in numerical_features if feat in df.columns]
    
    if not available_features:
        print("No numerical features found in the dataframe")
        return
    
    # Define consistent bins for all histograms based on overall target range
    target_min = df[target_col].min()
    target_max = df[target_col].max()
    bin_edges = np.linspace(target_min, target_max, bins + 1)
    
    # Calculate number of rows and columns for subplots
    n_features = len(available_features)
    n_cols = 2  # 2 columns for numerical features (fewer features typically)
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    for i, feature in enumerate(available_features):
        ax = axes_flat[i]
        
        # Calculate quartiles for this numerical feature
        quartiles = df[feature].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
        
        # Create quartile labels
        quartile_labels = [
            f'Q1: [{quartiles[0]:.2f}, {quartiles[1]:.2f})',
            f'Q2: [{quartiles[1]:.2f}, {quartiles[2]:.2f})',
            f'Q3: [{quartiles[2]:.2f}, {quartiles[3]:.2f})',
            f'Q4: [{quartiles[3]:.2f}, {quartiles[4]:.2f}]'
        ]
        
        # Create masks for each quartile
        q1_mask = (df[feature] >= quartiles[0]) & (df[feature] < quartiles[1])
        q2_mask = (df[feature] >= quartiles[1]) & (df[feature] < quartiles[2])
        q3_mask = (df[feature] >= quartiles[2]) & (df[feature] < quartiles[3])
        q4_mask = (df[feature] >= quartiles[3]) & (df[feature] <= quartiles[4])
        
        masks = [q1_mask, q2_mask, q3_mask, q4_mask]
        colors = ['skyblue', 'lightgreen', 'gold', 'salmon']
        
        # Create histograms for each quartile with consistent bins
        for j, (mask, label, color) in enumerate(zip(masks, quartile_labels, colors)):
            quartile_data = df[mask][target_col]
            ax.hist(quartile_data, bins=bin_edges, alpha=0.7, 
                   label=f'{label} (n={len(quartile_data)})', 
                   color=color, density=True)
        
        ax.set_title(f'{target_col} distribution by {feature} quartiles')
        ax.set_xlabel(target_col)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set consistent x-axis limits
        ax.set_xlim(target_min, target_max)
    
    # Hide any unused subplots
    for i in range(n_features, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


# Create time lag features based on train data
def create_lag_features(train_df, test_df, target_col='count'):
    """
    Create lag features for both train and test datasets based on train data only.
    
    Parameters:
    - train_df: Training dataframe with datetime and target columns
    - test_df: Test dataframe with datetime column
    - target_col: Target column name (default: 'count')
    
    Returns:
    - train_with_lags: Train dataframe with lag features
    - test_with_lags: Test dataframe with lag features
    """
    
    # Combine train and test for consistent datetime indexing
    train_copy = train_df.copy()
    test_copy = test_df.copy()
    
    # Ensure datetime is the index or create a lookup dictionary
    train_lookup = train_copy.set_index('datetime')[target_col].to_dict()
    
    def add_lag_features(df, is_train=True):
        df_result = df.copy()
        
        # Hourly lags (1, 2, 3 hours)
        # for lag_hours in [1, 2, 3]:
        #     lag_col = f'lag_{lag_hours}h'
        #     lag_datetime = df['datetime'] - pd.Timedelta(hours=lag_hours)
            
        #     if is_train:
        #         # For train, we can use the actual values from the same dataset
        #         df_result[lag_col] = lag_datetime.map(train_lookup)
        #     else:
        #         # For test, only use train data for lag values
        #         df_result[lag_col] = lag_datetime.map(train_lookup)
        
        # Daily lags (1, 2, 7, 14 days)
        for lag_days in [1, 2, 7, 14]:
            lag_col = f'lag_{lag_days}d'
            lag_datetime = df['datetime'] - pd.Timedelta(days=lag_days)
            
            if is_train:
                # For train, use values from the same dataset
                df_result[lag_col] = lag_datetime.map(train_lookup)
            else:
                # For test, only use train data for lag values
                df_result[lag_col] = lag_datetime.map(train_lookup)
        
        return df_result
    
    # Add lag features to both datasets
    train_with_lags = add_lag_features(train_copy, is_train=True)
    test_with_lags = add_lag_features(test_copy, is_train=False)
    
    return train_with_lags, test_with_lags

# ...existing code...

def create_log_lag_features(train_df, test_df, target_col='count', missing_value=-999):
    """
    Create log-transformed lag features for both train and test datasets based on train data only.
    
    Parameters:
    - train_df: Training dataframe with datetime and target columns
    - test_df: Test dataframe with datetime column
    - target_col: Target column name (default: 'count')
    - missing_value: Value to use for missing lag features (default: -999)
    
    Returns:
    - train_with_lags: Train dataframe with log lag features
    - test_with_lags: Test dataframe with log lag features
    """
    import pandas as pd
    import numpy as np
    
    # Create copies to avoid modifying original dataframes
    train_copy = train_df.copy()
    test_copy = test_df.copy()
    
    # Create lookup dictionary with log-transformed values from train data
    if target_col in train_copy.columns:
        # Apply log transformation: log(x + 1) to handle zeros
        log_values = train_copy[target_col]
        train_log_lookup = dict(zip(train_copy['datetime'], log_values))
    else:
        raise ValueError(f"Target column '{target_col}' not found in train dataframe")
    
    def add_log_lag_features_to_df(df, lookup_dict, missing_val):
        """Add log lag features to a dataframe"""
        df_result = df.copy()
        
        # Hourly lags (1, 2, 3 hours)
        # for lag_hours in [1, 2, 3]:
        #     lag_col = f'log_lag_{lag_hours}h'
        #     lag_datetime = df['datetime'] - pd.Timedelta(hours=lag_hours)
        #     df_result[lag_col] = lag_datetime.map(lookup_dict).fillna(missing_val)
        
        # Daily lags (1, 2, 7, 14 days)
        for lag_days in [1, 2, 7, 14]:
            lag_col = f'log_lag_{lag_days}d'
            lag_datetime = df['datetime'] - pd.Timedelta(days=lag_days)
            df_result[lag_col] = lag_datetime.map(lookup_dict).fillna(missing_val)
        
        return df_result
    
    # Add lag features to both datasets
    train_with_lags = add_log_lag_features_to_df(train_copy, train_log_lookup, missing_value)
    test_with_lags = add_log_lag_features_to_df(test_copy, train_log_lookup, missing_value)
    
    return train_with_lags, test_with_lags

def get_log_lag_feature_names():
    """
    Return list of log lag feature names
    
    Returns:
    - List of log lag feature column names
    """
    # hourly_lags = [f'log_lag_{h}h' for h in [1, 2, 3]]
    daily_lags = [f'log_lag_{d}d' for d in [1, 2, 7, 14]]
    # return hourly_lags + daily_lags
    return daily_lags

def analyze_lag_feature_coverage(df, lag_features, title_prefix=""):
    """
    Analyze the temporal coverage and statistics of lag features
    
    Parameters:
    - df: DataFrame with lag features and datetime column
    - lag_features: List of lag feature column names
    - title_prefix: Prefix for plot titles
    """
    import matplotlib.pyplot as plt
    
    print(f"\n{title_prefix}Lag Feature Coverage Analysis:")
    print("=" * 60)
    
    for feature in lag_features:
        if feature in df.columns:
            total_count = len(df)
            missing_count = (df[feature] == -999).sum()  # Assuming -999 is missing value
            available_count = total_count - missing_count
            coverage_pct = (available_count / total_count) * 100
            
            if available_count > 0:
                mean_val = df[df[feature] != -999][feature].mean()
                std_val = df[df[feature] != -999][feature].std()
                print(f"{feature}:")
                print(f"  Available: {available_count:,} ({coverage_pct:.1f}%)")
                print(f"  Missing: {missing_count:,} ({100-coverage_pct:.1f}%)")
                print(f"  Mean: {mean_val:.4f}, Std: {std_val:.4f}")
            else:
                print(f"{feature}: No available data")
        else:
            print(f"{feature}: Column not found")
        print()

def calculate_lag_feature_correlations(df, lag_features, target_col='count'):
    """
    Calculate correlations between lag features and target variable
    
    Parameters:
    - df: DataFrame with lag features and target column
    - lag_features: List of lag feature column names
    - target_col: Target column name
    
    Returns:
    - List of tuples (feature_name, correlation, n_observations)
    """
    correlations = []
    
    if target_col not in df.columns:
        print(f"Warning: Target column '{target_col}' not found")
        return correlations
    
    print(f"\nCorrelation between lag features and {target_col}:")
    print("=" * 50)
    
    for lag_feature in lag_features:
        if lag_feature in df.columns:
            # Only calculate correlation where lag feature is not missing (-999)
            mask = (df[lag_feature] != -999) & df[target_col].notnull()
            
            if mask.sum() > 0:
                corr = df.loc[mask, lag_feature].corr(df.loc[mask, target_col])
                correlations.append((lag_feature, corr, mask.sum()))
                print(f"{lag_feature}: {corr:.4f} (based on {mask.sum():,} observations)")
            else:
                correlations.append((lag_feature, 0.0, 0))
                print(f"{lag_feature}: No valid observations")
        else:
            print(f"{lag_feature}: Column not found")
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\nMost predictive lag features:")
    for i, (feature, corr, n_obs) in enumerate(correlations[:5], 1):
        print(f"{i}. {feature}: {corr:.4f} ({n_obs:,} obs)")
    
    return correlations

def analyze_temporal_coverage_detailed(train_df, test_df, start_date='2011-10-01', end_date='2011-10-05', datetime_col='datetime'):
    """
    Analyze detailed temporal coverage for consecutive days showing train/test distribution
    
    Parameters:
    - train_df: Training dataframe with datetime column
    - test_df: Test dataframe with datetime column  
    - start_date: Start date for analysis (YYYY-MM-DD format)
    - end_date: End date for analysis (YYYY-MM-DD format)
    - datetime_col: Name of datetime column
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Convert date strings to datetime
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    print(f"Temporal Coverage Analysis: {start_date} to {end_date}")
    print("=" * 80)
    
    # Create complete hourly range for the period
    complete_range = pd.date_range(start=start_dt, end=end_dt, freq='H')
    
    # Get train and test data for this period
    train_period = train_df[(train_df[datetime_col] >= start_dt) & (train_df[datetime_col] <= end_dt)].copy()
    test_period = test_df[(test_df[datetime_col] >= start_dt) & (test_df[datetime_col] <= end_dt)].copy()
    
    print(f"Complete time range: {len(complete_range)} hours")
    print(f"Train data points: {len(train_period)}")
    print(f"Test data points: {len(test_period)}")
    print(f"Missing hours: {len(complete_range) - len(train_period) - len(test_period)}")
    
    # Create analysis dataframe
    analysis_df = pd.DataFrame({
        'datetime': complete_range,
        'date': complete_range.date,
        'hour': complete_range.hour,
        'day_of_week': complete_range.day_name(),
        'data_type': 'missing'
    })
    
    # Mark train and test periods
    train_times = set(train_period[datetime_col])
    test_times = set(test_period[datetime_col])
    
    analysis_df.loc[analysis_df['datetime'].isin(train_times), 'data_type'] = 'train'
    analysis_df.loc[analysis_df['datetime'].isin(test_times), 'data_type'] = 'test'
    
    # Daily summary
    print("\nDaily Summary:")
    print("-" * 50)
    daily_summary = analysis_df.groupby('date')['data_type'].value_counts().unstack(fill_value=0)
    
    for date in analysis_df['date'].unique():
        date_data = daily_summary.loc[date] if date in daily_summary.index else pd.Series({'train': 0, 'test': 0, 'missing': 0})
        total = date_data.sum()
        train_pct = (date_data.get('train', 0) / 24) * 100
        test_pct = (date_data.get('test', 0) / 24) * 100
        missing_pct = (date_data.get('missing', 0) / 24) * 100
        
        print(f"{date} ({pd.to_datetime(date).strftime('%A')}):")
        print(f"  Train: {date_data.get('train', 0):2d} hours ({train_pct:5.1f}%)")
        print(f"  Test:  {date_data.get('test', 0):2d} hours ({test_pct:5.1f}%)")
        print(f"  Missing: {date_data.get('missing', 0):2d} hours ({missing_pct:5.1f}%)")
        print()
    
    # Hourly breakdown for each day
    print("Detailed Hourly Breakdown:")
    print("-" * 80)
    print("Hour: ", end="")
    for h in range(24):
        print(f"{h:2d}", end=" ")
    print()
    print("-" * 80)
    
    for date in analysis_df['date'].unique():
        day_data = analysis_df[analysis_df['date'] == date].sort_values('hour')
        day_name = pd.to_datetime(date).strftime('%a')
        print(f"{date} ({day_name}): ", end="")
        
        for hour in range(24):
            hour_data = day_data[day_data['hour'] == hour]
            if len(hour_data) > 0:
                data_type = hour_data['data_type'].iloc[0]
                if data_type == 'train':
                    symbol = 'T'
                elif data_type == 'test':
                    symbol = 'X'
                else:
                    symbol = '-'
            else:
                symbol = '?'
            print(f" {symbol}", end=" ")
        print()
    
    print("\nLegend: T=Train, X=Test, -=Missing, ?=No data")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Daily distribution
    daily_counts = analysis_df.groupby(['date', 'data_type']).size().unstack(fill_value=0)
    daily_counts.plot(kind='bar', stacked=True, ax=ax1, 
                     color={'train': 'blue', 'test': 'orange', 'missing': 'gray'})
    ax1.set_title(f'Daily Data Distribution ({start_date} to {end_date})')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Hours')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Hourly heatmap
    pivot_data = analysis_df.pivot_table(index='date', columns='hour', values='data_type', aggfunc='first')
    
    # Convert to numeric for heatmap (train=1, test=2, missing=0)
    pivot_numeric = pivot_data.copy()
    pivot_numeric = pivot_numeric.replace({'train': 1, 'test': 2, 'missing': 0})
    
    im = ax2.imshow(pivot_numeric.values, cmap='RdYlBu_r', aspect='auto')
    ax2.set_title('Hourly Data Availability Heatmap')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Date')
    ax2.set_xticks(range(24))
    ax2.set_xticklabels(range(24))
    ax2.set_yticks(range(len(pivot_data.index)))
    ax2.set_yticklabels([str(d) for d in pivot_data.index])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['Missing', 'Train', 'Test'])
    
    plt.tight_layout()
    plt.show()
    
    return analysis_df

def analyze_lag_feature_availability_for_period(train_df, test_df, start_date='2011-10-01', end_date='2011-10-05', 
                                               target_col='count', datetime_col='datetime'):
    """
    Analyze lag feature availability for a specific period
    """
    import pandas as pd
    import numpy as np
    
    print(f"\nLag Feature Availability Analysis: {start_date} to {end_date}")
    print("=" * 80)
    
    # Get the temporal coverage first
    analysis_df = analyze_temporal_coverage_detailed(train_df, test_df, start_date, end_date, datetime_col)
    
    # Create lag lookup from train data (using log-transformed values if available)
    if target_col in train_df.columns:
        if train_df[target_col].min() > 0 and train_df[target_col].max() < 10:  # Likely already log-transformed
            print(f"Using already log-transformed {target_col} values")
            train_lookup = dict(zip(train_df[datetime_col], train_df[target_col]))
        else:
            print(f"Applying log transformation to {target_col}")
            log_values = np.log1p(train_df[target_col].fillna(0))
            train_lookup = dict(zip(train_df[datetime_col], log_values))
    else:
        print(f"Target column {target_col} not found")
        return
    
    # Add lag features to analysis
    lag_configs = [
        # (1, 'hour'), (2, 'hour'), (3, 'hour'),
        (1, 'day'), (2, 'day'), (7, 'day'), (14, 'day')
    ]
    
    for lag_value, lag_unit in lag_configs:
        lag_col = f'log_lag_{lag_value}{"h" if lag_unit == "hour" else "d"}'
        
        if lag_unit == 'hour':
            lag_datetime = analysis_df['datetime'] - pd.Timedelta(hours=lag_value)
        else:
            lag_datetime = analysis_df['datetime'] - pd.Timedelta(days=lag_value)
        
        analysis_df[lag_col] = lag_datetime.map(train_lookup)
        analysis_df[f'{lag_col}_available'] = analysis_df[lag_col].notna()
    
    # Show lag availability for test data points only
    test_only = analysis_df[analysis_df['data_type'] == 'test'].copy()
    
    if len(test_only) > 0:
        print(f"\nLag Feature Availability for {len(test_only)} Test Data Points:")
        print("-" * 60)
        
        for lag_value, lag_unit in lag_configs:
            lag_col = f'log_lag_{lag_value}{"h" if lag_unit == "hour" else "d"}'
            available_count = test_only[f'{lag_col}_available'].sum()
            total_count = len(test_only)
            pct = (available_count / total_count) * 100
            
            print(f"{lag_col:12s}: {available_count:3d}/{total_count:3d} ({pct:5.1f}%)")
        
        # Show detailed breakdown for first few test points
        print(f"\nDetailed Lag Availability for First 10 Test Points:")
        print("-" * 100)
        display_cols = ['datetime', 'data_type'] + [f'log_lag_{v}{"h" if u == "hour" else "d"}_available' 
                                                    for v, u in lag_configs]
        
        if len(test_only) > 0:
            sample_data = test_only[display_cols].head(10)
            print(sample_data.to_string(index=False))
    
    return analysis_df