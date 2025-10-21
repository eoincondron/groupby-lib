Examples
========

This section provides practical examples of using groupby-lib for various data analysis tasks.

Basic Aggregations
------------------

Sum, Mean, and Count::

    from groupby_lib.groupby import GroupBy
    import pandas as pd
    import numpy as np
    
    # Sample sales data
    df = pd.DataFrame({
        'region': ['North', 'South', 'North', 'East', 'South', 'East'],
        'sales': [100, 150, 200, 120, 180, 90],
        'quantity': [10, 15, 20, 12, 18, 9]
    })
    
    # Group by region
    gb = GroupBy(df['region'])
    
    total_sales = gb.sum(df['sales'])
    avg_sales = gb.mean(df['sales'])
    count_orders = gb.count()
    
    print("Total sales by region:", total_sales.values)
    print("Average sales by region:", avg_sales.values)
    print("Order count by region:", count_orders.values)

Working with Different Data Types
----------------------------------

Numeric and Categorical Data::

    # Mixed data types
    data = pd.DataFrame({
        'category': pd.Categorical(['A', 'B', 'A', 'C', 'B', 'A']),
        'values': [1.5, 2.3, 1.8, 3.2, 2.1, 1.9],
        'counts': [10, 20, 15, 25, 18, 12]
    })
    
    gb = GroupBy(data['category'])
    
    # Multiple aggregations
    results = {
        'sum_values': gb.sum(data['values']),
        'mean_values': gb.mean(data['values']),
        'total_counts': gb.sum(data['counts'])
    }

Time Series Grouping
---------------------

Grouping by Date Components::

    # Time series data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    values = np.random.randn(100).cumsum()
    
    df_ts = pd.DataFrame({
        'date': dates,
        'value': values,
        'month': dates.month,
        'weekday': dates.day_name()
    })
    
    # Group by month
    gb_month = GroupBy(df_ts['month'])
    monthly_avg = gb_month.mean(df_ts['value'])
    
    # Group by weekday
    gb_weekday = GroupBy(df_ts['weekday'])
    weekday_avg = gb_weekday.mean(df_ts['value'])

Performance Optimization
------------------------

Working with Large Datasets::

    # Generate large dataset
    n = 5_000_000
    groups = np.random.randint(0, 10000, n)
    values = np.random.randn(n)
    
    # Convert to pandas for comparison
    df_large = pd.DataFrame({
        'group': groups,
        'value': values
    })
    
    # groupby-lib (optimized)
    gb = GroupBy(pd.Series(groups))
    fast_result = gb.sum(pd.Series(values))
    
    # This is typically 2-5x faster than pandas.groupby()

Custom Aggregations
-------------------

Using Multiple Operations::

    from groupby_lib.groupby import GroupBy
    
    # Sample data
    sales_data = pd.DataFrame({
        'store': ['A', 'B', 'A', 'C', 'B', 'A', 'C'],
        'revenue': [1000, 1500, 800, 1200, 1800, 900, 1100],
        'customers': [50, 75, 40, 60, 90, 45, 55]
    })
    
    gb = GroupBy(sales_data['store'])
    
    # Multiple metrics per group
    store_metrics = {
        'total_revenue': gb.sum(sales_data['revenue']),
        'avg_revenue': gb.mean(sales_data['revenue']),
        'total_customers': gb.sum(sales_data['customers']),
        'avg_customers': gb.mean(sales_data['customers'])
    }
    
    # Create summary dataframe
    summary = pd.DataFrame({
        'store': store_metrics['total_revenue'].index,
        'total_revenue': store_metrics['total_revenue'].values,
        'avg_revenue': store_metrics['avg_revenue'].values,
        'total_customers': store_metrics['total_customers'].values,
        'avg_customers': store_metrics['avg_customers'].values
    })

Memory Efficient Operations
---------------------------

For very large datasets, consider chunked processing::

    def process_large_file(filename, chunk_size=100_000):
        """Process large CSV in chunks with groupby-lib"""
        results = []
        
        for chunk in pd.read_csv(filename, chunksize=chunk_size):
            gb = GroupBy(chunk['group_column'])
            chunk_result = gb.sum(chunk['value_column'])
            results.append(chunk_result)
        
        # Combine results (would need custom merging logic)
        return results

Integration with Plotting
--------------------------

Visualizing GroupBy Results::

    import matplotlib.pyplot as plt
    from groupby_lib.plotting import group_scatter
    
    # Group and plot results
    gb = GroupBy(df['category'])
    means = gb.mean(df['values'])
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(means)), means.values)
    plt.xlabel('Groups')
    plt.ylabel('Mean Values')
    plt.title('Mean Values by Group')
    plt.show()
    
    # Use built-in plotting functions
    group_scatter(df['category'], df['values'], alpha=0.7)