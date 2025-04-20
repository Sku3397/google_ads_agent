# Portfolio Optimization Service

The Portfolio Optimization Service provides algorithms for optimizing Google Ads campaign budgets across multiple campaigns to maximize overall performance metrics such as conversions, clicks, or ROAS (Return on Ad Spend).

## Key Features

- **Campaign Portfolio Optimization**: Allocate budget across campaigns to maximize an objective function (conversions, clicks, ROAS) subject to constraints.
- **Cross-Campaign Keyword Analysis**: Identify keyword overlaps, cannibalization, and performance disparities across campaigns.
- **Time-Based Budget Allocation**: Optimize budget allocation over time, accounting for seasonality and trends.
- **Application of Recommendations**: Apply optimized budget recommendations to campaigns.

## How It Works

The Portfolio Optimization Service uses convex optimization techniques to find the optimal allocation of budget across campaigns that maximizes a given objective function while respecting constraints such as total budget limits and reasonable bounds on budget changes.

### Optimization Process

1. Historical campaign performance data is retrieved from the Google Ads API
2. The data is processed and prepared for optimization
3. A convex optimization problem is formulated:
   - Variables: Campaign budgets
   - Objective: Maximize performance (conversions, clicks, ROAS)
   - Constraints: Total budget, reasonable budget changes
4. The optimization problem is solved using the CVXPY library
5. Recommendations are generated based on the optimal solution
6. Recommendations can be reviewed and applied

## Usage Examples

### Optimize Campaign Portfolio

```python
# Optimize budget allocation to maximize conversions
result = ads_agent.services["portfolio_optimization"].optimize_campaign_portfolio(
    days=30,                     # Use last 30 days of data
    objective="conversions",     # Maximize conversions
    constraint="budget",         # Subject to budget constraint
    budget_limit=1000.0          # Total daily budget limit ($1000)
)

# Optimize budget allocation to maximize ROAS
result = ads_agent.services["portfolio_optimization"].optimize_campaign_portfolio(
    days=30,
    objective="roas",
    constraint="budget",
    budget_limit=1000.0
)

# Get recommendations for specific campaigns
result = ads_agent.services["portfolio_optimization"].optimize_campaign_portfolio(
    days=30,
    objective="conversions",
    campaign_ids=["123456789", "987654321"]
)
```

### Apply Portfolio Recommendations

```python
# Apply portfolio recommendations
result = ads_agent.services["portfolio_optimization"].apply_portfolio_recommendations(
    recommendations=optimization_result["recommendations"]
)
```

### Cross-Campaign Keyword Analysis

```python
# Analyze keywords across campaigns
result = ads_agent.services["portfolio_optimization"].cross_campaign_keyword_analysis(
    days=30,
    campaign_ids=["123456789", "987654321"]
)

# Review overlapping keywords
overlaps = result["overlapping_keywords"]
```

### Time-Based Budget Allocation

```python
# Optimize budget allocation over time
result = ads_agent.services["portfolio_optimization"].optimize_budget_allocation_over_time(
    days=30,                  # Use last 30 days of data
    forecast_days=30,         # Forecast for next 30 days
    objective="conversions"   # Maximize conversions
)
```

## Dependencies

- `pandas`: For data manipulation
- `numpy`: For numerical operations
- `cvxpy`: For convex optimization

## Implementation Notes

The service uses the following key methods:

- `optimize_campaign_portfolio()`: Entry point for portfolio optimization
- `cross_campaign_keyword_analysis()`: Analyzes keywords across campaigns
- `optimize_budget_allocation_over_time()`: Optimizes budget allocation over time
- `apply_portfolio_recommendations()`: Applies recommendations

Internal helper methods:
- `_run_optimization()`: Core optimization algorithm using CVXPY
- `_prepare_recommendations()`: Converts optimization results to actionable recommendations
- `_identify_keyword_overlaps()`: Identifies keywords appearing in multiple campaigns

## Future Enhancements

- Implement portfolio optimization with multiple objectives (multi-objective optimization)
- Add support for more complex constraints (e.g., target ROAS constraints)
- Enhance the time-based allocation with better forecasting models
- Implement portfolio optimization for bidding strategies across campaigns
- Add support for scenario analysis and sensitivity testing 