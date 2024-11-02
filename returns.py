def calculate_investment_returns(
    initial_investment,
    annual_return_rate,
    annual_increase_rate,
    years
):
    """
    Calculate investment returns with monthly investments that increase annually.
    
    Parameters:
    initial_investment (float): Initial monthly investment amount
    annual_return_rate (float): Annual return rate (as decimal)
    annual_increase_rate (float): Annual increase in monthly investment (as decimal)
    years (int): Investment period in years
    
    Returns:
    float: Final investment amount
    """
    months = years * 12
    monthly_return_rate = annual_return_rate / 12
    final_amount = 0
    current_monthly_investment = initial_investment
    
    for month in range(months):
        # Add this month's investment
        final_amount += current_monthly_investment
        
        # Apply compound interest to entire balance
        final_amount *= (1 + monthly_return_rate)
        
        # Increase monthly investment by annual_increase_rate every 12 months
        if (month + 1) % 12 == 0:
            current_monthly_investment *= (1 + annual_increase_rate)
    
    return final_amount

# Test the function with your parameters
initial_investment = 15000  # Initial monthly investment in INR
annual_return_rate = 0.24  # Annual return rate (12%)
annual_increase_rate = 0.20  # Annual increase rate (10%)
years = 20  # Investment period in years

result = calculate_investment_returns(
    initial_investment,
    annual_return_rate,
    annual_increase_rate,
    years
)

print(f"Final amount after {years} years: ₹{result:,.2f}")

# To verify monthly progression, here's a function to show year-by-year totals
def show_yearly_progression(
    initial_investment,
    annual_return_rate,
    annual_increase_rate,
    years
):
    monthly_return_rate = annual_return_rate / 12
    current_amount = 0
    current_monthly_investment = initial_investment
    
    for year in range(1, years + 1):
        for month in range(12):
            current_amount += current_monthly_investment
            current_amount *= (1 + monthly_return_rate)
        
        print(f"Year {year}: ₹{current_amount:,.2f} "
              f"(Monthly investment: ₹{current_monthly_investment:,.2f})")
        current_monthly_investment *= (1 + annual_increase_rate)

# Show yearly progression
show_yearly_progression(
    initial_investment,
    annual_return_rate,
    annual_increase_rate,
    years
)