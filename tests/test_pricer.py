import pytest
from src.options_pricer import black_scholes

@pytest.fixture
def call_inputs():
    return {
        'S': 100,       # Stock Price
        'K': 100,       # Strike Price
        'T': 1.0,       # Time (1 year)
        'r': 0.05,      # Risk-free rate
        'sigma': 0.2    # Volatility
    }
# ---------------------------------
#           UNIT TESTS
# ---------------------------------
def test_call_option_value(call_inputs):
    """
    Test against a known value (~$10.45).
    """
    price = black_scholes(
        call_inputs['S'], 
        call_inputs['K'], 
        call_inputs['T'], 
        call_inputs['r'], 
        call_inputs['sigma'], 
        'call'
    )
    expected_price = 10.45
    
    # pytest uses simple 'assert' statements
    assert price == pytest.approx(expected_price, abs=0.05)

def test_put_call_parity_logic(call_inputs):
    """
    Logic Check: If Stock Price is massive (Deep ITM), 
    Call Price should be roughly (Stock - Strike).
    """
    high_stock_price = 1000
    price = black_scholes(
        high_stock_price, 
        call_inputs['K'], 
        0.01, 
        call_inputs['r'], 
        call_inputs['sigma'], 
        'call'
    )
    # Intrinsic value = 1000 - 100 = 900
    assert price > 899, "Deep ITM Call should be worth at least intrinsic value"

def test_deep_otm_is_zero(call_inputs):
    """
    Logic Check: If Stock is 0, Call Option should be 0.
    """
    price = black_scholes(
        1, 
        1000, 
        0.1, 
        call_inputs['r'], 
        call_inputs['sigma'], 
        'call'
    )
    assert price < 0.01, "Deep OTM option should be worthless"