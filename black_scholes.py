import numpy as np
from scipy.stats import norm

class BlackScholes:
    def __init__(self, S, K, T, r, sigma):
        """
        S     = Current stock price
        K     = Strike price
        T     = Time to expiry in years (e.g. 0.5 = 6 months)
        r     = Risk-free interest rate (e.g. 0.06 for 6%)
        sigma = Volatility (annualised std dev of returns)
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def d1(self):
        return (np.log(self.S / self.K) +
                (self.r + 0.5 * self.sigma**2) * self.T) / \
               (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def call_price(self):
        return (self.S * norm.cdf(self.d1()) -
                self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2()))

    def put_price(self):
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2()) -
                self.S * norm.cdf(-self.d1()))
            # Delta
    def call_delta(self):
        return norm.cdf(self.d1())

    def put_delta(self):
        return norm.cdf(self.d1()) - 1

    # Gamma
    def gamma(self):
        return norm.pdf(self.d1()) / (
            self.S * self.sigma * np.sqrt(self.T)
        )

    # Vega
    def vega(self):
        return (
            self.S *
            norm.pdf(self.d1()) *
            np.sqrt(self.T)
        ) / 100

    # Theta
    def call_theta(self):

        term1 = (
            -self.S *
            norm.pdf(self.d1()) *
            self.sigma
        ) / (2 * np.sqrt(self.T))

        term2 = (
            self.r *
            self.K *
            np.exp(-self.r * self.T) *
            norm.cdf(self.d2())
        )

        return (term1 - term2) / 365

    def put_theta(self):

        term1 = (
            -self.S *
            norm.pdf(self.d1()) *
            self.sigma
        ) / (2 * np.sqrt(self.T))

        term2 = (
            self.r *
            self.K *
            np.exp(-self.r * self.T) *
            norm.cdf(-self.d2())
        )

        return (term1 + term2) / 365

    # Rho
    def call_rho(self):
        return (
            self.K *
            self.T *
            np.exp(-self.r * self.T) *
            norm.cdf(self.d2())
        ) / 100

    def put_rho(self):
        return (
            -self.K *
            self.T *
            np.exp(-self.r * self.T) *
            norm.cdf(-self.d2())
        ) / 100

# Quick test — run this file to verify it works
if __name__ == "__main__":
    bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
    print(f"Call Price: ₹{bs.call_price():.2f}")
    print(f"Put Price:  ₹{bs.put_price():.2f}")
        