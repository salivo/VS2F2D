import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define fitting functions
def linear(x, a, b):
    return a * x + b

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def root(x, a, b, c):
    return a * np.sqrt(b * x) + c

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Fit a function to data and generate plots.")
    parser.add_argument("-F", "--file", required=True, help="Path to the CSV file.")
    parser.add_argument("-x", "--xaxis", required=True, help="Name of the column for the x-axis.")
    parser.add_argument("-y", "--yaxis", required=True, help="Name of the column for the y-axis.")
    parser.add_argument("-f", "--function", required=True, choices=["linear", "quadratic", "exponential", "root"],
                        help="Type of function to fit: linear, quadratic, exponential, or root.")
    args = parser.parse_args()

    # Load the data
    data = pd.read_csv(args.file)
    if args.xaxis not in data.columns or args.yaxis not in data.columns:
        raise ValueError(f"Specified columns '{args.xaxis}' or '{args.yaxis}' not found in the file.")

    x_data = data[args.xaxis]
    y_data = data[args.yaxis]

    # Select the fitting function
    if args.function == "linear":
        func = linear
        p0 = [1, 1]  # Initial guess for linear
    elif args.function == "quadratic":
        func = quadratic
        p0 = [1, 1, 1]  # Initial guess for quadratic
    elif args.function == "exponential":
        func = exponential
        p0 = [1, 1, 1]  # Initial guess for exponential
    elif args.function == "root":
        func = root
        p0 = [1, 1, 1]  # Initial guess for root

    # Fit the curve
    popt, pcov = curve_fit(func, x_data, y_data, p0=p0)

    # Generate fitted values
    x_fit = np.linspace(x_data.min(), x_data.max(), 500)
    y_fit = func(x_fit, *popt)

    # Save the fitted data for LaTeX
    fitted_data = pd.DataFrame({args.xaxis: x_fit, args.yaxis: y_fit})
    fitted_file = "fitted_curve.csv"
    fitted_data.to_csv(fitted_file, index=False)
    print(f"Fitted data saved to: {fitted_file}")

    # Plot the original data and fitted curve
    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, label="Data", color="blue")
    plt.plot(x_fit, y_fit, label=f"Fitted {args.function} curve", color="red")
    plt.xlabel(args.xaxis)
    plt.ylabel(args.yaxis)
    plt.legend()
    plt.grid(True)
    plt.title("Data and Fitted Curve")
    plt.show()

if __name__ == "__main__":
    main()
