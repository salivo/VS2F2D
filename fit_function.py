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

def reciprocal(x, a, b, c):
    return a / (x + b) + c

def logarithmic(x, a, b):
    return a * np.log(x) + b

def sine(x, a, b, c, d):
    return a * np.sin(b * x + c) + d
def print_function_string(func_name, params):
    if func_name == "linear":
        a, b = params
        print(f"Fitted function: {a:.4g}x + {b:.4g}")
    elif func_name == "quadratic":
        a, b, c = params
        print(f"Fitted function: {a:.4g}x^2 + {b:.4g}x + {c:.4g}")
    elif func_name == "exponential":
        a, b, c = params
        print(f"Fitted function: {a:.4g} * exp({b:.4g}x) + {c:.4g}")
    elif func_name == "root":
        a, b, c = params
        print(f"Fitted function: {a:.4g} * sqrt({b:.4g}x) + {c:.4g}")
    elif func_name == "reciprocal":
        a, b, c = params
        print(f"Fitted function: {a:.4g} / (x + {b:.4g}) + {c:.4g}")
    elif func_name == "logarithmic":
        a, b = params
        print(f"Fitted function: {a:.4g} * ln(x) + {b:.4g}")
    elif func_name == "sine":
        a, b, c, d = params
        print(f"Fitted function: {a:.4g} * sin({b:.4g}x + {c:.4g}) + {d:.4g}")
    else:
        print("Function is not implemented")

def guess_sinusoidal_params(x, y):
    a_guess = (np.max(y) - np.min(y)) / 2
    y_detrended = y - np.mean(y)
    fft_freq = np.fft.fftfreq(len(x), d=(x[1] - x[0]))
    fft_magnitude = np.abs(np.fft.fft(y_detrended))
    peak_index = np.argmax(fft_magnitude[1:]) + 1
    freq_guess = 2 * np.pi * np.abs(fft_freq[peak_index])
    c_guess = 0
    d_guess = np.mean(y)
    return [a_guess, freq_guess, c_guess, d_guess]

def main():
    parser = argparse.ArgumentParser(description="Fit a function to data and generate plots.")
    parser.add_argument("-F", "--file", required=True, help="Path to the CSV file.")
    parser.add_argument("-x", "--xaxis", required=True, help="Name of the column for the x-axis.")
    parser.add_argument("-y", "--yaxis", required=True, help="Name of the column for the y-axis.")
    parser.add_argument("-f", "--function", required=True,
                        choices=["linear", "quadratic", "exponential", "root", "reciprocal", "logarithmic", "sine"],
                        help="Type of function to fit.")
    parser.add_argument("--delimiter", default=';', help="CSV delimiter (default: ';')")
    parser.add_argument("--decimal", default=',', help="Decimal separator (default: ',')")
    args = parser.parse_args()

    data = pd.read_csv(args.file, delimiter=args.delimiter, decimal=args.decimal)
    if args.xaxis not in data.columns or args.yaxis not in data.columns:
        raise ValueError(f"Specified columns '{args.xaxis}' or '{args.yaxis}' not found in the file.")

    x_data = data[args.xaxis]
    y_data = data[args.yaxis]

    # Select fitting function
    func_map = {
        "linear": (linear, [1, 1]),
        "quadratic": (quadratic, [1, 1, 1]),
        "exponential": (exponential, [1, -0.01, 1]),
        "root": (root, [1, 1, 1]),
        "reciprocal": (reciprocal, [1, 1, 1]),
        "logarithmic": (logarithmic, [1, 1]),
        "sine": (sine, guess_sinusoidal_params(x_data.to_numpy(), y_data.to_numpy()))
    }

    func, p0 = func_map[args.function]

    # Fit the curve
    popt, pcov = curve_fit(func, x_data, y_data, p0=p0)

    # Print fitted function
    print_function_string(args.function, popt)

    # Generate fitted values
    x_fit = np.linspace(x_data.min(), x_data.max(), 500)
    y_fit = func(x_fit, *popt)

    # Save fitted data
    fitted_data = pd.DataFrame({args.xaxis: x_fit, args.yaxis: y_fit})
    fitted_file = "fitted_curve.csv"
    fitted_data.to_csv(fitted_file, index=False, sep=args.delimiter, decimal=args.decimal)
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
