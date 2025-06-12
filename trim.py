import argparse
import csv
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Trim acceleration data interactively.")
    parser.add_argument("-F", "--file", required=True, help="Path to the CSV file.")
    parser.add_argument("-r", "--row", type=int, default=2, help="Row position to extract: 1=X, 2=Y, 3=Z, 4=ABS (default: 2)")
    parser.add_argument("--delimiter", default=';', help="CSV delimiter (default: ';')")
    parser.add_argument("--decimal", default=',', help="Decimal separator (default: ',')")
    return parser.parse_args()

def load_data(file_path, row_index, delimiter, decimal):
    time = []
    accel = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter, quotechar='"')
        for row in reader:
            if len(row) >= row_index:
                try:
                    t = float(row[0].replace(decimal, '.'))
                    a = float(row[row_index - 1].replace(decimal, '.'))
                    time.append(t)
                    accel.append(a)
                except ValueError:
                    continue
    return time, accel

def save_trimmed_csv(time_data, accel_data, path, delimiter, decimal):
    with open(path, 'w', newline='') as f:
        f.write(f'"time"{delimiter}"accel"\n')
        writer = csv.writer(f, delimiter=delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for t, a in zip(time_data, accel_data):
            writer.writerow([str(t).replace('.', decimal), str(a).replace('.', decimal)])
    print(f"Saved trimmed CSV to: {path} ({len(time_data)} points)")

def interactive_trim(time, accel, save_callback):
    def onselect(xmin, xmax):
        trimmed_time = [t for t in time if xmin <= t <= xmax]
        trimmed_accel = [a for t, a in zip(time, accel) if xmin <= t <= xmax]

        ax2.clear()
        ax2.plot(trimmed_time, trimmed_accel)
        ax2.set_title(f"Trimmed Data ({xmin:.2f}s to {xmax:.2f}s)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Acceleration (m/s²)")
        ax2.grid(True)
        fig.canvas.draw_idle()

        save_callback(trimmed_time, trimmed_accel)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(time, accel)
    ax1.set_title("Select Time Range to Trim")
    ax1.set_ylabel("Acceleration (m/s²)")
    ax1.grid(True)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Acceleration (m/s²)")
    ax2.grid(True)

    _ = SpanSelector(ax1, onselect, 'horizontal', useblit=True, props=dict(alpha=0.5, facecolor='red'), interactive=True)
    plt.tight_layout()
    plt.show()

def main():
    args = parse_arguments()

    time_data, accel_data = load_data(args.file, args.row, args.delimiter, args.decimal)
    trimed_file = f"{os.path.splitext(args.file)[0]}_trimed.csv"
    interactive_trim(time_data, accel_data,
        lambda t, a: save_trimmed_csv(t, a, trimed_file, args.delimiter, args.decimal))

if __name__ == "__main__":
    main()
