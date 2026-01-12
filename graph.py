# Creating a small, ready-to-run Python snippet that:
# - validates up to 10 numeric x and y values
# - displays a scatter plot using matplotlib (single plot, no custom colors/styles)
# - shows the data in a table for convenience
# - saves the plot to /mnt/data/xy_scatter.png so you can download it

import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from caas_jupyter_tools import display_dataframe_to_user

def plot_xy(x, y, labels=None, title="test color (R, G, B) norm", save_path="xy_scatter.png"):
    # Validate inputs
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) == 0:
        raise ValueError("x and y must contain at least one value")
    if len(x) > 10:
        raise ValueError("Maximum supported points is 10 (you provided {}).".format(len(x)))
    
    # Convert to numeric arrays
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    
    # Create a dataframe and display it to user
    df = pd.DataFrame({"x": x_arr, "y": y_arr})
    if labels is not None:
        if len(labels) != len(x):
            raise ValueError("labels length must match x/y")
        df["label"] = labels
    #display_dataframe_to_user("x_y_data", df)
    
    # Plot (one chart, matplotlib, no custom colors or styles)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(x_arr, y_arr)
    #ax.set_xlabel("x")
    #ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    
    # Annotate points with labels or index+values
    for i, (xx, yy) in enumerate(zip(x_arr, y_arr)):
        label_text = labels[i] if labels is not None else f"{i}: ({xx:.2f}, {yy:.2f})"
        # place annotation slightly offset
        ax.annotate(label_text, (xx, yy), textcoords="offset points", xytext=(4,4), fontsize=8)
    
    # Auto-scale limits with small margins
    xpad = (x_arr.max() - x_arr.min()) * 0.06 if len(x_arr) > 1 else 0.5
    ypad = (y_arr.max() - y_arr.min()) * 0.06 if len(y_arr) > 1 else 0.5
    ax.set_xlim(x_arr.min() - xpad, x_arr.max() + xpad)
    ax.set_ylim(y_arr.min() - ypad, y_arr.max() + ypad)
    
    # Save and show
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved to: {save_path}")

# Test mean LAB and BGR values of ketones
LAB = ([19.17348771, 2.87153028, 7.54582733],
        [39.1897007, 1.15551073, 7.27685066],
        [43.16282673, 0.37472653, 8.09133061],
        [21.87052599, 2.51523109, 7.2355859],
        [44.22907849, 0.64030514, 8.29531859],
        [48.81376906, -0.53298765, 9.44725926])

RGB = ([35.48823901, 44.48634643, 54.48733778],
        [80.27342747, 91.03343725, 99.1612568],
        [88.69766531, 101.22592653, 108.09733878],
        [41.78635621, 50.73138422, 60.34010271],
        [90.61291916, 103.70023599, 111.38614785],
        [100.18355556, 115.51817284, 121.67649383])

# Test mean LAB and BGR values of proteins
LAB_p = ([23.56372477, 1.39517912, 5.67595459],
        [42.61598636, -9.25408764, 7.87141923],
        [47.28010091, -14.86665661, 6.01202153],
        [27.02820493, -0.48136871, 7.61910976],
        [43.5553356, -7.14034219, 5.32692308],
        [53.68893578, -12.85735471, 4.10493609])

RGB_p = ([47.66371812, 55.06707946, 62.09789179],
        [87.62119032, 104.68920863, 89.92164814],
        [101.43202052, 118.94648157, 88.40018108],
        [51.96665217, 63.47426417, 68.28389155],
        [93.52502828, 106.09799208, 94.0436934],
        [120.52722265, 134.44071746, 106.37094838])

L_y_example = [math.sqrt(x[0] * x[0]) for x in LAB_p]
LAB_y_example = [math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) for x in LAB_p]
RGB_y_example = [math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) for x in RGB_p]

x_example = [1, 2, 3, 4, 5, 6]
#y_example = [2, 1.5, 3, 4.5, 2.8, 5]
labels_example = ["up_0","up_0,5","up_1","side_0","side_0,5","side_1"]
y_example = RGB_y_example
plot_xy(x_example, y_example, labels=labels_example)
