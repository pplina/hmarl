import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_data(filepath):
    # Read data to pandas df
    df = pd.read_csv(filepath)
    # Remove nan values
    return df


def generate_plots(filepath, x_value, y_values, export_to_file=False):
    df = process_data(filepath)

    # df.plot(x="time/time_elapsed", y="train/value_loss", kind="line")
    # plt.plot(df["time/time_elapsed"], df["train/value_loss"])

    plot_data_columns = {}

    plot_data_columns[x_value.split('/')[-1]] = df[x_value]
    for y in y_values:
        plot_data_columns[y.split('/')[-1]] = df[y]

    plot_data = pd.DataFrame(plot_data_columns)

    # plot_data = pd.DataFrame({'steps': df['time/total_timesteps'], 'value_loss': df['train/value_loss'],
    #                           'ep_rew_mean': df['rollout/ep_rew_mean']})

    plot_data.plot(x='total_timesteps', style='.-')

    if export_to_file:
        filename = x_value.split('/')[-1] + "," + ",".join(y_val.split('/')[-1] for y_val in y_values)
        plt.savefig(filename + '.png')

    plt.show()


if __name__ == '__main__':
    filepath = "../tb_log/progress.csv"
    print("Available Data Columns:")
    col_str = ", ".join(col for col in process_data(filepath).columns)
    print(col_str)

    generate_plots(filepath, x_value='time/total_timesteps', y_values=['train/value_loss', 'rollout/ep_rew_mean'],
                   export_to_file=True)