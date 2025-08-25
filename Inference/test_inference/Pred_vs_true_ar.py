import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.linear_model import LinearRegression
import numpy as np
import argparse

def create_interactive_plot(df, output_path, title, color_by_uc=False, opacity=1.0):
    df = df.dropna(subset=['12_COMPUTED_lacune_ratio_percent', 'AR_percent_in_Cortex_only'])

    layout = go.Layout(
        title=title,
        xaxis=dict(title='True AR % (GT)', automargin=True),
        yaxis=dict(title='Predicted AR %', automargin=True),
        legend=dict(x=0.01, y=0.99),
        autosize=True,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    fig = go.Figure(layout=layout)

    if color_by_uc:
        for uc_name, group in df.groupby('UC'):
            num_points = len(group)
            fig.add_trace(go.Scatter(
                x=group['12_COMPUTED_lacune_ratio_percent'],
                y=group['AR_percent_in_Cortex_only'],
                mode='markers',
                name=f"{uc_name} (N = {num_points})",
                marker=dict(size=10, opacity=opacity),
                hoverinfo='text',
                text=(
                    "Image: " + group['0_PARAM_ImgName'].astype(str) +
                    "<br>Experiment: " + group['Experiment'].astype(str) +
                    "<br>UC: " + group['UC'].astype(str) +
                    "<br>True AR: " + group['12_COMPUTED_lacune_ratio_percent'].round(2).astype(str) + "%" +
                    "<br>Predicted AR: " + group['AR_percent_in_Cortex_only'].round(2).astype(str) + "%"
                )
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df['12_COMPUTED_lacune_ratio_percent'],
            y=df['AR_percent_in_Cortex_only'],
            mode='markers',
            name='Points',
            marker=dict(size=10, opacity=opacity),
            hoverinfo='text',
            text=(
                "Image: " + df['0_PARAM_ImgName'].astype(str) +
                "<br>Experiment: " + df['Experiment'].astype(str) +
                "<br>UC: " + df['UC'].astype(str) +
                "<br>True AR: " + df['12_COMPUTED_lacune_ratio_percent'].round(2).astype(str) + "%" +
                "<br>Predicted AR: " + df['AR_percent_in_Cortex_only'].round(2).astype(str) + "%"
            )
        ))

    # Regression Line
    x = df['12_COMPUTED_lacune_ratio_percent'].values.reshape(-1, 1)
    y = df['AR_percent_in_Cortex_only'].values
    reg = LinearRegression().fit(x, y)
    x_range = np.linspace(x.min(), x.max(), 100)
    y_pred_line = reg.predict(x_range.reshape(-1, 1))
    r2 = reg.score(x, y)

    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_pred_line,
        mode='lines',
        name=f'Regression Line (R² = {r2:.2f})',
        line=dict(color='blue', dash='solid')
    ))

    # Perfect Prediction Line x = y
    min_val = min(df['12_COMPUTED_lacune_ratio_percent'].min(), df['AR_percent_in_Cortex_only'].min())
    max_val = max(df['12_COMPUTED_lacune_ratio_percent'].max(), df['AR_percent_in_Cortex_only'].max())

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction (x = y)',
        line=dict(color='gray', dash='dot')
    ))

    fig.update_layout(autosize=True)
    fig.write_html(output_path, full_html=True)
    print(f"Saved plot to: {output_path}")

def generate_all_plots(root_dir, opacity=1.0):
    all_data = []
    
    for uc_folder in sorted(os.listdir(root_dir)):
        uc_path = os.path.join(root_dir, uc_folder)
        if not os.path.isdir(uc_path):
            continue

        uc_data = []
        for exp_folder in sorted(os.listdir(uc_path)):
            exp_path = os.path.join(uc_path, exp_folder)
            cmp_path = os.path.join(exp_path, 'comparison_metrics.csv')
            if not os.path.isfile(cmp_path):
                continue

            df = pd.read_csv(cmp_path)
            df['UC'] = uc_folder
            df['Experiment'] = exp_folder
            uc_data.append(df)
            all_data.append(df)

        if uc_data:
            uc_df = pd.concat(uc_data, ignore_index=True)
            create_interactive_plot(
                uc_df,
                output_path=os.path.join(uc_path, 'plot.html'),
                title=f'AR Prediction vs Ground Truth - {uc_folder}',
                color_by_uc=False,
                opacity=opacity
            )

    # All UC combined
    if all_data:
        all_df = pd.concat(all_data, ignore_index=True)
        create_interactive_plot(
            all_df,
            output_path=os.path.join(root_dir, 'plot_all_uc.html'),
            title='AR Prediction vs Ground Truth - Test Data - All UCs',
            color_by_uc=True,
            opacity=opacity
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='path/to/all_use_cases_folder', help='Path to the folder containing UC folders')
    parser.add_argument('--opacity', type=float, default=0.7, help='Marker opacity (0.0 to 1.0)')
    args = parser.parse_args()

    generate_all_plots(args.root_dir, opacity=args.opacity)
