import plotly.express as px
import pandas as pd



def plot_experiment_trends(summary_df):

    """Returns a line plot object of the experiment group comparison of control vs treatment

    Parameter:
    ---------
    summary_df: pd.DataFrame

    Returns
    --------
    plotly.graph_objects.Figure
        Line plot
    """

    plot = px.line(summary_df,
              x='date',
              y='means',
              color='group',
              title = 'Mean Conversion Rates by Test Group'
              ).update_yaxes(rangemode='tozero')

    return plot


def plot_lift_posteriors(lift_posterior):

    """Returns a line plot object of the experiment group comparison of control vs treatment
    
        Parameter:
        ---------
        lift_posterior: list[numpy.ndarray]
    
        Returns
        --------
        plotly.graph_objects.Figure
            Histogram plot
        """

    hist = px.histogram(lift_posterior, 
                            nbins=100,
                            title='Histogram of lift values from simulated draws from the posterior'
                            ).update_layout(xaxis_tickformat='.0%',
                        xaxis_title='Lift %')

    return hist

