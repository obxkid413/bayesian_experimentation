import streamlit as st
from experiment import ExperimentSimulator, ExperimentAnalysis
from visualization import plot_experiment_trends
from visualization import plot_lift_posteriors
import numpy as np

st.set_page_config(
    layout="wide"
)

col1, col2 = st.columns(2)

st.title("Bayesian Experimentation Platform")

st.caption(
    "Interactive A/B testing simulation and statistical analysis."
)
st.caption("The goal of this application was to better understand how a Bayesian testing platform might work, both from a " \
"practical user interface standpoint and from a true business value perspective.")

st.divider()

st.markdown("""
A couple important notes about this app:
- Underlying treatment conversion rate was set to 12% and the underlying control conversion rate was set to 10%, with randomness added in via binomial distribution
- A prior alpha value was set to 100 and beta was set to 900, representing the baseline or control conversion rate we expect given historical data
- Treatment and control groups are assumed to be independent when generating the posterior distributions
"""
)

st.divider()

if st.button('Run Simulation'):

    experiment = ExperimentSimulator(daily_visitors = 30,
                                 control_conversion_rate = .10,
                                 treatment_conversion_rate = .12,
                                 start_date = '2024-06-01',
                                 duration_days = 30
        )


    experiment_df = experiment.run_simulator()

    st.session_state.experiment_data = experiment_df 

    group_comparison = experiment.generate_group_comparison(st.session_state.experiment_data)

    st.session_state.comparison = group_comparison

    analysis = ExperimentAnalysis(st.session_state.experiment_data, prior_alpha=100, prior_beta=900)
    
    all_posteriors = analysis.generate_posterior_lifts()
    
    st.session_state.posterior_samples = all_posteriors

if "posterior_samples" not in st.session_state:
    st.stop()

latest_results = st.session_state.experiment_data

st.session_state.summary = {
    "control_rate": latest_results["control_conversion_rate"].iloc[-1],
    "treatment_rate": latest_results["treatment_conversion_rate"].iloc[-1],
    "lift": latest_results["lift"].iloc[-1],
    "control_conversions": latest_results["cum_conversions_control"].iloc[-1],
    "treatment_conversions": latest_results["cum_conversions_treatment"].iloc[-1],
    "control_visitors": latest_results["cum_control_n"].iloc[-1],
    "treatment_visitors": latest_results["cum_treatment_n"].iloc[-1],
}
left, right = st.columns([1,1])

with left:
    with st.container(border=True, height=600):
        st.subheader("Trend in Conversion Rates")
        st.plotly_chart(plot_experiment_trends(st.session_state.comparison),
                        use_container_width=False
                        )
with right:
    with st.container(border=True, height=600):
        st.subheader("Experiment Summary")
        st.metric("Latest Control Conversion Rate", f"{st.session_state.summary['control_rate']:.2%}")
        st.metric("Latest Treatment Conversion Rate", f"{st.session_state.summary['treatment_rate']:.2%}")
        st.metric("Latest Lift %", f"{st.session_state.summary['lift']:.2%}")
        st.metric("Total Control Conversions", st.session_state.summary['control_conversions'])
        st.metric("Total Treatment Conversions", st.session_state.summary['treatment_conversions'])
        st.metric("Total Visitors - Control", st.session_state.summary['control_visitors'])
        st.metric("Total Visitors - Treatment", st.session_state.summary['treatment_visitors'])


st.divider()
st.subheader("Posterior Lift Analysis")

#if "posterior_samples" in st.session_state:
select_day = st.slider('Select Day', min_value=1,
                           max_value=len(st.session_state.posterior_samples),
                            value=len(st.session_state.posterior_samples) 
                            )
posterior_sample = st.session_state.posterior_samples[select_day - 1]

mean = posterior_sample.mean()

median = np.median(posterior_sample)

ci_lower = np.percentile(posterior_sample, 2.5)
ci_upper = np.percentile(posterior_sample, 97.5)

fig = plot_lift_posteriors(posterior_sample)

st.plotly_chart(fig)


with st.container(border=True):

    st.subheader(f"Posterior Lift Summary for Day {select_day}")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Posterior Mean", f"{mean:.2%}")
        st.metric("Posterior Median", f"{median:.2%}")

    with c2:
        st.write(
            f"**95% Credible Interval for Lift:** "
            f"({ci_lower:.2%}, {ci_upper:.2%})"
        )

    selected_lift = st.slider('Lift Value', 
                            min_value=float(posterior_sample.min()),
                            max_value = float(np.max(posterior_sample)),
                            value = float(posterior_sample.min())
    )

    
    p_lift = np.mean(posterior_sample > selected_lift)

    st.metric(
        f"P(Lift > {selected_lift} )",
        f"{p_lift:.1%}"
    )



st.markdown("""
A couple other considerations might be:
- You could envision knowing the lift needed to break even prior to running the test so this would tell you how likely it is you achieve that result.
- In addition to just having a posterior lift distribution, you could also incorporate costs of implementation to get an ROI distribution to see how likely it is you get a positive ROI.
"""
)





