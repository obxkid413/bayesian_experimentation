#### Bayesian A/B Testing Simulator

An interactive Streamlit app that allows a user to simulate some A/B testing data and see how Bayesian inference could be used to make actual business decisions.  This app simulates random treatment and control groups, updates the posterior distribution, and provides plots so the user can explore how they changed over time for that particular experiment.  Note that the underlying conversion rates for treatment and control are set to 12% and 10%, respectively.

#### Features

- Simulate randomized A/B test data
- Visulization of performance for treatment and control groups
- Overall summary statistics of performance 
- Visualization of posterior distributions (can view the posterior at different points in time)
- Posterior mean, median, credible intervals provided 
- Adjustable lift thresholds to see what the probability is that the lift exceeds that threshold 

#### Methodology 

The methods used here are important to discuss and make transparent.  I want to point out a couple things that I think will make things more clear.  In this particular app, the treatment and control groups are treated as independent groups.  Based on my reading, this is certainly defensible, but I have also read that it can be better to not make this assumption. Without this assumption, the calculations for generating the posterior becomes a bit more complicated.  The posterior lift distribution was computed as follows from the control and treatment posteriors:

[
\text{Lift} = \frac{p_{\text{treatment}} - p_{\text{control}}}{p_{\text{control}}}
]

In addition, you'll notice I used the beta distribution as a prior.  The beta distribution reflects the actual type of data we are dealing with (e.g. conversion rates thus cannot be less than zero).  In addition, a nice perk of using this distribution is that it is a conjugate of the binomial distribution.  Because of this property, updating the beta prior with new data becomes simple addition as you can see in the code.  The assumption here is that the underlying observed data from the experiment is really a binomial process, that is, someone either converts or they do not convert.  

For this particular exercise, I did not use PyMC which is another library that can be used for more complex modeling of posteriors.  


#### Libraries used here include:
 - Streamlit for visualization and app dev
 - Pandas 
 - Numpy
 - Scipy for a couple pieces not currently in the app yet
 - Plotly


#### Project Structure
.
├── app.py                 # Streamlit application
├── experiment.py          # Experiment simulation and Bayesian analysis
├── visualization.py       # Plotting functions
├── requirements.txt
└── README.md

#### Running the Application

Install the project dependencies:

pip install -r requirements.txt

Launch the Streamlit application:

streamlit run app.py


#### Future Enhancements

Potential improvements include:

- Add expected loss as a metric
- Configurable prior distributions

I wanted to show expected loss in this iteration but I wanted to read more about that metric and how to interpret it more before inserting it here. From what little I understand so far, it's basically telling you the cost associated with choosing the wrong variant as the "winner".  