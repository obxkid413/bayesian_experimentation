from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

@dataclass

class ExperimentSimulator:

    """This class simulates data from a randomized experiment/AB test. It also adds some additional fields like lift
    and standard errors.

    It generates daily records over a specified period in days.  Examples of created metrics include:
        -daily visitors
        -daily conversion counts
        -conversion rates
        -cumulative conversion counts
        -lift 
    """

    daily_visitors: int
    control_conversion_rate: float
    treatment_conversion_rate: float

    start_date: datetime

    duration_days: int


    def run_simulator(self):
        """Simulates data from an A/B test.
        
        Returns
        -------
        pandas.DataFrame
        
        A dataframe containing the experiment metrics organized by day of the experiment
        """

        dfs = []

        dates = pd.date_range(self.start_date, periods = self.duration_days)

        for d in dates:

            date = d.strftime('%Y-%m-%d')

            #Generate randomness based around daily visitors
            daily_visitors = np.random.poisson(lam=self.daily_visitors)
            
            control_n = daily_visitors // 2

            treatment_n = daily_visitors - control_n 
            
            control_conversions = np.random.binomial(n = control_n,
                                                             p = self.control_conversion_rate)
                    
            treatment_conversions = np.random.binomial(n=treatment_n,
                                                               p = self.treatment_conversion_rate)
                    
            results = pd.DataFrame({'date': [date],
                                            'total_visits': [daily_visitors],
                                            'control_n': [control_n],
                                            'treatment_n': [treatment_n],
                                            'conversions_control': [control_conversions],
                                            'conversions_treatment': [treatment_conversions]
                                            })
            
            dfs.append(results)

         
        experiment_df = pd.concat(dfs)

        experiment_df = self.add_agg_fields(experiment_df)

        return experiment_df



    def add_agg_fields(self, df):

        """Adds some additional fields to the base experiment table.  
        
        Parameters
        ----------
        df : pandas.DataFrame
            Output from ``run_simulator()``.

        Returns
        ----------
        pandas.DataFrame

        A dataframe with the added fields. Fields added are:
        -cum_conversions_control (cumulative conversions control group)
        -cum_conversions_treatment (cumulative conversions treatment group)
        -cum_control_n (cumulative count of control group participants)
        -cum_treatment_n (cumulative count of treatment group participants)

        """

        cum_cols = ['conversions_control', 'conversions_treatment', 'control_n', 'treatment_n']

        for i in cum_cols:
            try:
                df[f'cum_{i}'] = df[i].cumsum()
            except:
                print("Something went wrong adding cumulative columns to the dataframe!")

        df['control_conversion_rate'] = df['cum_conversions_control'] / df['cum_control_n']
        df['treatment_conversion_rate'] = df['cum_conversions_treatment'] / df['cum_treatment_n']
            
        #Add in SEs of the conversion rates which we can plot in a line plot.
        df['control_conversion_rate_se'] = np.sqrt((df['control_conversion_rate'] * (1 - df['control_conversion_rate'])) / df['cum_control_n'])
        df['treatment_conversion_rate_se'] = np.sqrt((df['treatment_conversion_rate'] * (1 - df['treatment_conversion_rate'])) / df['cum_treatment_n'])
            
        df['lift'] = df['treatment_conversion_rate'] / df['control_conversion_rate'] - 1
        df['lift_se'] = np.sqrt((df['treatment_conversion_rate_se']**2 / df['control_conversion_rate']**2) + ((df['control_conversion_rate_se']**2 * df['treatment_conversion_rate']**2) / df['control_conversion_rate']**4))

        return df
    

    def generate_group_comparison(self, df):

        """ 
        Parameters
        ----------
        df : pandas.DataFrame
            Output from ``run_simulator()``.
        
        Returns
        ----------
        pandas.DataFrame
        
        A formatted dataframe in long format for easier plotting of the experiment results, showing the difference
        in conversion rates. Standard errors also computed in case confidence intervals need to be shown on the plot. 

        """

        test_results_means = df.melt(id_vars='date',
                               value_vars=['control_conversion_rate', 'treatment_conversion_rate'],
                               var_name='group',
                               value_name='means')
        
        test_results_se = df.melt(id_vars='date',
                                  value_vars=['control_conversion_rate_se', 
                                              'treatment_conversion_rate_se'],
                                              var_name='group_se',
                                              value_name='se')
        
        long_results = pd.concat([test_results_means, test_results_se['se']], axis=1)

        return long_results
    

@dataclass
class ExperimentAnalysis():

    """This class does the analysis piece whereby it will intake the prior, and update using the latest data available
    to generate posterior distributions.
    
    """

    df: pd.DataFrame
    prior_alpha: int
    prior_beta: int


    def generate_posterior_lifts(self):

        """This method computes the posterior distribution of lift.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Output from ``run_simulator()``.
                
        Returns
        ----------
        list[numpy.ndarray]

        A list where each element contains the separate draws from the posterior lift distribution.
        So each row/day of the experiment would have its own separate list of posterior draws.

        """

        lift_dists = []

        for index, row in self.df.iterrows():

            control_alpha_post = self.prior_alpha + row['cum_conversions_control']

            treatment_alpha_post = self.prior_alpha + row['cum_conversions_treatment']

            control_beta_post = (
            self.prior_beta
            + (row['cum_control_n']
            - row['cum_conversions_control'])
                )

            treatment_beta_post = (
            self.prior_beta + (row['cum_treatment_n'] - row['cum_conversions_control'])
            )

            #Now draw from the posteriors
            control_draws = np.random.beta(control_alpha_post,
                                       control_beta_post,
                                       size=10000)
        
            treatment_draws = np.random.beta(treatment_alpha_post,
                                         treatment_beta_post,
                                         size=10000)

            lift_dist = treatment_draws / control_draws - 1

            lift_dists.append(lift_dist)

        return lift_dists


    #not currently using the methods below but could still add these later if desired!


    def expected_loss(self, lift_posterior):

        #Only interested where lift < 0
        filtered_posterior = abs(lift_posterior < 0)
        loss = np.sum(filtered_posterior) / len(filtered_posterior)
        return loss
    
    def exp_frequentist_stats(self):

        #Compute Z statistic for comapring two proportions 
        x1 = self.df['cum_conversions_treatment'].iloc[-1]

        x2 = self.df['cum_conversions_control'].iloc[-1]

        n1 = self.df['cum_treatment_n'].iloc[-1]
    
        n2 = self.df['cum_control_n'].iloc[-1]

        p1 = self.df['treatment_conversion_rate'].iloc[-1]
        p2 = self.df['control_conversion_rate'].iloc[-1]


        pooled_proportion = (x1 + x2) / (n1 + n2)

        z = (p1 - p2) / np.sqrt(pooled_proportion * (1-pooled_proportion) * (1/n1 + 1/n2))

        p = stats.norm.sf(abs(z)) * 2



    
        


    




