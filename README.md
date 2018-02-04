# bokeh-application-for-Feature-Selection
Feature selection using Mutual Information
use "bokeh serve --show main.py" from the command prompt pointing to the directory where FSapp is stored.

The application provides various ways to compute MI, Normalised MI, Adjusted MI

When it comes to continuous variables, python default MI calculation, discretizes varaiable into 10 equal bins.
This app gives you flexibility of discretizing into various levels of equal width bins,
equal percentile bins and binnning using K-means 3 clusters.
