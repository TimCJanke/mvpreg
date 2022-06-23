The MVPReg package.

The MVPReg (MultiVariate Probabilistic Regression) package offers a range of models to model multivariate predictive distributions.
Classic regression models only predict the expectation E(P(Y_d|X=x)), probabilistic regression and forecasting models model P(Y_d|X=x), i.e., they represent univariate conditional distributions.
Our goal is to model the full joint conditional distribution P(Y|X=x) with Y = [Y_1, ... Y_D]^T.
For example, y could be the wind power generation from D spatially distributed wind power plants based on the wind speed and direction forecast x.
The dimensions could also relate to temporal dimensions, e.g., the electricity demand of a region over the next 24 hours.

MVPReg offer mainly 2 type of model classes:
 - Conditonal Univariate Distributions + Copula (CUD+C)
 - Conditonal Generative Models (CGM)

The conditional marginal distributions in CUD+C models are either parametric or quantile regression based.
In both cases the underlying model is a multi-output deep neural network that predicts either the parameter vector or the quantiles of the target distributions.

CGMs model the full joint conditional distribution via samples {y_1,...,y_S}, i.e., the model outputs y_s=g(x,z_s) where Z~N(0,I).

Model API follows scikit learn style: model=mvpregmodel(**kwargs) --> model.fit(x,y) --> y=model.simulate(x).

MVPReg also comes with 3 data sets from the GEFCom 2014 as well as evaluation and plotting functions (proper scoring rules, DM tests, visualizations).

The package is under active developement so there might breaking changes.