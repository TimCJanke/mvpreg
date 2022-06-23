# The MVPReg package

The **MVPReg** (*M*ulti*v*ariate *P*robabilistic *Reg*ression) package offers a range of models to model multivariate predictive distributions.
Classic regression models only predict the expectation $E[P(Y_d|X=x)]$, probabilistic regression and forecasting models model $P(Y_d|X=x)$, i.e., they represent univariate conditional distributions.
Our goal is to model the full joint conditional distribution $P(Y|X=x)$ with $Y = [Y_1, ... Y_D]^T$.
For example, y could be the wind power generation from D spatially distributed wind power plants based on the wind speed and direction forecast x.
The dimensions could also relate to temporal dimensions, e.g., the electricity demand of a region over the next 24 hours.

MVPReg offers two approaches:
 - Conditional Univariate Distributions + Copula (CUD+C)
 - Conditional Generative Models (CGM)

The conditional marginal distributions in the CUD+C approach are either parametric or quantile regression based.
In both cases the underlying model is a multi-output deep neural network that predicts either the conditional parameter vector or the conditional quantiles of the target distributions.

The CGM approach represents the full joint conditional distribution via samples $\{y^1,...,y^S\}$, i.e., the model outputs $y^s=g(x,z^s)$ where $Z \sim N(0,I)$.

Model API follows scikit learn style: ``model=mvpregmodel(**kwargs)`` --> ``model.fit(x,y)`` --> ``y=model.simulate(x)``.

MVPReg also comes with 3 data sets from the GEFCom 2014 as well as evaluation and plotting functionionality (proper scoring rules, DM tests, visualizations).

The package is under active developement so there might breaking changes.