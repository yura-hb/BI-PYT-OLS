## In the beginning

Enter the repository folder and install environment with the next commands:
```
conda env update
source activate semestral_work
```

## Repository structure

* `~/OLS` - folder, which contains the main class for the OLS
* `~/datasets` - folder with the *Boston dataset*
* `~/message_tex` - `TeX` representation of the enter message
* `~/videos` - support animation videos, which shows learning process of different features from the *Boston dataset*
* `EDA.ipynb` - EDA analysis and OLS visualizations notebook

## Opening the repository

The `semestral_work` environment adds [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/), which provides better performance with the matplotlib 3d visualizations. Despite this jupyter notebooks also work well. So to start notebook:

```
jupyter lab or jupyter notebook
```

## OLS class
OLS is the core class in the repository, which implements 4 different approaches to perform linear regression. (Statistical Linear Regression, Gradient Descent, Statistical Gradient Descent and Minibatch Gradient Descent). 
Basic pipeline of the model is next:

```
model = OLS(some_config_params)
model.fit(features, target)
model.predict(test)
model.score(target, prediction)
```

To get more reference about the module creation check the EDA notebook methods `create_gd_model` and `create_linear_model` and the documentation from the class implementation
