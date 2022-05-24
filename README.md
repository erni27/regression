# regression

[![GitHub Workflow Status]https://img.shields.io/github/workflow/status/erni27/regression/CI?style=flat-square)](https://github.com/erni27/regression/actions?query=workflow%3ACI
[![Go Report Card](https://goreportcard.com/badge/github.com/erni27/regression?style=flat-square)](https://goreportcard.com/report/github.com/erni27/regression)
![Go Version](https://img.shields.io/badge/go%20version-%3E=1.18-61CFDD.svg?style=flat-square)

`regression` is a simple, written from scratch Go library for basic variants of two the most popular models from the Generalized Linear Models (GLM) family.

`regression/linear` provides implementation of the linear regression model.

`regression/logistic` provides implementation of the logistic regression model.

## Install

```shell
go get github.com/erni27/regression
```

## Why?

Does the world need another not fancy machine learning library? Actually, no. The main puropose of `regression` library is learning. Machine learning and AI algorithms remains mystified and magic. So here it is a simple, written from scratch implementation of one of the most popular algorithms solving regression and classification problem. It doesn't throw the responsibility for underyling math (like matrix calculus and iterative optimisation) to the external packages. Everything is embedded in this repository.

## Linear regression

`regression/linear` package provides two ways of computing the linear regression coefficients.

The first one uses an itertaive approach - gradient descent algorithm. `Options` from `regression/options` package allows to configure algorithm parameters. What can be configured are listed below:

* Learning rate - determines the size of each step taken by gradient descent
* Gradient descent variant - determines the gradient descent variant (batch or stochastic).
* Convergence type - determines the convergence type (iterative or automatic). An iterative convergence means that gradient descent will run excatly `i`. On the other hand, an automatic convergence declares convergence if a cost function decreseas less than `t` in one iteration.

```
```

## Logistic regression

`regression/logistic`