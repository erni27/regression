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

```golang
// Creates regression options with:
// 1) Learning rate equals 1e-8.
// 2) Batch gradient descent variant.
// 3) Iterative convergance with number of iterations equal 1000.
opt := options.WithIterativeConvergence(1e-8, options.Batch, 1000)
// Initialize linear regression with gradient descent (numerical approach).
r := linear.WithGradientDescent(opt)
// Create design matrix as a 2D slice.
x := [][]float64{
    {2104, 3},
    {1600, 3},
    {2400, 3},
    {1416, 2},
    {3000, 4},
    {1985, 4},
    {1534, 3},
    {1427, 3},
    {1380, 3},
    {1494, 3},
}
// Create target vector as a slice.
y := []float64{399900, 329900, 369000, 232000, 539900, 299900, 314900, 198999, 212000, 242500}
ctx := context.Background()
// Run linear regression.
m, err := r.Run(ctx, regression.TrainingSet{X: x, Y: y})
if err != nil {
    log.Fatal(err)
}
fmt.Println(m)
acc := m.Accuracy()
fmt.Printf("Accuracy: %f.\n", acc)
coeffs := m.Coefficients()
fmt.Printf("Coefficients: %v.\n", coeffs)
// Do a predicition for a new input feature vector.
in := []float64{2550, 4}
p, err := m.Predict(in)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("For vector %v, predicted value equals %f.\n", in, p)
```

The preceding code:
* Initializes regression options.
* Initializes linear regression with gradient descent and configures it through created options.
* Prepare `TrainingSet`.
* Run linear regression.
* Predict a value for a new vector.

An automatic convergance can be configured through the `WithAutomaticConvergence` factory method from `regression/options` package.

```golang
// Initalize regression options with
// 1) Learning rate equals 1e-8.
// 2) Stochastic gradient descent variant.
// 3) Automatic convergance with threshold equals 1e-6.
opt := options.WithAutomaticConvergence(1e-8, options.Stochastic, 1e-6)
```

Convergance through the automatic convergance test is rarely used in practice since it's really hard to set an appropriate threshold.

`regression/linear` package offers a second way of computing linear regression coefficients  - by solving the normal equation (analytical approach). Basically, to minimize the cost function, it sets its derivatives to zero.

```golang
// Initialize linear regression with normal equation (analytical approach).
r := linear.WithNormalEquation()
// Create design matrix as a 2D slice.
x := [][]float64{
    {2104, 3},
    {1600, 3},
    {2400, 3},
    {1416, 2},
    {3000, 4},
    {1985, 4},
    {1534, 3},
    {1427, 3},
    {1380, 3},
    {1494, 3},
}
// Create target vector as a slice.
y := []float64{399900, 329900, 369000, 232000, 539900, 299900, 314900, 198999, 212000, 242500}
ctx := context.Background()
// Run linear regression.
m, err := r.Run(ctx, regression.TrainingSet{X: x, Y: y})
if err != nil {
    log.Fatal(err)
}
fmt.Println(m)
acc := m.Accuracy()
fmt.Printf("Accuracy: %f.\n", acc)
coeffs := m.Coefficients()
fmt.Printf("Coefficients: %v.\n", coeffs)
// Do a predicition for a new input feature vector.
in := []float64{2550, 4}
p, err := m.Predict(in)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("For vector %v, predicted value equals %f.\n", in, p)
```

The preceding code:
* Initializes linear regression with normal equation.
* Prepare `TrainingSet`.
* Run linear regression.
* Predict a value for a new vector.

With the normal equation, there is no need to choose alpha but it can be slow for a very large number of features. It's caused by computing the matrix inversion under the hood.

## Logistic regression

`regression/logistic`, unlike `regression/linear`, provides only an iterative approach for computing the logistic regression coefficients. It uses known from linear regression gradient descent algorithm. So everything regarding gradient descent from the previous section applies here either.

```golang
// Creates regression options with:
// 1) Learning rate equals 1e-2.
// 2) Batch gradient descent variant.
// 3) Iterative convergance with number of iterations equal 100.
opt := options.WithIterativeConvergence(1e-2, options.Batch, 100)
// Initialize logistic regression with normal equation (analytical approach).
r := logistic.WithGradientDescent(opt)
// Create design matrix as a 2D slice.
x := [][]float64{
    {34, 78},
    {30, 43},
    {35, 72},
    {60, 86},
    {79, 75},
    {45, 56},
    {61, 96},
    {75, 46},
    {76, 87},
    {84, 43},
}
// Create target vector as a slice.
y := []float64{0, 0, 0, 1, 1, 0, 1, 1, 1, 1}
ctx := context.Background()
// Run logistic regression.
m, err := r.Run(ctx, regression.TrainingSet{X: x, Y: y})
if err != nil {
    log.Fatal(err)
}
fmt.Println(m)
acc := m.Accuracy()
fmt.Printf("Accuracy: %f.\n", acc)
coeffs := m.Coefficients()
fmt.Printf("Coefficients: %v.\n", coeffs)
// Do a predicition for a new input feature vector.
in := []float64{52, 88}
p, err := m.Predict(in)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("For vector %v, predicted value equals %d.\n", in, p)
```

The preceding code:
* Initializes regression options.
* Initializes logistic regression with gradient descent and configures it through created options.
* Prepare `TrainingSet`.
* Run logistic regression.
* Predict a value for a new vector.
