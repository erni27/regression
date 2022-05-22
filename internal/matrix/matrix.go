package matrix

import (
	"context"
	"errors"
)

var (
	ErrNonInvertibleMatrix = errors.New("matrix is not invertible")
	ErrOperationNotAllowed = errors.New("operation not allowed")
	ErrInvalidMatrix       = errors.New("invalid matrix")
)

// Inverse performs matrix inversion.
func Inverse(ctx context.Context, m [][]float64) ([][]float64, error) {
	if !IsRegular(m) {
		return nil, ErrInvalidMatrix
	}
	n := len(m)
	if n != len(m[0]) {
		return nil, ErrNonInvertibleMatrix
	}

	a := make([][]float64, n)
	copy(a, m)

	// p is a permuation matrix.
	p := make([][]float64, n)
	for i := 0; i < n; i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
			p[i] = make([]float64, n)
			p[i][i] = 1
		}
	}

	// Decomposition a=plu.
	// l under the main diagonal without the main diagonal.
	// u above the main diagonal with the main diagonal.
	for k := 0; k < n-1; k++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
			// Pivoting.
			for i := k; i < n-1; i++ {
				if a[k][k] != 0 {
					break
				}
				a[k], a[k+1] = a[k+1], a[k]
				p[k], p[k+1] = p[k+1], p[k]
			}
			// Decomposition.
			for i := k + 1; i < n; i++ {
				a[i][k] /= a[k][k]
				for j := k + 1; j < n; j++ {
					a[i][j] -= a[i][k] * a[k][j]
				}
			}
		}
	}

	r := make([][]float64, n)
	for i := 0; i < n; i++ {
		r[i] = make([]float64, n)
	}

	// b is a column vector of the permuted identity matrix.
	b := make([]float64, n)
	// x is a column vector of the inversed matrix.
	x := make([]float64, n)
	// Solving n sets of the equation ax=b.
	for i := 0; i < n; i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
			// Assign the column vector.
			for j := 0; j < n; j++ {
				if i == j {
					b[j] = p[i][j]
				} else {
					b[j] = p[i][j]
				}
			}
			// Forward substition.
			x[0] = b[0]
			for k := 1; k < n; k++ {
				var s float64
				for j := 0; j < k; j++ {
					s += a[k][j] * x[j]
				}
				// Dividing by l[i][i] omitted since l[i][i] is zero for every i.
				x[k] = b[k] - s
			}
			// Back substition.
			x[n-1] /= a[n-1][n-1]
			for k := n - 2; k >= 0; k-- {
				var s float64
				for j := k + 1; j < n; j++ {
					s += a[k][j] * x[j]
				}
				x[k] = (x[k] - s) / a[k][k]
			}
			// Write a column vector into the result.
			for j := 0; j < n; j++ {
				r[j][i] = x[j]
			}
		}
	}
	return r, nil
}

// Multiply produces matrix product z=xy.
func Multiply(ctx context.Context, x [][]float64, y [][]float64) ([][]float64, error) {
	if !IsRegular(x) || !IsRegular(y) {
		return nil, ErrInvalidMatrix
	}
	m, n, p := len(x), len(y), len(y[0])
	// The numbers of rows in x matrix must be equal to the number of columns in y matrix.
	if len(x[0]) != n {
		return nil, ErrOperationNotAllowed
	}
	z := make([][]float64, m)
	// Calculate a matrix-matrix product.
	for i := 0; i < m; i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
			z[i] = make([]float64, p)
			for j := 0; j < p; j++ {
				for k := 0; k < n; k++ {
					z[i][j] += x[i][k] * y[k][j]
				}
			}
		}
	}
	return z, nil
}

// MultiplyByVector multiples a given matrix by a given vector.
func MultiplyByVector(ctx context.Context, x [][]float64, y []float64) ([]float64, error) {
	if !IsRegular(x) {
		return nil, ErrInvalidMatrix
	}
	m, n := len(x), len(y)
	if n != len(x[0]) {
		return nil, ErrOperationNotAllowed
	}
	p := make([]float64, m)
	for i := 0; i < m; i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
			for k := 0; k < n; k++ {
				p[i] += x[i][k] * y[k]
			}
		}
	}
	return p, nil
}

// Transpose performs a matrix transposition.
func Transpose(ctx context.Context, x [][]float64) ([][]float64, error) {
	if !IsRegular(x) {
		return nil, ErrInvalidMatrix
	}
	n, m := len(x), len(x[0])
	t := make([][]float64, m)
	for j := 0; j < m; j++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
			t[j] = make([]float64, n)
			for i := 0; i < n; i++ {
				t[j][i] = x[i][j]
			}
		}
	}
	return t, nil
}

// IsRegular checks if a 2D slice is a non-nil, regular matrix.
func IsRegular(x [][]float64) bool {
	if len(x) == 0 {
		return false
	}
	n := len(x[0])
	for i := 1; i < len(x); i++ {
		if len(x[i]) != n {
			return false
		}
	}
	return true
}
