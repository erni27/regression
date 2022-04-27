package matrix

func Inverse(m [][]float64) ([][]float64, error) {
	n := len(m)
	a := make([][]float64, n)
	copy(a, m)

	// p is a permuation matrix.
	p := make([][]float64, n)
	for i := 0; i < n; i++ {
		p[i] = make([]float64, n)
		p[i][i] = 1
	}

	// Decomposition a=plu.
	// l under the main diagonal without the main diagonal.
	// u above the main diagonal with the main diagonal.
	for k := 0; k < n-1; k++ {
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
	return r, nil
}
