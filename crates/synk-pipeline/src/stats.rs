//! The paired t-test behind the verdict.
//!
//! The unit is a document: its queries share one text, so they are not
//! independent. Each document contributes one number, the mean change in its
//! queries' nDCG@10, and the test asks whether the mean change differs from 0.

/// Result of a paired t-test on per-unit differences.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairedTest {
    pub n: usize,
    pub mean: f64,
    /// 95% confidence interval for the mean difference.
    pub ci_low: f64,
    pub ci_high: f64,
    /// Two-sided p-value for "the mean difference is 0".
    pub p_value: f64,
    /// The smallest true difference this test would detect 80% of the time.
    pub min_detectable: f64,
}

/// `None` with fewer than two differences, where no interval exists.
pub fn paired_t(diffs: &[f64]) -> Option<PairedTest> {
    let n = diffs.len();
    if n < 2 {
        return None;
    }
    let nf = n as f64;
    let mean = diffs.iter().sum::<f64>() / nf;
    let var = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (nf - 1.0);
    let se = (var / nf).sqrt();
    let df = nf - 1.0;
    let t95 = t_quantile(0.05, df);
    if se == 0.0 {
        return Some(PairedTest {
            n,
            mean,
            ci_low: mean,
            ci_high: mean,
            p_value: if mean == 0.0 { 1.0 } else { 0.0 },
            min_detectable: 0.0,
        });
    }
    Some(PairedTest {
        n,
        mean,
        ci_low: mean - t95 * se,
        ci_high: mean + t95 * se,
        p_value: two_sided_p(mean / se, df),
        min_detectable: (t95 + t_quantile(0.4, df)) * se,
    })
}

/// Two-sided p-value of Student's t with `df` degrees of freedom.
pub fn two_sided_p(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    incomplete_beta(df / 2.0, 0.5, x).clamp(0.0, 1.0)
}

/// The `t > 0` whose two-sided p-value is `alpha`.
pub fn t_quantile(alpha: f64, df: f64) -> f64 {
    let (mut lo, mut hi) = (0.0, 1e4);
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        if two_sided_p(mid, df) > alpha {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

/// The regularized incomplete beta function I_x(a, b).
fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front =
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    // The continued fraction converges fast on this side; use symmetry otherwise.
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_fraction(a, b, x) / a
    } else {
        1.0 - front * beta_fraction(b, a, 1.0 - x) / b
    }
}

/// Lentz's method for the incomplete beta continued fraction.
fn beta_fraction(a: f64, b: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..300 {
        let m = f64::from(m);
        let m2 = 2.0 * m;
        for aa in [
            m * (b - m) * x / ((a + m2 - 1.0) * (a + m2)),
            -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0)),
        ] {
            d = 1.0 + aa * d;
            if d.abs() < TINY {
                d = TINY;
            }
            c = 1.0 + aa / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            h *= d * c;
        }
        if (d * c - 1.0).abs() < 1e-15 {
            break;
        }
    }
    h
}

/// ln Γ(x) for x > 0 (Lanczos approximation).
fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.001_208_650_973_866_179,
        -0.000_005_395_239_384_953,
    ];
    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();
    let mut ser = 1.000_000_000_190_015;
    for (i, g) in G.iter().enumerate() {
        ser += g / (x + 1.0 + i as f64);
    }
    -tmp + (2.506_628_274_631_000_5 * ser / x).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn t_distribution_matches_tables() {
        // Critical values of a two-sided 95% test.
        assert!(close(t_quantile(0.05, 1.0), 12.706, 1e-3));
        assert!(close(t_quantile(0.05, 9.0), 2.262, 1e-3));
        assert!(close(t_quantile(0.05, 100.0), 1.984, 1e-3));
        assert!(close(two_sided_p(2.262, 9.0), 0.05, 1e-4));
        assert!(close(two_sided_p(0.0, 5.0), 1.0, 1e-9));
    }

    #[test]
    fn paired_test_on_known_data() {
        // mean 0.1, sd 0.0816, n 4: t = 2.449, df 3, p ≈ 0.0917.
        let test = paired_t(&[0.0, 0.1, 0.1, 0.2]).unwrap();
        let sd = (0.02f64 / 3.0).sqrt();
        assert!(close(test.mean, 0.1, 1e-12));
        assert!(close(test.p_value, 0.0917, 1e-3));
        assert!(close(test.ci_low, 0.1 - 3.182 * sd / 2.0, 1e-3));
        assert!(test.ci_low < 0.0 && test.ci_high > 0.0);

        let clear = paired_t(&[0.1, 0.12, 0.09, 0.11, 0.1, 0.13]).unwrap();
        assert!(clear.ci_low > 0.0 && clear.p_value < 0.001);

        assert!(paired_t(&[0.5]).is_none());
        let constant = paired_t(&[0.2, 0.2]).unwrap();
        assert_eq!((constant.ci_low, constant.p_value), (0.2, 0.0));
    }
}
