using System.Numerics;
using System.Globalization;

namespace Schrödinger
{
    public class TimeDependent
    {
        int xRes, timeRes;
        double dt;
        double dxSquared;
        public Complex[,] psi;

        public TimeDependent(int xRes, int timeRes, double dt)
        {
            this.xRes = xRes;
            this.timeRes = timeRes;
            this.dt = dt;
            this.dxSquared = 1.0 / ((xRes - 1.0)*(xRes - 1.0));
        }

        double[] LinSpace()
        {
            double[] x = new double[xRes];
            for (int i = 0; i < xRes; i++)
            {
                x[i] = (double)i / xRes;
            }
            return x;
        }

        static double[] V_gaussian(double[] x, double a = 1e4, double sigma = 0.1, double mu = 0.5)
        {
            double[] y = new double[x.Length];
            double shifted;
            for (int i = 0; i < x.Length; i++)
            {
                shifted = x[i] - mu;
                y[i] = -a * Math.Exp(-shifted * shifted / (2 * sigma * sigma));
            }
            return y;
        }

        static Complex[] Psi0Sin(double[] x, double n = 1)
        {
            double sum = 0;
            Complex[] y = new Complex[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                y[i] = new(Math.Sqrt(2) * Math.Sin(n * Math.PI * x[i]), 0.0);
                sum += y[i].Magnitude;
            }
            
            for (int i = 0; i < y.Length; i++) y[i] /= sum;
            
            return y;
        }

        static void Normalize(Complex[] x)
        {
            double sum = 0;
            for (int j = 0; j < x.Length; j++) sum += x[j].Magnitude;
            for (int j = 0; j < x.Length; j++) x[j] /= sum;
        }

        public void FDM(int frames)
        {
            int every = timeRes / frames;
            double[] x = LinSpace();
            double[] potential = V_gaussian(x);
            Complex[] psi0 = Psi0Sin(x);
            Console.WriteLine(dt / dxSquared);

            psi = new Complex[frames, xRes];
            
            Complex psiLeft, psiRight;
            Complex[] mCurrent = psi0;
            Complex[] mNext = new Complex[xRes];

            for (int m = 0; m < frames * every - 1; m++) {
                Console.Write("\rFDM " + (100 * (double)m/ (frames * every)).ToString("0.00"));
                for (int j = 0; j < xRes; j++)
                {
                    psiLeft = j == 0 ? Complex.Zero : mCurrent[j - 1];
                    psiRight = j == xRes - 1 ? Complex.Zero : mCurrent[j + 1];
                    mNext[j] = mCurrent[j] + Complex.ImaginaryOne / 2 * dt / dxSquared * (psiLeft - 2 * mCurrent[j] + psiRight) - Complex.ImaginaryOne * potential[j] * dt * mCurrent[j];
                    if (Complex.IsNaN(mNext[j])) Console.WriteLine("NANANANANANANANAAA HELP!!!");
                }
                Normalize(mCurrent);
                if (m % every == 0) for (int j = 0; j < xRes; j++) psi[m / every, j] = mCurrent[j];

                mCurrent = mNext;
                mNext = new Complex[xRes];
            }
        }

        public void Save(string filePath)
        {
            using (StreamWriter outfile = new StreamWriter(filePath))
            {
                outfile.WriteLine($"{psi.GetLength(0)},{psi.GetLength(1)}");
                for (int i = 0; i < psi.GetLength(0); i++)
                {
                    for (int j = 0; j < psi.GetLength(1); j++)
                    {
                        outfile.Write($"{Convert.ToString(psi[i, j].Real, CultureInfo.InvariantCulture)},{Convert.ToString(psi[i, j].Imaginary, CultureInfo.InvariantCulture)}");
                        outfile.Write(j == psi.GetLength(1) - 1 ? "\n" : ";");
                    }
                }
            }
        }
    }

    public class DoIt
    {
        static void Main(string[] args)
        {
            TimeDependent td = new(X_SAMPLES, T_SAMPLES, DELTA_T);
            td.FDM(FRAMES_NUM);
            td.Save(TO_PATH);
        }
    }
}
