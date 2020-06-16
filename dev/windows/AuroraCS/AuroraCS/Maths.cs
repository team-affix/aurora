using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Runtime.Serialization;
using AuroraCS.Sessions;
using AuroraCS.Modeling;
using AuroraCS.Maths;
using System.Runtime.Serialization;
using System.Collections.Generic;
using System.Linq;


namespace AuroraCS
{
    namespace Maths
    {
        [Serializable]
        public class Activation
        {
            public virtual double Activate(double x) { return x; }
            public virtual double Derivative(double x) { return x; }
        }
        [Serializable]
        public class Tanh : Activation
        {
            public override double Activate(double x)
            {
                return System.Math.Tanh(x);
            }
            public override double Derivative(double x)
            {
                return 1 / Math.Pow(Math.Cosh(x), 2);
            }
        }
        [Serializable]
        public class Softmax : Activation
        {
            public override double Activate(double x)
            {
                return 1 / (1 + System.Math.Exp(-x));
            }
            public override double Derivative(double x)
            {
                return (1 - x) * (x);
            }
        }
        [Serializable]
        public class LeakyRelu : Activation
        {
            double M;
            public LeakyRelu(double m)
            {
                this.M = m;
            }
            public override double Activate(double x)
            {
                if(x > 0)
                {
                    return x;
                }
                else
                {
                    return M * x;
                }
            }
            public override double Derivative(double x)
            {
                if(x > 0)
                {
                    return 1;
                }
                else
                {
                    return M;
                }
            }
        }
        [Serializable]
        public class _1D : List<double>
        {
            public _1D()
            {

            }
            public _1D(int a)
            {
                for (int i = 0; i < a; i++)
                {
                    Add(0);
                }
            }
            public _1D(int a, Func<double> construct)
            {
                for (int i = 0; i < a; i++)
                {
                    Add(construct());
                }
            }
            public _1D(int a, _1D distribution, Random random, bool replace)
            {
                for (int i = 0; i < a; i++)
                {
                    int distributionIndex = random.Next(0, distribution.Count);
                    double value = distribution[distributionIndex];

                    Add(value);

                    if (!replace)
                    {
                        distribution.RemoveAt(distributionIndex);
                    }
                }
            }
            public _1D(double minimum, double maximum, double increment)
            {
                for (double x = minimum; x < maximum; x += increment)
                {
                    Add(x);
                }
            }
            public _1D Clone()
            {
                return (_1D)MemberwiseClone();
            }
            public double Sum()
            {
                double Result = 0;
                for (int i = 0; i < Count; i++)
                {
                    Result += base[i];
                }
                return Result;
            }
            public _2D Repeat(int a)
            {
                _2D Result = new _2D(a, Count);
                for (int i = 0; i < a; i++)
                {
                    Result.SetRow(i, Clone());
                }
                return Result;
            }
            public _1D ElementWise(Func<double, double> operation)
            {
                _1D Result = new _1D();
                for (int i = 0; i < Count; i++)
                {
                    Result.Add(operation(base[i]));
                }
                return Result;
            }
            public _1D Concat(_1D x)
            {
                _1D Result = new _1D();
                for (int i = 0; i < base.Count; i++)
                {
                    Result.Add(base[i]);
                }
                for (int i = 0; i < x.Count; i++)
                {
                    Result.Add(x[i]);
                }
                return Result;
            }
            public new _1D GetRange(int startIndex, int count)
            {
                _1D Result = new _1D();
                for (int i = 0; i < count; i++)
                {
                    Result.Add(base[i + startIndex]);
                }
                return Result;
            }
            public static _1D operator ^(_1D x, _1D y)
            {
                if (x.Count != y.Count) { throw new Exception("Invalid dimensions for this operation"); }
                _1D Result = new _1D();
                for (int i = 0; i < x.Count; i++)
                {
                    Result.Add(x[i] * y[i]);
                }
                return Result;
            }
            public static _2D operator ^(_1D x, _2D y)
            {
                _2D Result = new _2D(y.Rows, y.Cols);
                for (int i = 0; i < y.Cols; i++)
                {
                    Result.SetCol(i, x ^ y.GetCol(i));
                }
                return Result;
            }
            public static double operator *(_1D x, _1D y)
            {
                if (x.Count != y.Count) { throw new Exception("Invalid dimensions for this operation"); }
                _1D Result = new _1D();
                for (int i = 0; i < x.Count; i++)
                {
                    Result.Add(x[i] * y[i]);
                }
                return Result.Sum();
            }
            public static _1D operator *(_1D x, _2D y)
            {
                if (x.Count != y.Rows) { throw new Exception("Invalid dimensions for this operation"); }
                _1D Result = new _1D();
                for (int i = 0; i < y.Cols; i++)
                {
                    Result.Add(x * y.GetCol(i));
                }
                return Result;
            }
            public static _1D operator *(double x, _1D y)
            {
                _1D Result = new _1D();
                for (int i = 0; i < y.Count; i++)
                {
                    Result.Add(x * y[i]);
                }
                return Result;
            }
            public static _1D operator +(_1D x, _1D y)
            {
                if (x.Count != y.Count) { throw new Exception("Invalid dimensions for this operation"); }
                _1D Result = new _1D();
                for (int i = 0; i < x.Count; i++)
                {
                    Result.Add(x[i] + y[i]);
                }
                return Result;
            }
            public static _2D operator +(_1D x, _2D y)
            {
                if (x.Count != y.Rows) { throw new Exception("Invalid dimensions for this operation"); }
                _2D Result = new _2D();
                for (int i = 0; i < y.Cols; i++)
                {
                    Result.Add(x + y.GetCol(i));
                }
                return Result;
            }
            public static _1D operator -(_1D x, _1D y)
            {
                if (x.Count != y.Count) { throw new Exception("Invalid dimensions for this operation"); }
                _1D Result = new _1D();
                for (int i = 0; i < x.Count; i++)
                {
                    Result.Add(x[i] - y[i]);
                }
                return Result;
            }
            public static _2D operator -(_1D x, _2D y)
            {
                if (x.Count != y.Rows) { throw new Exception("Invalid dimensions for this operation"); }
                _2D Result = new _2D();
                for (int i = 0; i < y.Cols; i++)
                {
                    Result.Add(x - y.GetCol(i));
                }
                return Result;
            }
            public static _1D operator /(_1D x, double y)
            {
                _1D Result = new _1D();
                for (int i = 0; i < x.Count; i++)
                {
                    Result.Add(x[i] / y);
                }
                return Result;
            }
            public static double operator /(_1D x, _1D y)
            {
                if (x.Count != y.Count) { throw new Exception("Invalid dimensions for this operation"); }
                _1D Result = new _1D();
                for (int i = 0; i < x.Count; i++)
                {
                    Result.Add(x[i] / y[i]);
                }
                return Result.Sum();
            }
            public static _1D operator /(_1D x, _2D y)
            {
                if (x.Count != y.Rows) { throw new Exception("Invalid dimensions for this operation"); }
                _1D Result = new _1D();
                for (int i = 0; i < y.Cols; i++)
                {
                    Result.Add(x / y.GetCol(i));
                }
                return Result;
            }
            public static _1D Parse(List<double> plain)
            {
                _1D Result = new _1D();
                for (int i = 0; i < plain.Count; i++)
                {
                    Result.Add(plain[i]);
                }
                return Result;
            }
            public static explicit operator List<object>(_1D x)
            {
                List<object> Result = new List<object> { };
                for(int i = 0; i < x.Count; i++)
                {
                    Result.Add(x[i]);
                }
                return Result;
            }
            public static explicit operator _1D(List<object> x)
            {
                _1D Result = new _1D { };
                for (int i = 0; i < x.Count; i++)
                {
                    Result.Add((double)x[i]);
                }
                return Result;
            }
        }
        [Serializable]
        public class _2D : List<_1D>
        {
            public int Rows { get { return Count; } }
            public int Cols { get { if (Count > 0) { return base[0].Count; } else { return 0; } } }
            public Size Size { get { return new Size(Rows, Cols); } }
            public _2D()
            {

            }
            public _2D(int rows)
            {
                for (int i = 0; i < rows; i++)
                {
                    Add(new _1D());
                }
            }
            public _2D(int rows, int cols)
            {
                for (int i = 0; i < rows; i++)
                {
                    Add(new _1D(cols));
                }
            }
            public _2D(int rows, Func<_1D> construct)
            {
                for (int i = 0; i < rows; i++)
                {
                    Add(construct());
                }
            }
            public _1D GetRow(int index)
            {
                return base[index];
            }
            public _1D GetCol(int index)
            {
                _1D Result = new _1D();
                for (int i = 0; i < Count; i++)
                {
                    Result.Add(base[i][index]);
                }
                return Result;
            }
            public _2D GetRows(int startIndex, int count)
            {
                _2D Result = new _2D();
                for (int i = startIndex; i < startIndex + count; i++)
                {
                    Result.Add(GetRow(i));
                }
                return Result;
            }
            public _2D GetCols(int startIndex, int count)
            {
                _2D Result = new _2D();
                for (int i = startIndex; i < startIndex + count; i++)
                {
                    Result.Add(GetCol(i));
                }
                return Result;
            }
            public void SetRow(int index, _1D values)
            {
                if (values.Count != Cols) { throw new Exception("Invalid dimensions for this operation"); }
                this[index] = values;
            }
            public void SetCol(int index, _1D values)
            {
                if (values.Count != Rows) { throw new Exception("Invalid dimensions for this operation"); }
                for (int i = 0; i < Rows; i++)
                {
                    base[i][index] = values[i];
                }
            }
            public _1D SumDown()
            {
                _1D Result = new _1D(Cols);
                for (int i = 0; i < Count; i++)
                {
                    Result += base[i];
                }
                return Result;
            }
            public _1D SumAcross()
            {
                _1D Result = new _1D(Rows);
                for (int i = 0; i < Cols; i++)
                {
                    Result += GetCol(i);
                }
                return Result;
            }
            public _2D Flip()
            {
                _2D Result = new _2D(Cols, Rows);
                for (int i = 0; i < Rows; i++)
                {
                    Result.SetCol(i, base[i]);
                }
                return Result;
            }
            public _2D ElementWise(Func<_1D, _1D> operation)
            {
                _2D Result = new _2D(Rows, Cols);
                for (int i = 0; i < Rows; i++)
                {
                    Result.SetRow(i, operation(GetRow(i)));
                }
                return Result;
            }
            public _2D ElementWise(Func<double, double> operation)
            {
                _2D Result = new _2D(Rows, Cols);
                for (int i = 0; i < Count; i++)
                {
                    Result.SetRow(i, GetRow(i).ElementWise(operation));
                }
                return Result;
            }
            public static _2D operator ^(_2D x, _2D y)
            {
                if (!x.Size.Equals(y.Size)) { throw new Exception("Invalid dimensions for this operation"); }
                _2D Result = new _2D(x.Rows, x.Cols);
                for (int i = 0; i < x.Rows; i++)
                {
                    Result.SetRow(i, x.GetRow(i) ^ y.GetRow(i));
                }
                return Result;
            }
            public static _2D operator *(double x, _2D y)
            {
                _2D Result = new _2D(y.Rows, y.Cols);
                for (int i = 0; i < y.Rows; i++)
                {
                    Result.SetRow(i, x * y.GetRow(i));
                }
                return Result;
            }
            public static _2D operator *(_2D x, _2D y)
            {
                if (x.Cols != y.Rows) { throw new Exception("Invalid dimensions for this operation"); }
                _2D Result = new _2D(x.Rows, () => { return new _1D(y.Cols); });
                for (int i = 0; i < x.Rows; i++)
                {
                    for (int j = 0; j < y.Cols; j++)
                    {
                        _1D row = x.GetRow(i);
                        _1D col = y.GetCol(j);
                        Result[i][j] = row * col;
                    }
                }
                return Result;
            }
            public static _2D operator /(_2D x, double y)
            {
                _2D Result = new _2D(x.Rows, x.Cols);
                for (int i = 0; i < x.Rows; i++)
                {
                    Result.SetRow(i, x.GetRow(i) / y);
                }
                return Result;
            }
            public static _2D operator /(_2D x, _2D y)
            {
                if (x.Cols != y.Rows) { throw new Exception("Invalid dimensions for this operation"); }
                _2D Result = new _2D(x.Rows, () => { return new _1D(y.Cols); });
                for (int i = 0; i < x.Rows; i++)
                {
                    for (int j = 0; j < y.Cols; j++)
                    {
                        _1D row = x.GetRow(i);
                        _1D col = y.GetCol(j);
                        Result[i][j] = row / col;
                    }
                }
                return Result;
            }
            public static _2D operator +(_2D x, _2D y)
            {
                if (!x.Size.Equals(y.Size)) { throw new Exception("Invalid dimensions for this operation"); }
                _2D Result = new _2D(x.Rows, () => { return new _1D(x.Cols); });
                for (int i = 0; i < x.Rows; i++)
                {
                    Result[i] = x[i] + y[i];
                }
                return Result;
            }
            public static _2D operator -(_2D x, _2D y)
            {
                if (!x.Size.Equals(y.Size)) { throw new Exception("Invalid dimensions for this operation"); }
                _2D Result = new _2D(x.Rows, () => { return new _1D(x.Cols); });
                for (int i = 0; i < x.Rows; i++)
                {
                    Result[i] = x[i] - y[i];
                }
                return Result;
            }
            public static explicit operator List<object>(_2D x)
            {
                List<object> Result = new List<object> { };
                for (int i = 0; i < x.Count; i++)
                {
                    Result.Add(x[i]);
                }
                return Result;
            }
            public static explicit operator _2D(List<object> x)
            {
                _2D Result = new _2D { };
                for (int i = 0; i < x.Count; i++)
                {
                    Result.Add((_1D)x[i]);
                }
                return Result;
            }
        }
        public static class Defaults
        {
            public static _1D MakeDistribution(double minimum, double maximum, double incrementSize, int numberRepeats)
            {
                _1D Result = new _1D();
                for (double x = minimum; x < maximum; x += incrementSize)
                {
                    for (int i = 0; i < numberRepeats; i++)
                    {
                        Result.Add(x);
                    }
                }
                return Result;
            }
        }
    }
}