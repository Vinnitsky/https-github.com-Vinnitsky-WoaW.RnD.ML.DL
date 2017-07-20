namespace WoaW.RnD.DL.NNFramework
{
    public class Functions
    {
        public static float Activation(float x)
        {
            //return (1 / (1 + System.Math.Pow(System.Math.E, -x)));
            return 1.0f / (1.0f + (float)System.Math.Exp(-x));
        }
        public static float Derivative(float x)
        {
            return (x * (1 - x));
            //return (Activation(x) * (1 - Activation(x)));
        }
    }
}
