using System.Collections.Generic;

namespace WoaW.RnD.DL.NNFramework
{
    using System;
    using System.Linq;

    public enum NeuronType
    {
        Undefined = 0,
        Input = 1,
        Heidien = 2,
        Output = 3
    }

    public class Neuron
    {
        private List<Dendrite> _dendrites;
        private float _value;

        public Neuron()
        {
            ActivationFunction = Functions.Activation;
            DerivativeOfActivationFunction = Functions.Derivative;

            _dendrites = new List<Dendrite>();
            Acson = new Acson(this);
        }
        public Neuron(float value, NeuronType type, string id = null) : this()
        {
            Id = id;
            Value = value;
            Type = type;
        }

        #region properties
        public NeuronType Type { get; private set; }
        public List<Dendrite> Dendrites
        {
            get { return _dendrites; }
            private set { _dendrites = value; }
        }
        public Acson Acson { get; set; }
        public float Error { get; set; }
        public float DeltaWeight { get; set; }
        public string Id { get; set; }
        public Func<float, float> ActivationFunction { get; set; }
        public Func<float, float> DerivativeOfActivationFunction { get; set; }
        public float Value { get { return _value; } set { _value = value; Acson.Value = value; } }
        #endregion

        public void CalcuateValue()
        {
            var sum = CalcuateSumOfWeight(Dendrites);
            Value = ActivationFunction(sum);
            Acson.Value = Value;
        }

        public float CalcuateSumOfWeight(IEnumerable<Dendrite> dendrites)
        {
            if (dendrites.Count() == 0)
                return 0;

            var sum = 0F;
            foreach (var dendrite in dendrites)
            {
                var acson = dendrite.Acson;
                if (acson == null)
                    return 0;
                sum = sum + acson.Value * dendrite.Value;
            }

            return sum;
        }
    }
}
