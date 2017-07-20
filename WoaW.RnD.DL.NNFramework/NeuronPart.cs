using System;
using System.Collections.Generic;
using System.Linq;

namespace WoaW.RnD.DL.NNFramework
{
    public abstract class NeuronPart
    {
        public NeuronPart()
        {

        }
        public NeuronPart(Neuron neuron)
        {
            Neuron = neuron;
        }
        public Neuron Neuron { get; set; }
        public virtual float Value { get; set; }
    }
    public class Acson : NeuronPart
    {
        public Acson()
        {
            Dendrites = new List<Dendrite>();
        }
        public Acson(Neuron neuron) : base(neuron)
        {
            Dendrites = new List<Dendrite>();
        }

        public List<Dendrite> Dendrites { get; set; }

        public Neuron this[Dendrite index]
        {
            get { return Dendrites.SingleOrDefault(d=>d == index).Neuron; }
            set { throw new NotImplementedException();/* set the specified index to value here */ }
        }

        //public override double Value { get => base.Value; set => base.Value = value; }
    }

    public class Dendrite : NeuronPart
    {
        private Acson _acson;
        public Dendrite()
        {

        }
        public Dendrite(Neuron neuron) : base(neuron)
        {

        }
        public Dendrite WithWeight(float weght)
        {
            base.Value = weght;
            return this;
        }

        public Acson Acson { get { return _acson; } set { _acson = value; /*_acson.Dendrites.Add(this);*/ } }
    }
}
