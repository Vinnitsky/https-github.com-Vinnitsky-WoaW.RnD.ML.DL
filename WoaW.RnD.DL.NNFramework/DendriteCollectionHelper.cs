using System.Collections.Generic;
using System.Linq;

namespace WoaW.RnD.DL.NNFramework
{
    public static class DendriteCollectionHelper
    {
        public static void ConnectedTo(this List<Dendrite> list, params Neuron[] neurons)
        {
            foreach (var neuron in neurons)
            {
                var dendrite = new Dendrite(neuron)
                {
                    Acson = neuron.Acson
                };
                list.Add(dendrite);
            }
        }
        public static List<Dendrite> ConnectedTo(this List<Dendrite> list, Neuron owner, Neuron neuron, float weght)
        {
            var dendrite = new Dendrite(owner) { Value = weght, Acson = neuron.Acson };
            neuron.Acson.Dendrites.Add(dendrite);
            list.Add(dendrite);
            return list;
        }
        public static Dendrite Get(this List<Dendrite> list, Neuron index)
        {
            return list.SingleOrDefault(d => d.Neuron.Id == index.Id);
        }
    }
}
