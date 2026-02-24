
using System.Collections.Generic;

namespace SynthAi.Neural
{
    public class NeuralEngine
    {
        private readonly IGraphBuilder _graph;
        private readonly IFeatureBuilder _features;
        private readonly IFilmModulator _film;
        private readonly IReadouts _readouts;

        public NeuralEngine(IGraphBuilder graph, IFeatureBuilder features, IFilmModulator film, IReadouts readouts)
        {
            _graph = graph;
            _features = features;
            _film = film;
            _readouts = readouts;
        }

        public ReadoutResult Run(GraphConfig config, IEnumerable<Placement> placements, SunVector sun)
        {
            var g = _graph.Build(config, placements);
            var baseF = _features.Build(g, placements, sun);
            var modF = _film.Apply(baseF, sun);
            return _readouts.Compute(g, modF);
        }
    }
}
