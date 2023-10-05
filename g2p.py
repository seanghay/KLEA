"""
Guess word pronunciations using a Phonetisaurus FST

See bin/fst2npz.py to convert an FST to a numpy graph.

Reference: 
  https://github.com/rhasspy/gruut/blob/master/gruut/g2p_phonetisaurus.py
"""
import typing
from collections import defaultdict
from pathlib import Path
import numpy as np

NUMPY_GRAPH = typing.Dict[str, np.ndarray]
_NOT_FINAL = object()

class PhonetisaurusGraph:
    """Graph of numpy arrays that represents a Phonetisaurus FST

    Also contains shared cache of edges and final state probabilities.
    These caches are necessary to ensure that the .npz file stays small and fast
    to load.
    """

    def __init__(self, graph: NUMPY_GRAPH, preload: bool = False):
        self.graph = graph

        self.start_node = int(self.graph["start_node"].item())

        # edge_index -> (from_node, to_node, ilabel, olabel)
        self.edges = self.graph["edges"]
        self.edge_probs = self.graph["edge_probs"]

        # int -> [str]
        self.symbols = []
        for symbol_str in self.graph["symbols"]:
            symbol_list = symbol_str.replace("_", "").split("|")
            self.symbols.append((len(symbol_list), symbol_list))

        # nodes that are accepting states
        self.final_nodes = self.graph["final_nodes"]

        # node -> probability
        self.final_probs = self.graph["final_probs"]

        # Cache
        self.preloaded = preload
        self.out_edges: typing.Dict[int, typing.List[int]] = defaultdict(list)
        self.final_node_probs: typing.Dict[int, typing.Any] = {}

        if preload:
            # Load out edges
            for edge_idx, (from_node, *_) in enumerate(self.edges):
                self.out_edges[from_node].append(edge_idx)

            # Load final probabilities
            self.final_node_probs.update(zip(self.final_nodes, self.final_probs))

    @staticmethod
    def load(graph_path: typing.Union[str, Path], **kwargs) -> "PhonetisaurusGraph":
        """Load .npz file with numpy graph"""
        np_graph = np.load(graph_path, allow_pickle=True)
        return PhonetisaurusGraph(np_graph, **kwargs)

    def g2p_one(
        self,
        word: typing.Union[str, typing.Sequence[str]],
        eps: str = "<eps>",
        beam: int = 5000,
        min_beam: int = 100,
        beam_scale: float = 0.6,
        grapheme_separator: str = "",
        max_guesses: int = 1,
    ) -> typing.Iterable[typing.Tuple[typing.Sequence[str], typing.Sequence[str]]]:
        """Guess phonemes for word"""
        current_beam = beam
        graphemes: typing.Sequence[str] = []

        if isinstance(word, str):
            word = word.strip()

            if grapheme_separator:
                graphemes = word.split(grapheme_separator)
            else:
                graphemes = list(word)
        else:
            graphemes = word

        if not graphemes:
            return []

        # (prob, node, graphemes, phonemes, final, beam)
        q: typing.List[
            typing.Tuple[
                float,
                typing.Optional[int],
                typing.Sequence[str],
                typing.List[str],
                bool,
            ]
        ] = [(0.0, self.start_node, graphemes, [], False)]

        q_next: typing.List[
            typing.Tuple[
                float,
                typing.Optional[int],
                typing.Sequence[str],
                typing.List[str],
                bool,
            ]
        ] = []

        # (prob, phonemes)
        best_heap: typing.List[typing.Tuple[float, typing.Sequence[str]]] = []

        # Avoid duplicate guesses
        guessed_phonemes: typing.Set[typing.Tuple[str, ...]] = set()

        while q:
            done_with_word = False
            q_next = []

            for prob, node, next_graphemes, output, is_final in q:
                if is_final:
                    # Complete guess
                    phonemes = tuple(output)
                    if phonemes not in guessed_phonemes:
                        best_heap.append((prob, phonemes))
                        guessed_phonemes.add(phonemes)

                    if len(best_heap) >= max_guesses:
                        done_with_word = True
                        break

                    continue

                assert node is not None

                if not next_graphemes:
                    if self.preloaded:
                        final_prob = self.final_node_probs.get(node, _NOT_FINAL)
                    else:
                        final_prob = self.final_node_probs.get(node)
                        if final_prob is None:
                            final_idx = int(np.searchsorted(self.final_nodes, node))
                            if self.final_nodes[final_idx] == node:
                                # Cache
                                final_prob = float(self.final_probs[final_idx])
                                self.final_node_probs[node] = final_prob
                            else:
                                # Not a final state
                                final_prob = _NOT_FINAL
                                self.final_node_probs[node] = final_prob

                    if final_prob != _NOT_FINAL:
                        final_prob = typing.cast(float, final_prob)
                        q_next.append((prob + final_prob, None, [], output, True))

                len_next_graphemes = len(next_graphemes)
                if self.preloaded:
                    # Was pre-loaded in __init__
                    edge_idxs = self.out_edges[node]
                else:
                    # Build cache during search
                    maybe_edge_idxs = self.out_edges.get(node)
                    if maybe_edge_idxs is None:
                        edge_idx = int(np.searchsorted(self.edges[:, 0], node))
                        edge_idxs = []
                        while self.edges[edge_idx][0] == node:
                            edge_idxs.append(edge_idx)
                            edge_idx += 1

                        # Cache
                        self.out_edges[node] = edge_idxs
                    else:
                        edge_idxs = maybe_edge_idxs

                for edge_idx in edge_idxs:
                    _, to_node, ilabel_idx, olabel_idx = self.edges[edge_idx]
                    out_prob = self.edge_probs[edge_idx]

                    len_igraphemes, igraphemes = self.symbols[ilabel_idx]

                    if len_igraphemes > len_next_graphemes:
                        continue

                    if igraphemes == [eps]:
                        item = (prob + out_prob, to_node, next_graphemes, output, False)
                        q_next.append(item)
                    else:
                        sub_graphemes = next_graphemes[:len_igraphemes]
                        if igraphemes == sub_graphemes:
                            _, olabel = self.symbols[olabel_idx]
                            item = (
                                prob + out_prob,
                                to_node,
                                next_graphemes[len(sub_graphemes) :],
                                output + olabel,
                                False,
                            )
                            q_next.append(item)

            if done_with_word:
                break

            q_next = sorted(q_next, key=lambda item: item[0])[:current_beam]
            q = q_next

            current_beam = max(min_beam, (int(current_beam * beam_scale)))

        # Yield guesses
        if best_heap:
            for _, guess_phonemes in sorted(best_heap, key=lambda item: item[0])[
                :max_guesses
            ]:
                yield [p for p in guess_phonemes if p]
        else:
            # No guesses
            yield []